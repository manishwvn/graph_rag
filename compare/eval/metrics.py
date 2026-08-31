"""Metrics for the Vector vs Graph RAG comparison.

Both systems are scored in one id space (chunk_uid) and with one answer rubric,
so the columns are directly comparable:

* retrieval  -- hit@k / precision@k / recall@k / MRR over ranked `chunk_uid`s.
  Only answerable queries count: a query with no gold chunks used to score a
  free 1.0 for every system, which inflated every headline number.
* answers    -- deterministic `keyword_recall` over the required answer terms,
  plus an LLM judge (same judge, same rubric, both systems) for correctness and
  groundedness, because a lexical-overlap score structurally punishes the graph
  for answering from triples instead of prose.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

ABSTAIN_PATTERNS = (
    "i don't know",
    "i do not know",
    "don't know based on",
    "not in context",
    "insufficient",
    "does not contain",
    "no information",
)


# ------------------------------------------------------------------ retrieval


def hit_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & set(gold) else 0.0


def precision_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if r in set(gold)) / len(top)


def recall_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 0.0
    return sum(1 for g in set(gold) if g in ranked[:k]) / len(set(gold))


def mrr_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    goldset = set(gold)
    for i, r in enumerate(ranked[:k], 1):
        if r in goldset:
            return 1.0 / i
    return 0.0


# -------------------------------------------------------------------- answers


def is_abstention(answer: str) -> bool:
    a = (answer or "").lower()
    return any(p in a for p in ABSTAIN_PATTERNS)


def keyword_recall(answer: str, keywords: list[str]) -> float:
    """Fraction of required answer terms present, matched on word boundaries.

    Replaces the previous exact-substring check, which scored a fully correct
    answer 0.0 whenever it phrased the facts in a different order.
    """
    if not keywords:
        return 0.0
    a = (answer or "").lower()
    hits = 0
    for kw in keywords:
        k = kw.strip().lower()
        if not k:
            continue
        if re.search(rf"(?<!\w){re.escape(k)}(?!\w)", a):
            hits += 1
    return hits / len(keywords)


class Judgement(BaseModel):
    """Structured verdict returned by the shared LLM judge."""

    correct: bool = Field(description="Does the answer state the reference answer's facts?")
    grounded: bool = Field(description="Is every claim in the answer supported by the context?")
    abstained: bool = Field(description="Does the answer decline to answer?")
    reason: str = Field(default="", description="One short sentence")

    model_config = {"extra": "ignore"}


JUDGE_PROMPT = """You grade a retrieval-augmented answer. Be strict and literal.

Question: {question}
Reference answer: {expected}
Context the system was given:
---
{context}
---
System answer: {answer}

Grade:
- correct: true only if the system answer states the reference answer's facts. Wording, ordering and extra correct detail do not matter. If the reference answer is "NOT FOUND", correct is true only when the system declines to answer.
- grounded: true only if every factual claim in the system answer can be traced to the context above. A refusal makes no factual claim, so a refusal is ALWAYS grounded: true.
- abstained: true if the system declined to answer.

Respond with JSON: {{"correct": bool, "grounded": bool, "abstained": bool, "reason": str}}
"""


class JudgeCache:
    """Verdicts keyed by everything the judge sees.

    Judging is ~63% of an eval's token cost (54k of 86k) and is a pure function
    of its inputs, so caching it is free correctness-wise and is what makes
    iterating on retrieval affordable under a 200k tokens/day cap.
    """

    def __init__(self, path: str | Path = "compare/eval/judge_cache.json"):
        self.path = Path(path)
        self.hits = self.misses = 0
        self._data: dict[str, dict] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[judge-cache] ignoring unreadable cache: {e}")

    @staticmethod
    def key(model: str, question: str, expected: str, context: str, answer: str) -> str:
        blob = "\x00".join((model, question, expected, context, answer))
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> Judgement | None:
        hit = self._data.get(key)
        if hit is None:
            self.misses += 1
            return None
        self.hits += 1
        return Judgement(**hit)

    def put(self, key: str, judgement: Judgement) -> None:
        self._data[key] = judgement.model_dump()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data), encoding="utf-8")


class AnswerCache:
    """Generated answers keyed by everything that determines them.

    Unlike the judge cache, an answer depends on the *indexes* as well as the
    question, so the key carries a fingerprint of the graph file and the vector
    collection. Change either and every entry misses, which is what stops a
    stale answer surviving a rebuild.
    """

    def __init__(self, path: str | Path = "compare/eval/answer_cache.json", fingerprint: str = ""):
        self.path = Path(path)
        self.fingerprint = fingerprint
        self.hits = self.misses = 0
        self._data: dict[str, dict] = {}
        if self.path.exists():
            try:
                stored = json.loads(self.path.read_text(encoding="utf-8"))
                # a fingerprint mismatch invalidates the whole file at once
                if stored.get("fingerprint") == fingerprint:
                    self._data = stored.get("entries", {})
                else:
                    print("[answer-cache] indexes changed — starting a fresh cache")
            except Exception as e:
                print(f"[answer-cache] ignoring unreadable cache: {e}")

    def key(self, system: str, question: str, hops: int | None, k: int) -> str:
        blob = "\x00".join((self.fingerprint, system, question, str(hops), str(k)))
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> dict | None:
        hit = self._data.get(key)
        if hit is None:
            self.misses += 1
            return None
        self.hits += 1
        return dict(hit)

    def put(self, key: str, state: dict) -> None:
        self._data[key] = {
            "answer": state.get("answer", ""),
            "context": state.get("context", ""),
            "retrieved_chunk_ids": state.get("retrieved_chunk_ids", []),
            "tokens": state.get("tokens", 0),
            "embed_tokens": state.get("embed_tokens", 0),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"fingerprint": self.fingerprint, "entries": self._data}), encoding="utf-8"
        )


def judge_answer(
    llm,
    question: str,
    expected: str,
    context: str,
    answer: str,
    attempts: int = 3,
    cache: "JudgeCache | None" = None,
    model: str = "",
) -> tuple[Judgement | None, int]:
    """Run the shared judge. Returns (judgement or None, tokens). Never raises.

    A `None` judgement means the judge could not be scored, which is NOT the
    same as a failed answer. Returning a false/false verdict on a judge error
    silently counted provider hiccups as ungrounded answers -- in one run every
    single "ungrounded" row was really a JSON validation failure.
    """
    key = JudgeCache.key(model, question, expected, context, answer) if cache else ""
    if cache:
        cached = cache.get(key)
        if cached is not None:
            return cached, 0

    tokens = 0
    last = "no attempt made"
    for _ in range(max(1, attempts)):
        try:
            structured = llm.with_structured_output(Judgement, method="json_mode", include_raw=True)
            out = structured.invoke(
                JUDGE_PROMPT.format(
                    question=question, expected=expected, context=context[:6000] or "(empty)", answer=answer
                )
            )
            usage = (getattr(out.get("raw"), "response_metadata", {}) or {}).get("token_usage", {}) or {}
            tokens += int(usage.get("total_tokens", 0))
            parsed = out.get("parsed")
            if parsed is not None:
                if cache:
                    cache.put(key, parsed)
                return parsed, tokens
            last = "judge returned unparseable output"
        except Exception as e:  # judging must never take the run down
            last = str(e)
    print(f"    [judge] unscored after {attempts} attempts: {last[:120]}")
    return None, tokens


# ------------------------------------------------------------------ assembly


def compute_all_metrics(query: dict[str, Any], result: dict[str, Any], system: str, k: int) -> dict[str, Any]:
    """One row of the results table for one (query, system) pair."""
    gold = query.get("gold_chunk_ids") or []
    answerable = bool(gold)
    ranked = result.get("retrieved_chunk_ids", [])
    answer = result.get("answer", "")
    judgement: Judgement | None = result["judgement"]

    row: dict[str, Any] = {
        "id": query.get("id"),
        "query": query["question"],
        "type": query.get("type", "unknown"),
        "system": system,
        "answerable": answerable,
        "retrieved_chunk_ids": ranked[:k],
        "gold_chunk_ids": gold,
        "keyword_recall": keyword_recall(answer, query.get("answer_keywords", [])) if answerable else 0.0,
        # None, not 0.0: an unscored judgement must not count as a wrong answer.
        "judge_correct": None if judgement is None else float(judgement.correct),
        "judge_grounded": None if judgement is None else float(judgement.grounded),
        "judge_scored": float(judgement is not None),
        "abstained": float(is_abstention(answer) or (judgement is not None and judgement.abstained)),
        "latency_s": result.get("latency_s", 0.0),
        "llm_tokens": result.get("llm_tokens", 0),
        "embed_tokens": result.get("embed_tokens", 0),
        "answer": answer,
        "judge_reason": "" if judgement is None else judgement.reason,
        "context_len": len(result.get("context", "")),
    }
    if answerable:
        row |= {
            f"hit@{k}": hit_at_k(ranked, gold, k),
            f"precision@{k}": precision_at_k(ranked, gold, k),
            f"recall@{k}": recall_at_k(ranked, gold, k),
            "mrr": mrr_at_k(ranked, gold, k),
        }
    return row


NUMERIC_KEYS = ("hit@", "precision@", "recall@", "mrr", "keyword_recall", "judge_", "latency_s", "llm_tokens", "embed_tokens")


def aggregate(rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    """Aggregate a system's rows. Retrieval means cover answerable queries only."""
    if not rows:
        return {}
    answerable = [r for r in rows if r["answerable"]]
    negative = [r for r in rows if not r["answerable"]]

    def mean(subset, key):
        vals = [r[key] for r in subset if isinstance(r.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else 0.0

    agg = {
        "count": len(rows),
        "answerable_count": len(answerable),
        "negative_count": len(negative),
        f"hit@{k}": mean(answerable, f"hit@{k}"),
        f"precision@{k}": mean(answerable, f"precision@{k}"),
        f"recall@{k}": mean(answerable, f"recall@{k}"),
        "mrr": mean(answerable, "mrr"),
        "keyword_recall": mean(answerable, "keyword_recall"),
        # judge means skip unscored rows; `judge_coverage` says how many that was
        "judge_correct_answerable": mean(answerable, "judge_correct"),
        "judge_grounded": mean(rows, "judge_grounded"),
        "judge_coverage": mean(rows, "judge_scored"),
        "abstention_on_answerable": mean(answerable, "abstained"),
        "abstention_on_negative": mean(negative, "abstained"),
        "context_chars": mean(rows, "context_len"),
        # a cached answer took 0.0s to "produce"; averaging that in would report
        # a latency the system never achieved
        "latency_s": mean([r for r in rows if not r.get("from_cache")], "latency_s"),
        "answers_from_cache": mean(rows, "from_cache"),
        "llm_tokens": mean(rows, "llm_tokens"),
        "embed_tokens": mean(rows, "embed_tokens"),
        "llm_tokens_total": sum(r["llm_tokens"] for r in rows),
        "embed_tokens_total": sum(r["embed_tokens"] for r in rows),
    }
    return agg
