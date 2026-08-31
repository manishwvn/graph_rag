"""Evaluation harness: one retrieval per (query, system), one shared judge.

Both systems are driven through their compiled agent only. The previous version
called the retriever *and* the agent, which retrieved twice per query: double
the API cost, roughly double the reported latency, and metrics computed on a
different retrieval than the one that produced the answer.
"""

from __future__ import annotations

import json
import re
import time
from collections import deque
from pathlib import Path
from typing import Any

from langchain_groq import ChatGroq

from compare.eval.metrics import aggregate, compute_all_metrics, judge_answer
from compare.vector.agent_vector import build_vector_agent
from compare.vector.embed_nvidia import NVIDIAEmbeddings
from compare.vector.retriever_vector import VectorRetriever
from compare.vector.store_chroma import VectorStoreChroma
from graph_rag.agent import build_agent
from graph_rag.config import settings
from graph_rag.store import GraphStore

# Groq free tier, per model. Requests were never the binding limit here:
# a full eval runs at ~25 req/min against an RPM of 30, but at ~10k (generation)
# and ~17k (judge) tokens/min against a TPM of 8000.
RPM_LIMIT = 30
TPM_LIMIT = 8000

QUERIES = "compare/data_large/queries.json"
METRICS = "compare/eval/metrics.json"


class RateLimiter:
    """Client-side pacing against per-minute request and token limits.

    Blowing the token limit is not merely a retry: the retried request still
    consumes a per-day request unit, so a run that nominally needs 128 requests
    can burn the entire 1000/day allowance. Pacing to stay under TPM is what
    keeps the daily budget intact.
    """

    def __init__(self, rpm: int = RPM_LIMIT, tpm: int = TPM_LIMIT, headroom: float = 0.85, sleep=time.sleep, now=time.monotonic):
        self.rpm, self.tpm, self.headroom = rpm, tpm, headroom
        self._sleep, self._now = sleep, now
        self._events: deque[tuple[float, int]] = deque()

    def _prune(self, now: float) -> None:
        while self._events and now - self._events[0][0] >= 60.0:
            self._events.popleft()

    def _usage(self, now: float) -> tuple[int, int]:
        self._prune(now)
        return len(self._events), sum(t for _, t in self._events)

    def wait(self, estimated_tokens: int) -> float:
        """Block until this call fits inside the minute window. Returns seconds slept."""
        slept = 0.0
        while True:
            now = self._now()
            requests, tokens = self._usage(now)
            fits = (
                requests + 1 <= self.rpm * self.headroom
                and tokens + estimated_tokens <= self.tpm * self.headroom
            )
            if fits or not self._events:
                return slept
            delay = max(0.05, 60.0 - (now - self._events[0][0]))
            self._sleep(delay)
            slept += delay

    def record(self, tokens: int) -> None:
        self._events.append((self._now(), max(0, int(tokens))))


def retry_after_seconds(message: str) -> float | None:
    """Groq puts the real wait in the 429 body: 'Please try again in 1m26.4s.'"""
    m = re.search(r"try again in\s+(?:(\d+)m)?([\d.]+)s", message)
    if not m:
        return None
    return float(m.group(1) or 0) * 60 + float(m.group(2))


def load_queries(path: str = QUERIES) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class DailyQuotaExhausted(RuntimeError):
    """The provider's per-day request cap is gone. Retrying cannot help today."""


def run_with_retry(func, *args, max_retries: int = 4, base_delay: float = 3.0):
    """Retry a per-minute rate limit; fail fast on a per-day cap."""
    for attempt in range(max_retries):
        try:
            return func(*args)
        except Exception as e:
            msg = str(e).lower()
            rate_limited = "429" in msg or "rate limit" in msg or "rate_limit" in msg
            if rate_limited and ("per day" in msg or "rpd" in msg):
                # Backing off cannot clear a daily cap; surface it immediately
                # rather than burning four sleeps and reporting a generic 429.
                raise DailyQuotaExhausted(
                    "Groq daily request limit reached — rerun `python app_compare.py eval` "
                    "after it resets. Partial results were not written."
                ) from e
            if rate_limited and attempt < max_retries - 1:
                # honour the provider's own number when it gives one: a TPM
                # window needs ~60s, and guessing 3s just burns another request
                delay = retry_after_seconds(msg) or base_delay * (2**attempt)
                print(f"    [retry] rate limited, waiting {delay:.0f}s ({attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
            raise


def run_eval(
    queries_path: str = QUERIES,
    k: int | None = None,
    graph_hops_list: list[int] | None = None,
    output_path: str = METRICS,
    sleep_s: float = 0.5,
    vector_agent: Any = None,
    graph_agent: Any = None,
    judge_llm: Any = None,
    judge: bool = True,
) -> dict[str, Any]:
    """Score both systems over the query set.

    The three agents are injectable so the whole loop can be exercised offline
    with stubs; left as None they are built from settings as usual. `judge=False`
    skips the LLM judge (rows come back unscored) for cheap retrieval-only runs.
    """
    k = k if k is not None else settings.compare_k
    graph_hops_list = graph_hops_list or [1, 2]
    queries = load_queries(queries_path)

    print(f"[eval] k={k} graph_hops={graph_hops_list} queries={len(queries)} judge={judge}")

    injected = vector_agent is not None and graph_agent is not None
    if injected:
        graph_info, vector_info = "injected", "injected"
    else:
        embeddings = NVIDIAEmbeddings()  # shared so the query cache is reused
        vector_store = VectorStoreChroma()
        if vector_store.count() == 0:
            raise RuntimeError("Vector store is empty — run `python app_compare.py build-vector` first")
        vector_agent = build_vector_agent(
            k=k, retriever=VectorRetriever(store=vector_store, embeddings=embeddings, k=k)
        )
        graph_store = GraphStore(settings.graph_large_path)
        graph_store.load()
        graph_agent = build_agent(store=graph_store, k=k)
        graph_info, vector_info = graph_store.stats(), vector_store.count()
        print(f"[eval] graph: {graph_info} | vector: {vector_info} chunks")

    # A judge from the same family as the generator grades its own output style
    # favourably, so the judge model is deliberately different.
    if judge and judge_llm is None:
        judge_llm = ChatGroq(
            model=settings.judge_model, temperature=0, max_tokens=1200, groq_api_key=settings.groq_api_key
        )

    # One limiter per model: generation/extraction and judging bill separately.
    gen_limit, judge_limit = RateLimiter(), RateLimiter()

    systems: list[tuple[str, Any]] = [("vector", None)] + [(f"graph_hops{h}", h) for h in graph_hops_list]
    results: dict[str, list[dict]] = {name: [] for name, _ in systems}

    for i, q in enumerate(queries):
        print(f"\n[eval] {i + 1}/{len(queries)} [{q.get('type')}] {q['question']}")
        for name, hops in systems:
            payload = {"question": q["question"], "k": k}
            agent = vector_agent if hops is None else graph_agent
            if hops is not None:
                payload["hops"] = hops

            # a graph call is two LLM round-trips (entity extraction + answer)
            gen_limit.wait(2200 if hops is not None else 1400)
            t0 = time.time()
            state = run_with_retry(agent.invoke, payload)
            latency = time.time() - t0
            gen_limit.record(state.get("tokens", 0))
            time.sleep(sleep_s)
            if judge:
                judge_limit.wait(1300)
                judgement, judge_tokens = run_with_retry(
                    judge_answer, judge_llm, q["question"], q["expected_answer"],
                    state.get("context", ""), state.get("answer", ""),
                )
            else:
                judgement, judge_tokens = None, 0
            judge_limit.record(judge_tokens)
            result = {
                "retrieved_chunk_ids": state.get("retrieved_chunk_ids", []),
                "context": state.get("context", ""),
                "answer": state.get("answer", ""),
                "latency_s": latency,
                "llm_tokens": state.get("tokens", 0),
                "embed_tokens": state.get("embed_tokens", 0),
                "judgement": judgement,
            }
            row = compute_all_metrics(q, result, name, k)
            row["judge_tokens"] = judge_tokens
            results[name].append(row)
            verdict = "—" if row["judge_correct"] is None else f"{row['judge_correct']:.0f}/{row['judge_grounded']:.0f}"
            print(
                f"    {name:<12} hit@{k}={row.get(f'hit@{k}', float('nan')):.2f} "
                f"kw={row['keyword_recall']:.2f} correct/grounded={verdict} "
                f"{row['latency_s']:.1f}s"
            )
            time.sleep(sleep_s)

    summary = {name: aggregate(rows, k) for name, rows in results.items()}
    output = {
        "config": {
            "k": k,
            "graph_hops": graph_hops_list,
            "max_triples": settings.compare_max_triples,
            "chunk_size": settings.compare_chunk_size,
            "chunk_overlap": settings.compare_chunk_overlap,
            "model": settings.groq_model,
            "judge_model": settings.judge_model if judge else None,
            "embed_model": settings.nvidia_embed_model,
            "queries": len(queries),
            "vector_chunks": vector_info,
            "graph": graph_info,
        },
        "summary": summary,
        "details": results,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\n[eval] saved to {output_path}")
    return output
