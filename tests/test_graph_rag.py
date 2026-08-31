"""Regression tests for the defects the Vector-vs-Graph comparison uncovered.

Every test here pins a bug that silently corrupted results before: dropped
chunk-size settings, filename-keyed checkpoints, position-zipped extractions,
hub-starved multi-hop retrieval and free points for negative queries.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.documents import Document

from compare.eval.metrics import (
    Judgement,
    aggregate,
    compute_all_metrics,
    hit_at_k,
    is_abstention,
    keyword_recall,
    mrr_at_k,
    precision_at_k,
    recall_at_k,
)
from compare.eval.harness import DailyQuotaExhausted, run_eval, run_with_retry
from compare.vector.pipeline_vector import build_vector_store
from graph_rag.ingestion import load_and_split, split_documents
from graph_rag.schemas import ExtractionResult
from graph_rag.store import GraphStore, Triple, normalize_name, normalize_relation

CORPUS = "compare/data_large"


def extraction(entities, relations) -> ExtractionResult:
    return ExtractionResult(
        entities=[{"name": n, "type": t} for n, t in entities],
        relations=[{"source": s, "target": o, "relation": r} for s, r, o in relations],
    )


# --------------------------------------------------------------- ingestion


def test_chunk_size_is_actually_applied():
    """The pipelines computed 800/80 and then split at the 400/40 default."""
    small = load_and_split(CORPUS, 400, 40)
    large = load_and_split(CORPUS, 800, 80)
    assert len(large) < len(small)
    assert max(len(d.page_content) for d in large) > 400


def test_chunk_uid_is_unique_and_stable():
    docs = load_and_split(CORPUS, 800, 80)
    uids = [d.metadata["chunk_uid"] for d in docs]
    assert len(uids) == len(set(uids))
    assert uids[0] == f"{docs[0].metadata['source']}:0"


def test_split_documents_empty_input():
    assert split_documents([]) == []


# ------------------------------------------------------------------- store


def test_parallel_relations_are_not_concatenated():
    """DiGraph collapsed them into 'works_at, teaches_at' on one edge."""
    s = GraphStore()
    s.add_extraction(
        extraction(
            [("Carol", "PERSON")],
            [("Carol", "works_at", "Stanford"), ("Carol", "teaches_at", "Stanford")],
        ),
        "d:0",
    )
    rels = {k for _, _, k in s.graph.edges(keys=True)}
    assert rels == {"works_at", "teaches_at"}


def test_relation_label_variants_merge():
    assert normalize_relation("co-leads") == normalize_relation("Co Leads") == "co_leads"
    s = GraphStore()
    s.add_extraction(extraction([], [("A", "co-leads", "B"), ("A", "co_leads", "B")]), "d:0")
    assert s.graph.number_of_edges() == 1


def test_canonicalize_merges_surface_variants():
    s = GraphStore()
    s.add_extraction(
        extraction(
            [("Carol Zhang", "PERSON"), ("Carol", "PERSON"), ("Stanford's Quantum Initiative", "ORG")],
            [("Carol", "co_leads", "Stanford Quantum Initiative"), ("Carol Zhang", "works_at", "Stanford University")],
        ),
        "d:0",
    )
    mapping = s.canonicalize()
    assert mapping["Carol"] == "Carol Zhang"
    assert normalize_name("Stanford's Quantum Initiative") == normalize_name("Stanford Quantum Initiative")
    assert "Carol" not in s.graph


def test_canonicalize_leaves_ambiguous_stubs_alone():
    """'Stanford' prefixes two different entities, so merging it would guess."""
    s = GraphStore()
    s.add_extraction(
        extraction([("Stanford", "ORG"), ("Stanford University", "ORG"), ("Stanford Quantum Initiative", "ORG")], []),
        "d:0",
    )
    s.canonicalize()
    assert "Stanford" in s.graph


def test_provenance_is_per_chunk_not_per_file():
    docs = [
        Document(page_content="a", metadata={"source": "f.txt", "chunk_id": 0, "chunk_uid": "f.txt:0"}),
        Document(page_content="b", metadata={"source": "f.txt", "chunk_id": 1, "chunk_uid": "f.txt:1"}),
    ]
    s = GraphStore()
    s.build_from_documents(docs, [extraction([("A", "ORG")], []), extraction([("B", "ORG")], [])])
    assert s.graph.nodes["A"]["sources"] == {"f.txt:0"}
    assert s.graph.nodes["B"]["sources"] == {"f.txt:1"}
    assert s.stats()["chunks"] == 2


def test_build_from_documents_rejects_length_mismatch():
    with pytest.raises(ValueError):
        GraphStore().build_from_documents([Document(page_content="a", metadata={"chunk_uid": "f:0"})], [])


def test_json_roundtrip_preserves_multigraph_and_sources(tmp_path):
    s = GraphStore(tmp_path / "g.json")
    s.add_extraction(extraction([("A", "ORG")], [("A", "r1", "B"), ("A", "r2", "B")]), "d:0")
    s.save()
    assert json.loads((tmp_path / "g.json").read_text())["multigraph"] is True
    s2 = GraphStore(tmp_path / "g.json")
    s2.load()
    assert s2.graph.number_of_edges() == 2
    assert s2.graph.nodes["A"]["sources"] == {"d:0"}


def test_load_missing_graph_raises():
    with pytest.raises(FileNotFoundError):
        GraphStore("does/not/exist.json").load()


# --------------------------------------------------------------- retrieval


def _hub_store() -> GraphStore:
    """Seed -> Advisor -> Workplace, with a 40-edge hub hanging off the seed."""
    s = GraphStore()
    rels = [("Advisor", "advises", "Seed"), ("Advisor", "works_at", "Workplace"), ("Seed", "interns_at", "Hub")]
    rels += [("Hub", f"rel_{i}", f"Filler{i:02d}") for i in range(40)]
    s.add_extraction(extraction([], rels), "d:0")
    return s


def test_hub_does_not_starve_the_second_hop():
    """The bug: 30 hop-1 hub edges filled the budget and buried Advisor->Workplace."""
    s = _hub_store()
    triples = s.get_subgraph(["Seed"], hops=2, max_triples=30)
    assert ("Advisor", "works_at", "Workplace") in [(t.source, t.relation, t.target) for t in triples]
    assert len(triples) <= 30


def test_per_node_cap_is_load_bearing():
    """The cap is not redundant with round-robin. On an uneven frontier it
    limits how far a high-degree node keeps supplying edges after the smaller
    nodes are exhausted. Removing it silently changed the retrieved chunks on
    real queries, so it is pinned here."""
    s = GraphStore()
    rels = [("Seed", "near", "Big")] + [("Seed", "near", f"S{i}") for i in range(5)]
    rels += [("Big", f"e{j}", f"F{j:02d}") for j in range(50)]
    rels += [(f"S{i}", "only", f"T{i}") for i in range(5)]
    s.add_extraction(extraction([], rels), "d:0")

    def from_big(cap):
        triples = s.get_subgraph(["Seed"], hops=2, max_triples=20, max_per_node=cap)
        return sum(1 for t in triples if t.hop == 2 and t.source == "Big")

    assert from_big(3) == 3
    assert from_big(50) > 3, "max_per_node had no effect — the cap is being ignored"


def test_more_hops_never_loses_a_closer_fact():
    s = _hub_store()
    got = {h: {(t.source, t.relation, t.target) for t in s.get_subgraph(["Seed"], hops=h, max_triples=30)} for h in (1, 2)}
    assert ("Advisor", "advises", "Seed") in got[1]
    assert ("Advisor", "advises", "Seed") in got[2]


def test_only_traversed_edges_are_returned():
    """Induced-subgraph retrieval also returned edges between two hop-2 nodes."""
    s = GraphStore()
    s.add_extraction(extraction([], [("Seed", "r", "A"), ("Seed", "r", "B"), ("A", "sibling", "B")]), "d:0")
    triples = s.get_subgraph(["Seed"], hops=1, max_triples=30)
    assert ("A", "sibling", "B") not in [(t.source, t.relation, t.target) for t in triples]


def test_query_relevance_reorders_equally_close_edges():
    """`where do they work` should surface works_at over an unrelated relation."""
    s = GraphStore()
    s.add_extraction(extraction([], [("Seed", "advises", "X"), ("Seed", "works_at", "Y")]), "d:0")
    ranked = [t.relation for t in s.get_subgraph(["Seed"], hops=1, max_triples=1, query="where do they work")]
    assert ranked == ["works_at"]


def test_context_budget_is_enforced():
    """Graph context must be capped to the same char budget as the vector context."""
    s = _hub_store()
    triples = s.get_subgraph(["Seed"], hops=2, max_triples=60, max_chars=200)
    rendered = s.triples_as_text(triples)
    assert 0 < len(rendered) <= 200


def test_unmatched_seed_returns_nothing():
    assert _hub_store().get_subgraph(["Nobody"], hops=2) == []


def test_match_seeds_is_case_and_punctuation_insensitive():
    s = GraphStore()
    s.add_extraction(extraction([("Acme Corp", "ORG")], []), "d:0")
    assert s.match_seeds(["acme corp."]) == ["Acme Corp"]


def test_chunk_ids_from_triples_rank_by_triple_order():
    triples = [Triple("A", "r", "B", ("c:1",), 1), Triple("B", "r", "C", ("c:0", "c:1"), 2)]
    assert GraphStore.chunk_ids_from_triples(triples) == ["c:1", "c:0"]
    assert GraphStore.chunk_ids_from_triples(triples, k=1) == ["c:1"]



def test_token_matching_does_not_explode():
    """Raw substring matching let the entity 'quantum' seed every quantum node."""
    from graph_rag.retriever import _token_eq

    assert _token_eq("corp", "corporation")
    assert not _token_eq("the", "theory")
    assert not _token_eq("qua", "quantum")  # below the 4-char floor


# ----------------------------------------------------------------- metrics


def test_retrieval_metrics_are_consistent_at_k():
    ranked, gold = ["a", "b", "c", "d", "e"], ["e"]
    assert hit_at_k(ranked, gold, 4) == 0.0  # 'e' sits outside k
    assert mrr_at_k(ranked, gold, 4) == 0.0  # MRR must respect the same cutoff
    assert recall_at_k(ranked, gold, 5) == 1.0
    assert precision_at_k(ranked, gold, 5) == 0.2


def test_negative_queries_get_no_free_retrieval_credit():
    """Empty gold used to score a vacuous 1.0 for every system."""
    assert hit_at_k(["a"], [], 4) == 0.0
    assert recall_at_k(["a"], [], 4) == 0.0
    q = {"id": "q", "question": "?", "type": "negative", "expected_answer": "NOT FOUND", "gold_chunk_ids": []}
    row = compute_all_metrics(q, _result("I don't know based on the provided context."), "vector", 4)
    assert row["answerable"] is False
    assert "hit@4" not in row
    assert row["abstained"] == 1.0


def test_keyword_recall_beats_exact_substring():
    answer = "Carol Zhang advises Dave Kim, and she works at Stanford University."
    assert keyword_recall(answer, ["Carol Zhang", "Stanford"]) == 1.0
    assert keyword_recall(answer, ["Carol Zhang", "MIT"]) == 0.5
    assert keyword_recall("scan the barcode", ["can"]) == 0.0  # word boundaries


def test_daily_quota_fails_fast_but_minute_limit_retries():
    """A per-day cap cannot be waited out inside a run; a per-minute one can."""
    calls = []

    def daily():
        calls.append(1)
        raise RuntimeError("Error code: 429 ... rate limit ... on requests per day (RPD): Limit 1000")

    with pytest.raises(DailyQuotaExhausted):
        run_with_retry(daily)
    assert len(calls) == 1  # no pointless backoff

    attempts = []

    def minute():
        attempts.append(1)
        if len(attempts) < 2:
            raise RuntimeError("Error code: 429 rate_limit_exceeded requests per minute")
        return "ok"

    assert run_with_retry(minute, base_delay=0.01) == "ok"
    assert len(attempts) == 2


def test_abstention_detection():
    assert is_abstention("I don't know based on the provided context.")
    assert not is_abstention("Alice Chen leads the team.")


def _result(answer: str, ranked=None, correct=False, grounded=True) -> dict:
    return {
        "retrieved_chunk_ids": ranked or [],
        "context": "ctx",
        "answer": answer,
        "latency_s": 1.0,
        "llm_tokens": 10,
        "embed_tokens": 0,
        "judgement": Judgement(correct=correct, grounded=grounded, abstained=False),
    }


def test_unscored_judgement_is_excluded_not_counted_wrong():
    """A judge that errors must not be recorded as a wrong, ungrounded answer.
    In one run every 'ungrounded' row was really a JSON validation failure."""
    k = 4
    q = {"id": "a", "question": "?", "type": "factual_single", "expected_answer": "x",
         "answer_keywords": ["x"], "gold_chunk_ids": ["c:0"]}
    good = compute_all_metrics(q, _result("x", ranked=["c:0"], correct=True), "s", k)
    unscored = dict(_result("x", ranked=["c:0"]))
    unscored["judgement"] = None
    bad = compute_all_metrics(q, unscored, "s", k)

    assert bad["judge_correct"] is None and bad["judge_grounded"] is None
    assert bad["judge_scored"] == 0.0
    agg = aggregate([good, bad], k)
    assert agg["judge_correct_answerable"] == 1.0  # the unscored row is skipped, not averaged as 0
    assert agg["judge_coverage"] == 0.5


def test_aggregate_separates_answerable_from_negative():
    k = 4
    q_ans = {"id": "a", "question": "?", "type": "factual_single", "expected_answer": "x", "answer_keywords": ["x"], "gold_chunk_ids": ["c:0"]}
    q_neg = {"id": "n", "question": "?", "type": "negative", "expected_answer": "NOT FOUND", "gold_chunk_ids": []}
    rows = [
        compute_all_metrics(q_ans, _result("x", ranked=["c:0"], correct=True), "s", k),
        compute_all_metrics(q_neg, _result("I don't know based on the provided context.", correct=True), "s", k),
    ]
    agg = aggregate(rows, k)
    assert agg["answerable_count"] == 1 and agg["negative_count"] == 1
    assert agg[f"hit@{k}"] == 1.0  # averaged over the answerable query only
    assert agg["abstention_on_negative"] == 1.0
    assert agg["abstention_on_answerable"] == 0.0
    assert agg["llm_tokens_total"] == 20


# ------------------------------------------------------------- query set


def test_query_set_gold_labels_exist_in_the_indexed_corpus():
    """Gold ids were hand-written and four pointed at chunks lacking the answer."""
    queries = json.loads(open("compare/data_large/queries.json").read())
    docs = load_and_split(CORPUS, 800, 80)
    text = {d.metadata["chunk_uid"]: d.page_content.lower() for d in docs}
    for q in queries:
        for cid in q["gold_chunk_ids"]:
            assert cid in text, f"{q['id']} references missing chunk {cid}"
        if q["gold_chunk_ids"]:
            for kw in q["answer_keywords"]:
                assert any(kw.lower() in text[c] for c in q["gold_chunk_ids"]), f"{q['id']}: no gold chunk contains {kw!r}"
        else:
            assert q["expected_answer"] == "NOT FOUND"


# ------------------------------------------------- harness, exercised offline


class _StubAgent:
    """Returns a canned state, so the eval loop can run with no API calls."""

    def __init__(self, answer: str, chunks: list[str]):
        self.answer, self.chunks, self.calls = answer, chunks, 0

    def invoke(self, payload):  # noqa: ARG002 — mirrors the compiled agent's signature
        self.calls += 1
        return {
            "answer": self.answer,
            "context": "ctx",
            "retrieved_chunk_ids": self.chunks,
            "tokens": 7,
            "embed_tokens": 0,
        }


def test_run_eval_end_to_end_offline(tmp_path):
    """The whole harness loop, with stubs: one retrieval per (query, system),
    no judge, metrics written. Covers the wiring that needed live API calls."""
    queries = [
        {"id": "q1", "question": "Who?", "expected_answer": "Alice", "type": "factual_single",
         "answer_keywords": ["Alice"], "gold_chunk_ids": ["c:0"]},
        {"id": "q2", "question": "Nothing?", "expected_answer": "NOT FOUND", "type": "negative",
         "answer_keywords": [], "gold_chunk_ids": []},
    ]
    qp = tmp_path / "q.json"
    qp.write_text(json.dumps(queries))
    out = tmp_path / "m.json"

    vec = _StubAgent("Alice leads it.", ["c:0", "c:9"])
    gra = _StubAgent("I don't know based on the provided context.", ["c:9"])
    result = run_eval(
        queries_path=str(qp), k=4, graph_hops_list=[1, 2], output_path=str(out),
        sleep_s=0, vector_agent=vec, graph_agent=gra, judge=False,
    )

    # one agent call per (query, system) -- not two, which was the original bug
    assert vec.calls == 2
    assert gra.calls == 4  # 2 queries x 2 hop settings
    assert out.exists()
    summary = result["summary"]
    assert summary["vector"]["hit@4"] == 1.0
    assert summary["graph_hops1"]["hit@4"] == 0.0
    assert summary["vector"]["judge_coverage"] == 0.0  # judge skipped, nothing invented
    assert summary["graph_hops1"]["abstention_on_negative"] == 1.0
    assert result["config"]["judge_model"] is None


def test_build_vector_store_offline():
    """Vector build with a fake embedding client and a fake store."""

    class FakeEmbeddings:
        api_calls, tokens = 1, 42

        def embed_passages(self, texts):
            return [[float(len(t)), 1.0] for t in texts]

    class FakeStore:
        def __init__(self):
            self.docs = None

        def clear(self):
            pass

        def add_documents(self, docs, embeddings):
            assert len(docs) == len(embeddings)
            self.docs = docs

        def stats(self):
            return {"count": len(self.docs), "size_mb": 0.0}

    store = FakeStore()
    stats = build_vector_store(CORPUS, 800, 80, embeddings=FakeEmbeddings(), store=store)
    assert stats["chunks_indexed"] == stats["chunks_total"] == len(load_and_split(CORPUS, 800, 80))
    assert stats["dim"] == 2
    # ids must be the shared chunk_uid space, not positional indices
    assert store.docs[0].metadata["chunk_uid"].endswith(":0")
