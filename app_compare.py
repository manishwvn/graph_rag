#!/usr/bin/env python3
"""Comparison CLI: build both RAGs over one corpus, query them, evaluate, check quotas."""

import argparse
import json
import sys
import time
from pathlib import Path

from compare.eval.harness import METRICS, QUERIES, RPM_LIMIT, TPM_LIMIT, load_queries, run_eval
from compare.eval.report import BUILD_STATS, generate_report
from compare.graph.pipeline_graph_throttled import build_graph_throttled
from compare.vector.agent_vector import build_vector_agent
from compare.vector.pipeline_vector import build_vector_store
from graph_rag.agent import build_agent
from graph_rag.config import settings
from graph_rag.ingestion import load_and_split
from graph_rag.store import GraphStore

CORPUS = "compare/data_large"


def _save_build_stats(key: str, stats: dict):
    """Both builds append to one file so the report can quote real build costs."""
    p = Path(BUILD_STATS)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if p.exists():
        try:
            existing = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    existing[key] = stats
    p.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"[build] stats -> {p}")


def check_quotas():
    """Estimate the run against the real free-tier limits.

    Requests are not the binding constraint: a full eval runs at roughly 25
    req/min against an RPM of 30, but at ~10k (generation) and ~17k (judge)
    tokens/min against a TPM of 8000. Tokens are what force the pacing, and an
    unpaced run turns every token-limit 429 into a retry that still spends one
    of the 1000 daily requests.
    """
    chunks = load_and_split(CORPUS, settings.compare_chunk_size, settings.compare_chunk_overlap)
    n = len(chunks) if not settings.compare_max_chunks else min(len(chunks), settings.compare_max_chunks)
    q = len(load_queries())
    embed_batches = (n + 7) // 8

    # measured on the committed run; falls back to rough per-call estimates
    try:
        detail = json.loads(Path(METRICS).read_text())["details"]
        gen_tok = sum(r["llm_tokens"] for rows in detail.values() for r in rows)
        judge_tok = sum(r.get("judge_tokens", 0) for rows in detail.values() for r in rows)
        measured = True
    except Exception:
        gen_tok, judge_tok, measured = q * 2000, q * 3 * 1100, False

    print("=== Quota estimate ===")
    print(f"Per model, free tier: {RPM_LIMIT} RPM · 1000 RPD · {TPM_LIMIT} TPM · 200k TPD")
    print(f"Corpus: {n} chunks @ {settings.compare_chunk_size}/{settings.compare_chunk_overlap}; {q} eval queries")
    print(f"Token figures {'measured from ' + METRICS if measured else 'are rough estimates'}")
    print()

    print("Build")
    print(f"  vector : {embed_batches} NVIDIA requests (batch 8) ≈ {embed_batches * 1.5:.0f}s")
    print(f"  graph  : {n} Groq extractions, throttled 2.5s ≈ {n * 2.5:.0f}s")
    print()

    print(f"Eval ({q} queries x 3 systems, one retrieval each)")
    print(f"  {settings.groq_model}")
    print(f"    requests : {q * 5}  ({q} vector answers + {q * 2} entity extractions + {q * 2} graph answers)")
    print(f"    tokens   : {gen_tok:,} -> at least {gen_tok / TPM_LIMIT:.1f} min of TPM budget")
    print(f"  {settings.judge_model} (judge)")
    print(f"    requests : {q * 3}")
    print(f"    tokens   : {judge_tok:,} -> at least {judge_tok / TPM_LIMIT:.1f} min of TPM budget")
    print(f"  wall clock : ~{(gen_tok + judge_tok) / TPM_LIMIT:.0f} min, token-paced")
    tpd = 200_000
    limits = {
        f"{settings.groq_model} requests": 1000 // max(1, q * 5),
        f"{settings.groq_model} tokens": tpd // max(1, gen_tok),
        f"{settings.judge_model} requests": 1000 // max(1, q * 3),
        f"{settings.judge_model} tokens": tpd // max(1, judge_tok),
    }
    tightest = min(limits, key=limits.get)
    print(f"  daily headroom: {limits[tightest]} full evals/day, bound by {tightest}")
    print()

    print("Notes")
    print("  - RateLimiter paces on tokens; without it the retries alone exhaust the daily requests")
    print("  - `eval --no-judge` drops the judge entirely for retrieval-only iterations")
    print(f"  - embedding cache: {settings.embedding_cache_path}")
    print("  - graph checkpoint resumes per chunk_uid: compare/graph/extractions_partial.json")


def build_vector():
    print("=== Building vector store ===")
    stats = build_vector_store(CORPUS)
    _save_build_stats("vector", stats)


def build_graph():
    print("=== Building graph store (throttled) ===")
    stats = build_graph_throttled(CORPUS)
    _save_build_stats("graph", stats)


def _print_state(title: str, state: dict, extra: dict):
    print(f"--- {title} ---")
    for key, val in extra.items():
        print(f"{key}: {val}")
    print(f"chunks: {state.get('retrieved_chunk_ids', [])}")
    print(f"tokens: {state.get('tokens', 0)} groq, {state.get('embed_tokens', 0)} embed")
    print(f"answer: {state.get('answer', '')}\n")


def query_vector(question: str, k: int):
    agent = build_vector_agent(k=k)
    t0 = time.time()
    state = agent.invoke({"question": question, "k": k})
    _print_state(f"Vector k={k}", state, {"latency": f"{time.time() - t0:.2f}s"})
    return state


def query_graph(question: str, hops: int, k: int):
    store = GraphStore(settings.graph_large_path)
    store.load()
    agent = build_agent(store=store, k=k)
    t0 = time.time()
    state = agent.invoke({"question": question, "hops": hops, "k": k})
    _print_state(
        f"Graph hops={hops} k={k}",
        state,
        {
            "latency": f"{time.time() - t0:.2f}s",
            "query_entities": state.get("query_entities", []),
            "matched_nodes": state.get("matched_nodes", []),
            "triples": len(state.get("context", "").splitlines()),
        },
    )
    return state


def main():
    parser = argparse.ArgumentParser(description="Vector vs Graph RAG comparison")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check-quotas", help="Estimate API quota usage")
    sub.add_parser("build-vector", help="Build vector store only")
    sub.add_parser("build-graph", help="Build graph store only (throttled)")
    sub.add_parser("build-both", help="Build both stores")

    for name, help_ in [("query-vector", "Query vector RAG"), ("query-graph", "Query graph RAG"), ("query", "Query both")]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("question")
        p.add_argument("--k", type=int, default=settings.compare_k)
        if name != "query-vector":
            # default to settings.hops (1), not 2: hops=2 measured worse on
            # recall@4 and judged correctness, so it is opt-in, not the default
            p.add_argument("--hops", type=int, default=settings.hops)

    ev = sub.add_parser("eval", help="Run the full evaluation and write the report")
    ev.add_argument("--k", type=int, default=settings.compare_k)
    ev.add_argument("--queries", default=QUERIES)
    ev.add_argument("--no-judge", action="store_true", help="Skip the LLM judge (retrieval-only, ~1/3 the API calls)")

    sub.add_parser("report", help="Regenerate the report from existing metrics.json")

    args = parser.parse_args()

    if args.cmd == "check-quotas":
        check_quotas()
    elif args.cmd == "build-vector":
        build_vector()
    elif args.cmd == "build-graph":
        build_graph()
    elif args.cmd == "build-both":
        build_vector()
        build_graph()
    elif args.cmd == "query-vector":
        query_vector(args.question, args.k)
    elif args.cmd == "query-graph":
        query_graph(args.question, args.hops, args.k)
    elif args.cmd == "query":
        print(f"Q: {args.question}\n")
        query_vector(args.question, args.k)
        query_graph(args.question, args.hops, args.k)
    elif args.cmd == "eval":
        run_eval(queries_path=args.queries, k=args.k, output_path=METRICS, judge=not args.no_judge)
        generate_report(METRICS)  # the CLI now actually produces the report it documents
    elif args.cmd == "report":
        generate_report(METRICS)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
