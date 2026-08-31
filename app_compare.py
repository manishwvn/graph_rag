#!/usr/bin/env python3
"""Comparison CLI: build both RAGs over one corpus, query them, evaluate, check quotas."""

import argparse
import json
import sys
import time
from pathlib import Path

from compare.eval.harness import METRICS, QUERIES, load_queries, run_eval
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
    """Estimate API calls against free-tier limits, from the real corpus and query set."""
    chunks = load_and_split(CORPUS, settings.compare_chunk_size, settings.compare_chunk_overlap)
    n = len(chunks) if not settings.compare_max_chunks else min(len(chunks), settings.compare_max_chunks)
    queries = load_queries()
    q = len(queries)
    embed_batches = (n + 7) // 8
    graph_systems = 2  # hops=1 and hops=2

    print("=== Quota estimate ===")
    print("Groq free tier ~30 RPM / ~14.4k RPD · NVIDIA free tier ~40 RPM")
    print(f"Corpus: {n} chunks @ {settings.compare_chunk_size}/{settings.compare_chunk_overlap}; {q} eval queries\n")

    print("Build")
    print(f"  vector : {embed_batches} NVIDIA requests (batch 8) ≈ {embed_batches * 1.5:.0f}s")
    print(f"  graph  : {n} Groq extractions, throttled 2.5s ≈ {n * 2.5:.0f}s\n")

    per_vector = 2  # generate + judge
    per_graph = 3  # entity extraction + generate + judge
    groq_calls = q * (per_vector + graph_systems * per_graph)
    print("Eval (1 retrieval per query per system — no double retrieval)")
    print(f"  vector       : {q} generate + {q} judge")
    print(f"  graph hops=1 : {q} entity-extract + {q} generate + {q} judge")
    print(f"  graph hops=2 : {q} entity-extract + {q} generate + {q} judge")
    print(f"  Groq total   : {groq_calls} requests ≈ {groq_calls / 30 * 60:.0f}s at 30 RPM")
    print(f"  NVIDIA total : {q} query embeddings (cached after first run)\n")

    print("Notes")
    print(f"  - embedding cache: {settings.embedding_cache_path}")
    print("  - graph checkpoint resumes per chunk_uid: compare/graph/extractions_partial.json")
    print("  - build-vector is cheap; build-graph is the throttled one, split it across runs if needed")


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
            p.add_argument("--hops", type=int, default=2)

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
