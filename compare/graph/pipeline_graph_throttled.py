"""Throttled Graph RAG pipeline: load_and_split -> throttled extract -> canonicalize -> persist.

Free-tier safe: sleeps between chunks (~24 RPM), checkpoints every extraction and
resumes from it. The checkpoint is keyed by chunk_uid, not by filename -- keying
it by filename made every chunk of a single-file corpus look already-done, so any
resume silently produced a graph from just the first chunk.
"""

import json
import time
from pathlib import Path
from typing import Any

from graph_rag.config import settings
from graph_rag.extraction import extract_from_text
from graph_rag.ingestion import load_and_split
from graph_rag.schemas import ExtractionResult
from graph_rag.store import GraphStore

CHECKPOINT = "compare/graph/extractions_partial.json"


def build_graph_throttled(
    data_dir: str = "compare/data_large",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    max_chunks: int | None = None,
    sleep_s: float = 2.5,
    checkpoint_path: str | None = None,
    graph_path: str | None = None,
) -> dict[str, Any]:
    """Build the comparison graph with throttled extraction and checkpoint resume."""
    start = time.time()
    chunk_size = chunk_size or settings.compare_chunk_size
    chunk_overlap = chunk_overlap or settings.compare_chunk_overlap
    max_chunks = settings.compare_max_chunks if max_chunks is None else max_chunks
    checkpoint = Path(checkpoint_path or CHECKPOINT)
    graph_path = graph_path or settings.graph_large_path

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    Path(graph_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"[graph-throttled] loading {data_dir} chunk_size={chunk_size} overlap={chunk_overlap}")
    docs = load_and_split(data_dir, chunk_size, chunk_overlap)
    if not docs:
        raise ValueError(f"No documents loaded from {data_dir}")
    total = len(docs)
    if max_chunks:
        docs = docs[:max_chunks]
    print(f"[graph-throttled] {len(docs)}/{total} chunks to extract")

    done: dict[str, dict] = {}
    if checkpoint.exists():
        try:
            done = {p["chunk_uid"]: p for p in json.loads(checkpoint.read_text())}
            print(f"[graph-throttled] resuming from checkpoint: {len(done)} chunks already extracted")
        except Exception as e:
            print(f"[graph-throttled] checkpoint load failed, starting fresh: {e}")
            done = {}

    api_calls = 0
    for i, doc in enumerate(docs):
        uid = doc.metadata["chunk_uid"]
        if uid in done:
            print(f"[graph-throttled] skip {uid} (checkpoint)")
            continue
        print(f"[graph-throttled] extracting {uid} ({i + 1}/{len(docs)})...")
        try:
            result = extract_from_text(doc.page_content)
            api_calls += 1
        except Exception as e:
            checkpoint.write_text(json.dumps(list(done.values())))
            print(f"[graph-throttled] {uid} FAILED: {e} — checkpoint saved, rerun to resume")
            raise
        done[uid] = {"chunk_uid": uid, "result": result.model_dump()}
        checkpoint.write_text(json.dumps(list(done.values())))
        print(f"[graph-throttled] {uid} -> {len(result.entities)} entities, {len(result.relations)} relations")
        if i < len(docs) - 1:
            time.sleep(sleep_s)

    # pair each doc with its own extraction by chunk_uid; never by list position
    pairs = [(d, ExtractionResult(**done[d.metadata["chunk_uid"]]["result"])) for d in docs if d.metadata["chunk_uid"] in done]
    print(f"[graph-throttled] building graph from {len(pairs)} extractions...")
    store = GraphStore(graph_path)
    store.build_from_documents([d for d, _ in pairs], [r for _, r in pairs])
    raw = store.stats()
    merged = store.canonicalize()
    store.save()

    if checkpoint.exists():
        checkpoint.unlink()

    stats = {
        "chunks_total": total,
        "chunks_indexed": len(pairs),
        "extractions": len(pairs),
        "api_calls": api_calls,
        "nodes_before_canonicalization": raw["nodes"],
        "edges_before_canonicalization": raw["edges"],
        "aliases_merged": len(merged),
        "nodes": store.graph.number_of_nodes(),
        "edges": store.graph.number_of_edges(),
        "total_time_s": round(time.time() - start, 1),
        "graph_path": graph_path,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
    print(f"[graph-throttled] done: {stats}")
    return stats
