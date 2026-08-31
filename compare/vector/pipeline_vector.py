"""Vector pipeline: load_and_split -> embed passages -> Chroma persist."""

import time
from typing import Any

from compare.vector.embed_nvidia import NVIDIAEmbeddings
from compare.vector.store_chroma import VectorStoreChroma
from graph_rag.config import settings
from graph_rag.ingestion import load_and_split


def build_vector_store(
    data_dir: str = "compare/data_large",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    max_chunks: int | None = None,
) -> dict[str, Any]:
    """Build vector store from the comparison corpus. Returns stats."""
    start = time.time()
    chunk_size = chunk_size or settings.compare_chunk_size
    chunk_overlap = chunk_overlap or settings.compare_chunk_overlap
    max_chunks = settings.compare_max_chunks if max_chunks is None else max_chunks

    print(f"[vector-pipeline] loading {data_dir} chunk_size={chunk_size} overlap={chunk_overlap}")
    # chunk params are passed through: previously they were computed, printed and
    # then dropped, so the corpus was silently split at the default 400/40.
    docs = load_and_split(data_dir, chunk_size, chunk_overlap)
    if not docs:
        raise ValueError(f"No documents loaded from {data_dir}")
    total = len(docs)
    if max_chunks:
        docs = docs[:max_chunks]
    print(f"[vector-pipeline] {len(docs)}/{total} chunks indexed")

    embeddings = NVIDIAEmbeddings()
    texts = [d.page_content for d in docs]
    emb_start = time.time()
    vectors = embeddings.embed_passages(texts)
    emb_time = time.time() - emb_start

    store = VectorStoreChroma()
    store.clear()  # rebuild from scratch so stale chunk ids cannot survive
    store.add_documents(docs, vectors)

    stats = {
        "chunks_total": total,
        "chunks_indexed": len(docs),
        "embeddings": len(vectors),
        "dim": len(vectors[0]) if vectors else 0,
        "embed_time_s": round(emb_time, 1),
        "total_time_s": round(time.time() - start, 1),
        "api_calls": embeddings.api_calls,
        "embed_tokens": embeddings.tokens,
        "chroma": store.stats(),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
    print(f"[vector-pipeline] done: {stats}")
    return stats
