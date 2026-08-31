"""End-to-end pipeline: ingest -> extract -> canonicalize -> store."""

import logging

from graph_rag.config import settings
from graph_rag.extraction import extract_from_text
from graph_rag.ingestion import load_and_split
from graph_rag.store import GraphStore

logger = logging.getLogger(__name__)


def build_graph(
    data_dir: str = "data",
    graph_path: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
):
    docs = load_and_split(data_dir, chunk_size, chunk_overlap)
    if not docs:
        raise ValueError(f"No documents found in {data_dir} — add .txt files")
    print(f"Loaded {len(docs)} chunks from {data_dir}")

    # keep (doc, result) paired: a failed chunk must drop its doc too, otherwise
    # every later result is attributed to the wrong source chunk.
    pairs = []
    for d in docs:
        print(f"  Extracting {d.metadata['chunk_uid']}...")
        try:
            res = extract_from_text(d.page_content)
        except Exception as e:
            logger.exception("Extraction failed for %s", d.metadata.get("chunk_uid"))
            print(f"    !! failed: {e} — skipping")
            continue
        print(f"    -> {len(res.entities)} entities, {len(res.relations)} relations")
        pairs.append((d, res))

    if not pairs:
        raise RuntimeError("All extractions failed — no graph to build")
    if len(pairs) != len(docs):
        logger.warning("%d of %d chunks failed — building from the successful ones", len(docs) - len(pairs), len(docs))

    store = GraphStore(graph_path or settings.graph_path)
    store.build_from_documents([d for d, _ in pairs], [r for _, r in pairs])
    merged = store.canonicalize()
    store.save()
    print(f"Graph built: {store.stats()} ({len(merged)} aliases merged) -> {store.path}")
    return store
