"""End-to-end pipeline: ingest -> extract -> store."""

import logging
from graph_rag.ingestion import load_and_split
from graph_rag.extraction import extract_from_text
from graph_rag.store import GraphStore
from graph_rag.config import settings

logger = logging.getLogger(__name__)


def build_graph(data_dir: str = "data", graph_path: str | None = None):
    docs = load_and_split(data_dir)
    if not docs:
        raise ValueError(f"No documents found in {data_dir} — add .txt files")
    print(f"Loaded {len(docs)} chunks from {data_dir}")

    results = []
    for d in docs:
        print(f"  Extracting {d.metadata['source']} chunk {d.metadata['chunk_id']}...")
        try:
            res = extract_from_text(d.page_content)
        except Exception as e:
            logger.exception("Extraction failed for %s chunk %s", d.metadata.get("source"), d.metadata.get("chunk_id"))
            print(f"    !! failed: {e} — skipping")
            continue
        print(f"    -> {len(res.entities)} entities, {len(res.relations)} relations")
        results.append(res)

    if not results:
        raise RuntimeError("All extractions failed — no graph to build")
    if len(docs) != len(results):
        logger.warning("Some chunks failed: %d docs vs %d results — building with successful only", len(docs), len(results))
        docs = docs[: len(results)]

    store = GraphStore(graph_path or settings.graph_path)
    store.build_from_documents(docs, results)
    store.save()
    print(f"Graph built: {store.stats()} -> {store.path}")
    return store
