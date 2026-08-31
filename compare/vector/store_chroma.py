"""Vector store wrapper for ChromaDB with cosine similarity."""

import hashlib
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document

from graph_rag.config import settings


class VectorStoreChroma:
    """Persistent ChromaDB collection for vector RAG."""

    def __init__(self, persist_path: str | None = None, collection_name: str = "compare_vector"):
        self.persist_path = Path(persist_path or settings.vector_store_path)
        self.collection_name = collection_name
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.persist_path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )
        print(f"[vector] Chroma init {self.persist_path} collection={self.collection_name} count={self._collection.count()}")

    def add_documents(self, docs: list[Document], embeddings: list[list[float]]):
        """Upsert documents keyed by chunk_uid — the id space shared with the graph store."""
        if not docs or not embeddings:
            return
        if len(docs) != len(embeddings):
            raise ValueError(f"docs {len(docs)} != embeddings {len(embeddings)}")
        ids, documents, metadatas = [], [], []
        for i, doc in enumerate(docs):
            src = doc.metadata.get("source", "unknown")
            chunk_id = doc.metadata.get("chunk_id", i)
            ids.append(doc.metadata.get("chunk_uid") or f"{src}:{chunk_id}")
            documents.append(doc.page_content)
            metadatas.append({"source": src, "chunk_id": chunk_id})
        self._collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        print(f"[vector] upserted {len(docs)} docs, total={self._collection.count()}")

    def query(self, query_embedding: list[float], n_results: int = 4) -> dict[str, Any]:
        """Cosine similarity query. Embeddings are deliberately not fetched back."""
        res = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        return {
            "documents": res["documents"][0] if res["documents"] else [],
            "metadatas": res["metadatas"][0] if res["metadatas"] else [],
            "distances": res["distances"][0] if res["distances"] else [],
            "ids": res["ids"][0] if res["ids"] else [],
        }

    def clear(self):
        """Delete and recreate collection."""
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = self._client.create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )
        print(f"[vector] cleared collection {self.collection_name}")

    def count(self) -> int:
        return self._collection.count()

    def fingerprint(self) -> str:
        """Hash of the indexed chunk ids — changes whenever the index changes."""
        ids = sorted(self._collection.get(include=[])["ids"])
        return hashlib.sha256("\x00".join(ids).encode()).hexdigest()[:16]

    def stats(self) -> dict[str, Any]:
        size = sum(f.stat().st_size for f in self.persist_path.rglob("*") if f.is_file())
        return {
            "count": self.count(),
            "persist_path": str(self.persist_path),
            "size_bytes": size,
            "size_mb": round(size / 1024 / 1024, 2),
        }
