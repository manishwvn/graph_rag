"""Vector retriever: embed query -> cosine top-k -> Document."""

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from compare.vector.embed_nvidia import NVIDIAEmbeddings
from compare.vector.store_chroma import VectorStoreChroma
from graph_rag.config import settings


class VectorRetriever(BaseRetriever):
    """NVIDIA embeddings + ChromaDB cosine search.

    Mirrors GraphRetriever: same BaseRetriever contract, and the same
    `retrieved_chunk_ids` metadata key so both systems can be scored identically.
    """

    store: VectorStoreChroma = Field(description="Chroma vector store")
    embeddings: NVIDIAEmbeddings = Field(description="Embedding client")
    k: int = Field(default=4, description="Top-k chunks")

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        store: VectorStoreChroma | None = None,
        embeddings: NVIDIAEmbeddings | None = None,
        k: int | None = None,
        **kwargs,
    ):
        super().__init__(
            store=store or VectorStoreChroma(),
            embeddings=embeddings or NVIDIAEmbeddings(),
            k=k if k is not None else settings.compare_k,
            **kwargs,
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        before = self.embeddings.tokens
        q_emb = self.embeddings.embed_query(query)
        res = self.store.query(q_emb, n_results=self.k)
        docs = res["documents"]
        context = "\n\n".join(f"[{i + 1}] {d}" for i, d in enumerate(docs))
        return [
            Document(
                page_content=context,
                metadata={
                    "query": query,
                    "matched": len(docs),
                    "retrieved_chunk_ids": res["ids"],
                    "all_chunk_ids": res["ids"],
                    "distances": res["distances"],
                    "k": self.k,
                    # embedding tokens are billed by NVIDIA, not Groq; tracked separately
                    "embed_tokens": self.embeddings.tokens - before,
                    "retrieval_tokens": 0,
                },
            )
        ]
