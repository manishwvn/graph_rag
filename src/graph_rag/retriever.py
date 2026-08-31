from pydantic import BaseModel, Field
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_groq import ChatGroq

from graph_rag.config import settings
from graph_rag.store import GraphStore, normalize_name


class QueryEntities(BaseModel):
    entities: list[str] = Field(description="Entity names from query")

    model_config = {"extra": "forbid"}


def _tokens(name: str) -> list[str]:
    return [t for t in normalize_name(name).split() if t]


def _token_eq(a: str, b: str) -> bool:
    """Tokens match exactly, or one is a >=4 char prefix of the other.

    Lets ``Corporation`` match ``Corp`` without letting ``the`` match ``theory``.
    """
    if a == b:
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    return len(short) >= 4 and long.startswith(short)


class GraphRetriever(BaseRetriever):
    """LangChain retriever: entity extraction + graph expansion -> Documents."""

    store: GraphStore = Field(description="Graph store")
    llm: ChatGroq = Field(description="LLM for query entity extraction")
    hops: int = Field(default=1, description="BFS hops")
    max_triples: int = Field(default=30, description="Max triples to return")
    k: int = Field(default=4, description="Chunks reported for retrieval scoring")
    max_context_chars: int | None = Field(default=None, description="Context budget in chars")
    min_token_overlap: float = Field(default=0.5, description="Jaccard floor for fuzzy node match")

    def __init__(
        self,
        store: GraphStore,
        hops: int | None = None,
        max_triples: int = 30,
        k: int = 4,
        max_context_chars: int | None = None,
        llm: ChatGroq | None = None,
        **kwargs,
    ):
        if llm is None:
            llm = ChatGroq(
                model=settings.groq_model, temperature=0, max_tokens=200, groq_api_key=settings.groq_api_key
            )
        super().__init__(
            store=store,
            llm=llm,
            hops=hops if hops is not None else settings.hops,
            max_triples=max_triples,
            k=k,
            max_context_chars=(
                settings.compare_max_context_chars if max_context_chars is None else max_context_chars
            ),
            **kwargs,
        )

    def _extract_query_entities(self, query: str) -> tuple[list[str], int]:
        """Returns (entity names, prompt+completion tokens used)."""
        try:
            structured = self.llm.with_structured_output(
                QueryEntities, method="json_mode", include_raw=True
            )
            out = structured.invoke(
                f'Extract entity names from query. Return JSON {{"entities": [str]}}. Query: {query} Respond with JSON.'
            )
            parsed: QueryEntities | None = out.get("parsed")
            raw = out.get("raw")
            usage = (getattr(raw, "response_metadata", {}) or {}).get("token_usage", {}) or {}
            return (parsed.entities if parsed else [], int(usage.get("total_tokens", 0)))
        except Exception:
            return [], 0

    def _match_nodes(self, query_entities: list[str]) -> list[str]:
        """Exact normalized match, else best token-overlap candidate above a floor.

        The previous raw-substring rule matched any node containing the query
        string in either direction, so the entity ``quantum`` seeded 30 nodes.
        """
        if not query_entities:
            return []
        by_norm: dict[str, str] = {}
        for n in self.store.graph.nodes:
            by_norm.setdefault(normalize_name(n), n)
        node_tokens = {n: _tokens(n) for n in self.store.graph.nodes}

        matched: list[str] = []
        for qe in query_entities:
            qn = normalize_name(qe)
            if len(qn) < 2:
                continue
            if qn in by_norm:
                if by_norm[qn] not in matched:
                    matched.append(by_norm[qn])
                continue
            qt = _tokens(qe)
            if not qt:
                continue
            best: tuple[float, str] | None = None
            for node, nt in node_tokens.items():
                if not nt:
                    continue
                common = sum(1 for a in qt if any(_token_eq(a, b) for b in nt))
                union = len(qt) + len(nt) - common
                score = common / union if union else 0.0
                if score >= self.min_token_overlap and (best is None or (score, node) > (best[0], best[1])):
                    best = (score, node)
            if best and best[1] not in matched:
                matched.append(best[1])
        return matched

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        q_entities, tokens = self._extract_query_entities(query)
        matched = self._match_nodes(q_entities)
        triples = self.store.get_subgraph(matched, hops=self.hops, max_triples=self.max_triples, query=query, max_chars=self.max_context_chars)
        context = self.store.triples_as_text(triples)
        return [
            Document(
                page_content=context,
                metadata={
                    "query_entities": q_entities,
                    "matched_nodes": matched,
                    "triples": [tuple(t[:3]) for t in triples],
                    # ranked chunk_uids -- the shared id space with the vector store
                    "retrieved_chunk_ids": self.store.chunk_ids_from_triples(triples, self.k),
                    "all_chunk_ids": self.store.chunk_ids_from_triples(triples),
                    "hops": self.hops,
                    "k": self.k,
                    "retrieval_tokens": tokens,
                },
            )
        ]
