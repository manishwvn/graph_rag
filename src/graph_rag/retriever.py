from pydantic import BaseModel, Field
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_groq import ChatGroq
from graph_rag.store import GraphStore
from graph_rag.config import settings


class QueryEntities(BaseModel):
    entities: list[str] = Field(description="Entity names from query")

    model_config = {"extra": "forbid"}


class GraphRetriever(BaseRetriever):
    """LangChain retriever: entity extraction + graph BFS -> Documents."""

    store: GraphStore = Field(description="Graph store")
    llm: ChatGroq = Field(description="LLM for query entity extraction")
    hops: int = Field(default=1, description="BFS hops")
    max_triples: int = Field(default=30, description="Max triples to return")

    def __init__(self, store: GraphStore, hops: int | None = None, max_triples: int = 30, llm: ChatGroq | None = None, **kwargs):
        if llm is None:
            llm = ChatGroq(
                model=settings.groq_model, temperature=0, max_tokens=200, groq_api_key=settings.groq_api_key
            )
        super().__init__(store=store, llm=llm, hops=hops if hops is not None else settings.hops, max_triples=max_triples, **kwargs)

    def _extract_query_entities(self, query: str) -> list[str]:
        # structured output = robust vs raw json.loads
        try:
            structured = self.llm.with_structured_output(QueryEntities, method="json_mode")
            result: QueryEntities = structured.invoke(
                f'Extract entity names from query. Return JSON {{"entities": [str]}}. Query: {query} Respond with JSON.'
            )
            return result.entities
        except Exception:
            return []

    def _match_nodes(self, query_entities: list[str]) -> list[str]:
        if not query_entities:
            return []
        lower_map = {n.lower(): n for n in self.store.graph.nodes}
        matched: set[str] = set()
        for qe in query_entities:
            ql = qe.strip().lower()
            if not ql or len(ql) < 2:  # guard tiny "a" matching all
                continue
            # exact
            if ql in lower_map:
                matched.add(lower_map[ql])
                continue
            # substring but only if qe length >=3 or lower length >=3 to reduce false positives
            for lower, orig in lower_map.items():
                if len(ql) >= 3 and len(lower) >= 3 and (ql in lower or lower in ql):
                    matched.add(orig)
        return list(matched)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        q_entities = self._extract_query_entities(query)
        matched = self._match_nodes(q_entities)
        triples = self.store.get_subgraph(matched, hops=self.hops, max_triples=self.max_triples)
        context = self.store.triples_as_text(triples)
        return [
            Document(
                page_content=context,
                metadata={
                    "query_entities": q_entities,
                    "matched_nodes": matched,
                    "triples": triples,
                    "hops": self.hops,
                },
            )
        ]
