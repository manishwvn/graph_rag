import logging
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from graph_rag.config import settings
from graph_rag.retriever import GraphRetriever
from graph_rag.store import GraphStore

logger = logging.getLogger(__name__)

ANSWER_SYSTEM = (
    "Answer the question using ONLY the context below. "
    "If the context is insufficient, reply exactly: I don't know based on the provided context.\n"
    "Context:\n{context}"
)


class GraphRAGState(TypedDict, total=False):
    question: str
    query_entities: list[str]
    matched_nodes: list[str]
    retrieved_chunk_ids: list[str]
    context: str
    answer: str
    hops: int
    k: int
    tokens: int


def build_agent(store: GraphStore | None = None, model: str | None = None, k: int | None = None):
    store = store or GraphStore(settings.graph_path)
    if store.graph.number_of_nodes() == 0:
        try:
            store.load()
        except FileNotFoundError:
            logger.warning("Graph not found at %s — agent returns empty context until `python app.py --build`", store.path)

    llm = ChatGroq(
        model=model or settings.groq_model,
        temperature=0,
        max_tokens=500,
        groq_api_key=settings.groq_api_key,
    )
    prompt = ChatPromptTemplate.from_messages([("system", ANSWER_SYSTEM), ("human", "{question}")])
    chain = prompt | llm

    # one retriever per hop setting, reused across calls: rebuilding it per
    # invocation also rebuilt a ChatGroq client per query.
    retrievers: dict[tuple[int, int], GraphRetriever] = {}
    default_k = k if k is not None else settings.compare_k

    def _retriever(hops: int, want_k: int) -> GraphRetriever:
        key = (hops, want_k)
        if key not in retrievers:
            retrievers[key] = GraphRetriever(
                store=store, hops=hops, max_triples=settings.compare_max_triples, k=want_k
            )
        return retrievers[key]

    def retrieve_node(state: GraphRAGState):
        hops = state.get("hops", settings.hops)
        want_k = state.get("k", default_k)
        doc = _retriever(hops, want_k).invoke(state["question"])[0]
        return {
            "query_entities": doc.metadata["query_entities"],
            "matched_nodes": doc.metadata["matched_nodes"],
            "retrieved_chunk_ids": doc.metadata["retrieved_chunk_ids"],
            "context": doc.page_content,
            "tokens": doc.metadata.get("retrieval_tokens", 0),
        }

    def generate_node(state: GraphRAGState):
        try:
            resp = chain.invoke({"context": state["context"], "question": state["question"]})
            usage = (getattr(resp, "response_metadata", {}) or {}).get("token_usage", {}) or {}
            return {"answer": resp.content, "tokens": state.get("tokens", 0) + int(usage.get("total_tokens", 0))}
        except Exception as e:
            logger.exception("generate failed")
            return {"answer": f"Error generating answer: {e}"}

    builder = StateGraph(GraphRAGState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    return builder.compile()
