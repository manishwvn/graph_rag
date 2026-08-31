"""Vector RAG agent: LangGraph StateGraph (retrieve -> generate).

Deliberately mirrors `graph_rag.agent`: same graph shape, same LLM settings and
the *same* prompt string (imported, not copied), so the only difference between
the two systems under evaluation is retrieval.
"""

from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from compare.vector.retriever_vector import VectorRetriever
from graph_rag.agent import ANSWER_SYSTEM
from graph_rag.config import settings


class VectorState(TypedDict, total=False):
    question: str
    context: str
    answer: str
    retrieved_chunk_ids: list[str]
    k: int
    tokens: int
    embed_tokens: int


def build_vector_agent(k: int | None = None, retriever: VectorRetriever | None = None, model: str | None = None):
    """Build the compiled vector agent. `retriever` is injectable so the eval
    harness can reuse one embedding client (and its cache) across systems."""
    k = k if k is not None else settings.compare_k
    retriever = retriever or VectorRetriever(k=k)

    llm = ChatGroq(
        model=model or settings.groq_model,
        temperature=0,
        max_tokens=500,
        groq_api_key=settings.groq_api_key,
    )
    prompt = ChatPromptTemplate.from_messages([("system", ANSWER_SYSTEM), ("human", "{question}")])
    chain = prompt | llm

    def retrieve_node(state: VectorState):
        doc = retriever.invoke(state["question"])[0]
        return {
            "context": doc.page_content,
            "retrieved_chunk_ids": doc.metadata["retrieved_chunk_ids"],
            "embed_tokens": doc.metadata.get("embed_tokens", 0),
            "tokens": 0,
        }

    def generate_node(state: VectorState):
        resp = chain.invoke({"context": state["context"], "question": state["question"]})
        usage = (getattr(resp, "response_metadata", {}) or {}).get("token_usage", {}) or {}
        return {"answer": resp.content, "tokens": state.get("tokens", 0) + int(usage.get("total_tokens", 0))}

    graph = StateGraph(VectorState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()
