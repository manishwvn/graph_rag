import logging
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from graph_rag.store import GraphStore
from graph_rag.retriever import GraphRetriever
from graph_rag.config import settings

logger = logging.getLogger(__name__)


class GraphRAGState(TypedDict):
    question: str
    query_entities: list[str]
    matched_nodes: list[str]
    context: str
    answer: str
    hops: int


def build_agent(store: GraphStore | None = None, model: str | None = None):
    store = store or GraphStore(settings.graph_path)
    try:
        store.load()
    except FileNotFoundError:
        logger.warning("Graph not found at %s — agent will return empty context until `python app.py --build` is run", store.path)

    llm = ChatGroq(
        model=model or settings.groq_model,
        temperature=0,
        max_tokens=500,
        groq_api_key=settings.groq_api_key,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the question using ONLY the context triples below. If insufficient, say so. Context:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )
    chain = prompt | llm

    def retrieve_node(state: GraphRAGState):
        question = state["question"]
        hops = state.get("hops", settings.hops)
        # create isolated retriever per call to avoid mutating shared state
        retriever = GraphRetriever(store=store, hops=hops)
        docs = retriever.invoke(question)
        doc = docs[0]
        return {
            "query_entities": doc.metadata["query_entities"],
            "matched_nodes": doc.metadata["matched_nodes"],
            "context": doc.page_content,
        }

    def generate_node(state: GraphRAGState):
        try:
            resp = chain.invoke({"context": state["context"], "question": state["question"]})
            return {"answer": resp.content}
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


def get_graph():
    """Lazy singleton — avoids import-time side effects. Use this in app.py."""
    return build_agent()
