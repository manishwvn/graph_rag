from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from graph_rag.schemas import ExtractionResult
from graph_rag.config import settings


SYSTEM_PROMPT = """You extract entities and relations from text.
Return ONLY valid JSON matching the schema.

Entity types: PERSON, ORG, LOCATION, CONCEPT
Entity: {{"name": "Alice", "type": "PERSON"}}
Relation: {{"source": "Alice", "target": "Acme Corp", "relation": "works_at"}}
IMPORTANT: source/target must be entity NAME strings, not IDs.
Respond with JSON.
"""


def get_extraction_chain(model: str | None = None):
    """Production LangChain chain: ChatGroq with structured output."""
    llm = ChatGroq(
        model=model or settings.groq_model,
        temperature=0,
        max_tokens=800,
        groq_api_key=settings.groq_api_key,
    )
    structured = llm.with_structured_output(ExtractionResult, method="json_mode")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Text: {text}\nExtract entities and relations."),
        ]
    )
    return prompt | structured


def extract_from_text(text: str, model: str | None = None) -> ExtractionResult:
    chain = get_extraction_chain(model)
    return chain.invoke({"text": text})
