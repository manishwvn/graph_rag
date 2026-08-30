from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from graph_rag.config import settings


def load_documents(data_dir: str | Path = "data") -> list[Document]:
    """Load .txt files as LangChain Documents with metadata."""
    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"data_dir not found: {p.resolve()}")
    if not p.is_dir():
        raise NotADirectoryError(f"data_dir is not a directory: {p}")
    docs: list[Document] = []
    for f in sorted(p.glob("*.txt")):
        text = f.read_text(encoding="utf-8").strip()
        if not text:
            continue
        docs.append(Document(page_content=text, metadata={"source": f.name}))
    return docs


def split_documents(
    docs: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Production splitter: RecursiveCharacterTextSplitter."""
    if not docs:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    # per-source chunk_id (reset per source for clarity)
    from collections import Counter

    counter: Counter[str] = Counter()
    for d in chunks:
        src = d.metadata.get("source", "unknown")
        d.metadata["chunk_id"] = counter[src]
        counter[src] += 1
    return chunks


def load_and_split(data_dir: str | Path = "data") -> list[Document]:
    docs = load_documents(data_dir)
    return split_documents(docs)
