from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    groq_api_key: str = Field(..., alias="GROQ_API_KEY", description="Groq API key")
    groq_model: str = Field(default="qwen/qwen3.8-27b", alias="GROQ_MODEL", description="Groq model id")
    judge_model: str = Field(
        default="openai/gpt-oss-120b",
        alias="JUDGE_MODEL",
        description="Evaluation judge — a different family from groq_model, so neither system is graded by its own generator",
    )
    nvidia_api_key: str | None = Field(default=None, alias="NVIDIA_API_KEY", description="NVIDIA embedding API key")
    nvidia_embed_model: str = Field(default="nvidia/nemotron-3-embed-1b", alias="NVIDIA_EMBED_MODEL", description="NVIDIA embedding model")
    nvidia_embed_dim: int = Field(default=2048, description="Expected embedding dimension (validated on response)")
    chunk_size: int = Field(default=400, description="Splitter chunk size")
    chunk_overlap: int = Field(default=40, description="Splitter overlap")
    graph_path: str = Field(default="graph.json", description="Graph store path (JSON node-link)")
    hops: int = Field(default=1, description="Default BFS hops")
    # comparison corpus: both systems must see identical chunks
    compare_chunk_size: int = Field(default=800, description="Compare corpus chunk size")
    compare_chunk_overlap: int = Field(default=80, description="Compare corpus chunk overlap")
    compare_max_chunks: int = Field(default=0, description="0 = index every chunk; >0 truncates the corpus")
    compare_k: int = Field(default=4, description="Chunks retrieved per query by both systems")
    compare_max_triples: int = Field(default=60, description="Hard ceiling on triples per graph query")
    compare_max_context_chars: int = Field(default=2200, description="Context budget, matched to the vector system's k chunks")
    vector_store_path: str = Field(default="compare/vector/chroma_db", description="Chroma persist dir")
    graph_large_path: str = Field(default="compare/graph/graph_large.json", description="Comparison graph path")
    embedding_cache_path: str = Field(default="compare/vector/embed_cache.json", description="Embedding cache (JSON)")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )


settings = Settings()
