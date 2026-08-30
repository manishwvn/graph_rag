from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    groq_api_key: str = Field(..., alias="GROQ_API_KEY", description="Groq API key")
    groq_model: str = Field(default="qwen/qwen3.8-27b", alias="GROQ_MODEL", description="Groq model id")
    chunk_size: int = Field(default=400, description="Splitter chunk size")
    chunk_overlap: int = Field(default=40, description="Splitter overlap")
    graph_path: str = Field(default="graph.gpickle", description="Pickled graph path")
    hops: int = Field(default=1, description="Default BFS hops")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
