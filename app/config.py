from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(default="")
    embed_model: str = Field(default="text-embedding-3-small")
    embed_dims: int = Field(default=1536)
    llm_model: str = Field(default="gpt-4o-mini")

    qdrant_path: str = Field(default="./data/qdrant_store")
    qdrant_collection: str = Field(default="companies")
    corpus_version_path: str = Field(default="./data/.corpus_version")

    internal_base_url: str = Field(default="http://127.0.0.1:8000")

    retrieval_timeout_s: float = Field(default=5.0)
    retrieval_max_retries: int = Field(default=3)
    retrieval_max_concurrency: int = Field(default=10)
    embed_batch_size: int = Field(default=64)

    score_floor: float = Field(default=0.30)
    weight_vector: float = Field(default=0.60)
    weight_geography: float = Field(default=0.25)
    weight_keyword_density: float = Field(default=0.10)
    weight_name_match: float = Field(default=0.05)

    prompt_version: str = Field(default="semantic_diagnoser_v1")
    max_no_improvement_iterations: int = Field(default=2)

    embedding_cache_size: int = Field(default=512)
    retrieval_cache_size: int = Field(default=256)
    diagnosis_cache_size: int = Field(default=256)


settings = Settings()
