from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


class EnvKey(StrEnum):
    OPENAI_API_KEY = "OPENAI_API_KEY"
    OPENAI_EMBEDDING_MODEL = "OPENAI_EMBEDDING_MODEL"
    LLM_MODEL = "LLM_MODEL"
    VECTOR_DB_URL = "VECTOR_DB_URL"
    VECTOR_DB_API_KEY = "VECTOR_DB_API_KEY"
    VECTOR_DB_PROVIDER = "VECTOR_DB_PROVIDER"
    VECTOR_DB_COLLECTION = "VECTOR_DB_COLLECTION"


def get_env(key: EnvKey, default: str | None = None) -> str | None:
    raw = os.environ.get(key.value)
    if raw is None or not raw.strip():
        return default
    return raw


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    openai_embedding_model: str
    llm_model: str
    vector_db_url: str | None
    vector_db_api_key: str | None
    vector_db_provider: str
    vector_db_collection: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        openai_api_key=get_env(EnvKey.OPENAI_API_KEY),
        openai_embedding_model=get_env(
            EnvKey.OPENAI_EMBEDDING_MODEL, "text-embedding-3-small"
        )
        or "text-embedding-3-small",
        llm_model=get_env(EnvKey.LLM_MODEL, "gpt-4o-mini"),
        vector_db_url=get_env(EnvKey.VECTOR_DB_URL),
        vector_db_api_key=get_env(EnvKey.VECTOR_DB_API_KEY),
        vector_db_provider=get_env(EnvKey.VECTOR_DB_PROVIDER, "qdrant"),
        vector_db_collection=get_env(EnvKey.VECTOR_DB_COLLECTION, "companies")
        or "companies",
    )
