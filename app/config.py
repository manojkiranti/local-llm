from __future__ import annotations

import logging
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_ENV: str = "development"

    # Database
    DATABASE_URL: str = ""

    # Documents
    DOCS_DIR: str = "./download"

    # Qdrant
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str = ""
    QDRANT_DISTANCE: str = "Cosine"
    QDRANT_VECTOR_NAME: str | None = None

    # Embeddings
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DIM: int = 1024
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_NORMALIZE: bool = True

    # Chunking
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120

    # Language / preprocessing
    ENABLE_PREETI_CONVERSION: bool = True
    PREETI_FONT_NAME: str = "Preeti"

    # LLM (OpenAI-compatible: DeepSeek, OpenAI, etc.)
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = ""
    LLM_MODEL: str = ""
    LLM_TIMEOUT_SECONDS: int = 120

    # API / retrieval
    DEFAULT_TOP_K: int = 5
    MAX_TOP_K: int = 20
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def configure_logging(log_level: str) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )