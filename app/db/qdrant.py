from __future__ import annotations

import logging
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    settings = get_settings()

    client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        timeout=30.0,
    )

    logger.info("Connected Qdrant client to %s", settings.QDRANT_URL)
    return client


def ensure_collection_exists() -> None:
    settings = get_settings()
    client = get_qdrant_client()

    if client.collection_exists(settings.QDRANT_COLLECTION):
        logger.info("Qdrant collection '%s' is available", settings.QDRANT_COLLECTION)
    else:
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=settings.EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Created Qdrant collection '%s'", settings.QDRANT_COLLECTION)


def close_qdrant_client() -> None:
    try:
        client = get_qdrant_client()
        close_method = getattr(client, "close", None)
        if callable(close_method):
            close_method()
    except Exception:
        pass