from __future__ import annotations

import logging
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    VectorParams,
)

from app.config import get_settings

logger = logging.getLogger(__name__)

GROUP_PAYLOAD_FIELD = "group_name"


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

    _ensure_group_payload_index(client, settings.QDRANT_COLLECTION)


def _ensure_group_payload_index(client: QdrantClient, collection: str) -> None:
    try:
        client.create_payload_index(
            collection_name=collection,
            field_name=GROUP_PAYLOAD_FIELD,
            field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.info("Created payload index on '%s'", GROUP_PAYLOAD_FIELD)
    except Exception as exc:
        # Already-exists is the common path; log at debug level.
        logger.debug("Payload index for '%s' not created: %s", GROUP_PAYLOAD_FIELD, exc)


def build_group_filter(group_name: str | None) -> Filter | None:
    if not group_name:
        return None
    return Filter(
        must=[
            FieldCondition(
                key=GROUP_PAYLOAD_FIELD,
                match=MatchValue(value=group_name),
            )
        ]
    )


def delete_points_by_group(group_name: str) -> None:
    settings = get_settings()
    client = get_qdrant_client()
    flt = build_group_filter(group_name)
    if flt is None:
        return
    client.delete(
        collection_name=settings.QDRANT_COLLECTION,
        points_selector=flt,
    )
    logger.info("Deleted Qdrant points for group '%s'", group_name)


def close_qdrant_client() -> None:
    try:
        client = get_qdrant_client()
        close_method = getattr(client, "close", None)
        if callable(close_method):
            close_method()
    except Exception:
        pass
