from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from qdrant_client.models import ScoredPoint
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.db.qdrant import get_qdrant_client
from app.models.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

TEXT_KEYS = (
    "text",
    "chunk_text",
    "content",
    "page_content",
    "document",
    "body",
)


class RetrievalService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = get_qdrant_client()
        self.model = SentenceTransformer(
            self.settings.EMBEDDING_MODEL,
            device=self.settings.EMBEDDING_DEVICE,
            trust_remote_code=True,
        )
        logger.info(
            "Loaded embedding model '%s' on device '%s'",
            self.settings.EMBEDDING_MODEL,
            self.settings.EMBEDDING_DEVICE,
        )

    def embed_query(self, query: str) -> list[float]:
        vector = self.model.encode(
            query,
            normalize_embeddings=self.settings.EMBEDDING_NORMALIZE,
            convert_to_numpy=True,
        )
        return vector.tolist()

    def search(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        safe_top_k = top_k or self.settings.DEFAULT_TOP_K
        safe_top_k = min(max(safe_top_k, 1), self.settings.MAX_TOP_K)

        query_vector = self.embed_query(query)

        if self.settings.QDRANT_VECTOR_NAME:
            vector_input: list[float] | tuple[str, list[float]] = (
                self.settings.QDRANT_VECTOR_NAME,
                query_vector,
            )
        else:
            vector_input = query_vector

        search_kwargs: dict[str, Any] = {
            "collection_name": self.settings.QDRANT_COLLECTION,
            "query": vector_input,
            "limit": safe_top_k,
            "with_payload": True,
            "with_vectors": False,
        }

        if score_threshold is not None:
            search_kwargs["score_threshold"] = score_threshold

        result = self.client.query_points(**search_kwargs)
        points = result.points

        results: list[RetrievedChunk] = []
        for point in points:
            chunk = self._to_chunk(point)
            if chunk.text.strip():
                results.append(chunk)

        return results

    def _to_chunk(self, point: ScoredPoint) -> RetrievedChunk:
        payload = point.payload or {}
        text = self._extract_text(payload)

        source = self._pick_first(
            payload,
            ["source", "source_path", "file_path", "path", "document_path", "url"],
        )
        title = self._pick_first(payload, ["title", "document_title", "name"])
        file_name = self._pick_first(payload, ["file_name", "filename", "document_name"])
        page = self._pick_first(payload, ["page", "page_number"])
        chunk_id = self._pick_first(payload, ["chunk_id", "chunk_index", "index"])

        if not file_name and isinstance(source, str):
            file_name = Path(source).name

        metadata = {
            key: value
            for key, value in payload.items()
            if key not in TEXT_KEYS
        }

        return RetrievedChunk(
            id=str(point.id),
            score=float(point.score),
            text=text,
            source=source if isinstance(source, str) else None,
            title=title if isinstance(title, str) else None,
            file_name=file_name if isinstance(file_name, str) else None,
            page=self._safe_int(page),
            chunk_id=chunk_id,
            metadata=metadata,
        )

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        for key in TEXT_KEYS:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        for value in payload.values():
            if isinstance(value, str) and len(value.strip()) > 100:
                return value.strip()

        return ""

    @staticmethod
    def _pick_first(payload: dict[str, Any], keys: list[str]) -> Any:
        for key in keys:
            if key in payload and payload[key] is not None:
                return payload[key]
        return None

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        try:
            if value is None:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None


@lru_cache(maxsize=1)
def get_retrieval_service() -> RetrievalService:
    return RetrievalService()