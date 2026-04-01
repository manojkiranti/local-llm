from __future__ import annotations

import logging

from fastapi import APIRouter

from app.config import get_settings
from app.db.qdrant import get_qdrant_client
from app.models.schemas import ComponentHealth, HealthResponse
from app.services.llm import get_llm_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    settings = get_settings()

    qdrant_status = ComponentHealth(status="ok", detail="Qdrant connection is healthy.")
    llm_status = ComponentHealth(status="ok", detail="LLM connection is healthy.")

    try:
        client = get_qdrant_client()
        client.get_collection(settings.QDRANT_COLLECTION)
        qdrant_status = ComponentHealth(
            status="ok",
            detail=f"Connected to collection '{settings.QDRANT_COLLECTION}'.",
        )
    except Exception as exc:
        logger.exception("Qdrant health check failed: %s", exc)
        qdrant_status = ComponentHealth(
            status="error",
            detail=str(exc),
        )

    try:
        llm_ok, llm_detail = get_llm_service().health_check()
        llm_status = ComponentHealth(
            status="ok" if llm_ok else "error",
            detail=llm_detail,
        )
    except Exception as exc:
        logger.exception("LLM health check failed: %s", exc)
        llm_status = ComponentHealth(
            status="error",
            detail=str(exc),
        )

    overall_status = (
        "ok"
        if qdrant_status.status == "ok" and llm_status.status == "ok"
        else "degraded"
    )

    return HealthResponse(
        status=overall_status,
        app_env=settings.APP_ENV,
        qdrant=qdrant_status,
        llm=llm_status,
    )