from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from app.config import get_settings
from app.models.schemas import AskRequest, AskResponse, SearchResponse
from app.services.llm import get_llm_service
from app.services.retrieval import get_retrieval_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


@router.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    settings = get_settings()
    retrieval_service = get_retrieval_service()
    llm_service = get_llm_service()

    top_k = min(max(request.top_k, 1), settings.MAX_TOP_K)

    try:
        chunks = retrieval_service.search(
            query=request.question,
            top_k=top_k,
            score_threshold=request.score_threshold,
        )

        fallback_message = llm_service.get_fallback_message(request.question)

        if not chunks:
            return AskResponse(
                question=request.question,
                answer=fallback_message,
                sources=[],
                used_context=0,
                fallback=True,
            )

        answer = llm_service.generate_answer(
            question=request.question,
            chunks=chunks,
        )

        fallback = answer.strip() == fallback_message.strip()

        return AskResponse(
            question=request.question,
            answer=answer,
            sources=chunks,
            used_context=len(chunks),
            fallback=fallback,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to answer question: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to generate answer.",
        ) from exc


@router.get("/search", response_model=SearchResponse)
def semantic_search(
    query: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(default=5, ge=1, le=50),
    score_threshold: float | None = Query(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Optional minimum similarity score",
    ),
) -> SearchResponse:
    settings = get_settings()
    retrieval_service = get_retrieval_service()

    safe_top_k = min(max(top_k, 1), settings.MAX_TOP_K)

    try:
        results = retrieval_service.search(
            query=query,
            top_k=safe_top_k,
            score_threshold=score_threshold,
        )
        return SearchResponse(
            query=query,
            top_k=safe_top_k,
            results=results,
        )
    except Exception as exc:
        logger.exception("Semantic search failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to search the knowledge base.",
        ) from exc