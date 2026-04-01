from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes.chat import router as chat_router
from app.api.routes.documents import router as documents_router
from app.api.routes.health import router as health_router
from app.api.routes.scraper import router as scraper_router
from app.config import configure_logging, get_settings
from app.db.postgres import close_db, init_db
from app.db.qdrant import close_qdrant_client, ensure_collection_exists
from app.services.llm import close_llm_service
from app.services.retrieval import get_retrieval_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)

    logger.info("Starting RAG backend in %s mode", settings.APP_ENV)
    init_db()
    ensure_collection_exists()
    get_retrieval_service()

    try:
        yield
    finally:
        logger.info("Shutting down RAG backend")
        close_llm_service()
        close_qdrant_client()
        close_db()


app = FastAPI(
    title="RAG Retrieval and Answer API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled application error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error."},
    )


app.include_router(health_router)
app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(scraper_router)
