from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    id: str
    score: float
    text: str
    source: str | None = None
    title: str | None = None
    file_name: str | None = None
    page: int | None = None
    chunk_id: str | int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    score_threshold: float | None = Field(default=None, ge=-1.0, le=1.0)


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[RetrievedChunk]
    used_context: int
    fallback: bool


class SearchResponse(BaseModel):
    query: str
    top_k: int
    results: list[RetrievedChunk]


class EmbeddedFileOut(BaseModel):
    id: int
    filepath: str
    filename: str
    extension: str | None = None
    chunk_count: int
    processed_at: datetime

    model_config = {"from_attributes": True}


class EmbedRequest(BaseModel):
    """Optional: specify file paths to embed. If empty, embeds all new files."""
    filepaths: list[str] = Field(default_factory=list)


class EmbedResponse(BaseModel):
    message: str
    task_id: str


class DownloadedNoticeOut(BaseModel):
    id: int
    url: str
    title: str
    filename: str
    filepath: str
    page: int
    bytes: int
    status: str
    downloaded_at: datetime

    model_config = {"from_attributes": True}


class ScrapeRequest(BaseModel):
    department: str = Field(default="ofg")
    max_pages: int = Field(default=200, ge=1, le=500)


class ScrapeResponse(BaseModel):
    message: str
    task_id: str


class ComponentHealth(BaseModel):
    status: Literal["ok", "error"]
    detail: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    app_env: str
    qdrant: ComponentHealth
    llm: ComponentHealth
