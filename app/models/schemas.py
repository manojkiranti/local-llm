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
    group_name: str | None = Field(
        default=None,
        description="Optional group/index name to scope retrieval to",
    )


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
    group_name: str | None = None
    processed_at: datetime

    model_config = {"from_attributes": True}


class EmbedRequest(BaseModel):
    """Optional: specify file paths to embed. If empty, embeds all new files."""
    filepaths: list[str] = Field(default_factory=list)
    group_name: str | None = Field(
        default=None,
        description="Optional group/index name to scope these documents under",
    )


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


class ExtractResponse(BaseModel):
    instruction: str
    filename: str
    answer: str


class ProcessTextRequest(BaseModel):
    content: str = Field(..., min_length=1, description="Raw text content to process")
    instruction: str = Field(..., min_length=1, description="What the LLM should do with the content")


class ProcessTextResponse(BaseModel):
    instruction: str
    answer: str


class GroupCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_\-]+$")
    description: str = Field(default="", max_length=4000)


class GroupUpdate(BaseModel):
    description: str = Field(..., max_length=4000)


class GroupOut(BaseModel):
    name: str
    description: str
    document_count: int = 0
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class VerifyItem(BaseModel):
    requirement: str
    evidence: str | None = None


class VerifyReport(BaseModel):
    group_name: str
    instruction: str
    filename: str
    satisfied: list[VerifyItem] = Field(default_factory=list)
    missing: list[VerifyItem] = Field(default_factory=list)
    unclear: list[VerifyItem] = Field(default_factory=list)
    summary: str
    reference_files: list[str] = Field(default_factory=list)


class ComponentHealth(BaseModel):
    status: Literal["ok", "error"]
    detail: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    app_env: str
    qdrant: ComponentHealth
    llm: ComponentHealth
