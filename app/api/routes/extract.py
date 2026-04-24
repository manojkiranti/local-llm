from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.models.schemas import ExtractResponse
from app.services.embedding import SUPPORTED_EXTS, extract_text, normalize_text
from app.services.llm import get_llm_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["extract"])


@router.post("/extract", response_model=ExtractResponse)
def extract_from_document(
    file: UploadFile = File(..., description="Document file (.txt, .md, .pdf, .docx)"),
    instruction: str = Form(..., min_length=1, description="What to extract from the document"),
) -> ExtractResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {', '.join(sorted(SUPPORTED_EXTS))}",
        )

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(file.file.read())
            tmp.flush()
            raw_text = extract_text(Path(tmp.name))
    except Exception as exc:
        logger.exception("Failed to read uploaded file: %s", exc)
        raise HTTPException(status_code=400, detail="Failed to extract text from file.") from exc

    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted from the uploaded file.")

    document_text = normalize_text(raw_text)

    try:
        llm_service = get_llm_service()
        answer = llm_service.generate_from_document(
            instruction=instruction,
            document_text=document_text,
        )
    except Exception as exc:
        logger.exception("LLM generation failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to process document with LLM.") from exc

    return ExtractResponse(
        instruction=instruction,
        filename=file.filename or "unknown",
        answer=answer,
    )
