from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.postgres import get_db
from app.db.qdrant import delete_points_by_group
from app.models.database import DocumentGroup, EmbeddedFile
from app.models.schemas import (
    GroupCreate,
    GroupOut,
    GroupUpdate,
    VerifyReport,
)
from app.services.embedding import SUPPORTED_EXTS, extract_text, normalize_text
from app.services.llm import get_llm_service
from app.services.retrieval import get_retrieval_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/groups", tags=["groups"])


def _to_group_out(group: DocumentGroup, document_count: int) -> GroupOut:
    return GroupOut(
        name=group.name,
        description=group.description or "",
        document_count=document_count,
        created_at=group.created_at,
        updated_at=group.updated_at,
    )


@router.post("", response_model=GroupOut, status_code=201)
def create_group(payload: GroupCreate, db: Session = Depends(get_db)) -> GroupOut:
    if db.query(DocumentGroup).filter_by(name=payload.name).first():
        raise HTTPException(status_code=409, detail=f"Group '{payload.name}' already exists")

    group = DocumentGroup(name=payload.name, description=payload.description or "")
    db.add(group)
    db.commit()
    db.refresh(group)
    return _to_group_out(group, document_count=0)


@router.get("", response_model=list[GroupOut])
def list_groups(db: Session = Depends(get_db)) -> list[GroupOut]:
    rows = (
        db.query(
            DocumentGroup,
            func.count(EmbeddedFile.id).label("document_count"),
        )
        .outerjoin(EmbeddedFile, EmbeddedFile.group_name == DocumentGroup.name)
        .group_by(DocumentGroup.name)
        .order_by(DocumentGroup.created_at.desc())
        .all()
    )
    return [_to_group_out(group, count) for group, count in rows]


@router.get("/{name}", response_model=GroupOut)
def get_group(name: str, db: Session = Depends(get_db)) -> GroupOut:
    group = db.query(DocumentGroup).filter_by(name=name).first()
    if not group:
        raise HTTPException(status_code=404, detail=f"Group '{name}' not found")
    count = db.query(func.count(EmbeddedFile.id)).filter_by(group_name=name).scalar() or 0
    return _to_group_out(group, count)


@router.patch("/{name}", response_model=GroupOut)
def update_group(name: str, payload: GroupUpdate, db: Session = Depends(get_db)) -> GroupOut:
    group = db.query(DocumentGroup).filter_by(name=name).first()
    if not group:
        raise HTTPException(status_code=404, detail=f"Group '{name}' not found")
    group.description = payload.description
    db.commit()
    db.refresh(group)
    count = db.query(func.count(EmbeddedFile.id)).filter_by(group_name=name).scalar() or 0
    return _to_group_out(group, count)


@router.delete("/{name}")
def delete_group(name: str, db: Session = Depends(get_db)) -> dict:
    group = db.query(DocumentGroup).filter_by(name=name).first()
    if not group:
        raise HTTPException(status_code=404, detail=f"Group '{name}' not found")

    try:
        delete_points_by_group(name)
    except Exception as exc:
        logger.warning("Failed to delete Qdrant points for group %s: %s", name, exc)

    db.query(EmbeddedFile).filter_by(group_name=name).delete(synchronize_session=False)
    db.delete(group)
    db.commit()
    return {"message": f"Deleted group '{name}' and its embedded documents"}


@router.post("/{name}/verify", response_model=VerifyReport)
async def verify_against_group(
    name: str,
    file: UploadFile = File(..., description="Document to verify"),
    instruction: str = Form(..., min_length=1, description="What to check for"),
    top_k: int = Form(default=10, ge=1, le=50),
    db: Session = Depends(get_db),
) -> VerifyReport:
    group = db.query(DocumentGroup).filter_by(name=name).first()
    if not group:
        raise HTTPException(status_code=404, detail=f"Group '{name}' not found")

    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported extension '{ext}'. Allowed: {sorted(SUPPORTED_EXTS)}",
        )

    # Persist to a temp path so existing extractors (which take Path) work unchanged.
    tmp_dir = Path(tempfile.mkdtemp(prefix="verify_"))
    tmp_path = tmp_dir / (file.filename or f"upload{ext}")
    try:
        with tmp_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        await file.close()

    try:
        raw = extract_text(tmp_path)
        target_text = normalize_text(raw)
        if not target_text:
            raise HTTPException(
                status_code=422,
                detail="Could not extract any text from the uploaded document",
            )

        retrieval = get_retrieval_service()
        # Query relevant references using the instruction plus a sample of the target
        # so we catch both rule-driven and content-driven matches.
        target_sample = target_text[:1500]
        query = f"{instruction}\n\n{target_sample}".strip()
        chunks = retrieval.search(
            query=query,
            top_k=top_k,
            group_name=name,
        )

        llm = get_llm_service()
        report = llm.verify_document(
            group_description=group.description or "",
            instruction=instruction,
            reference_chunks=chunks,
            target_text=target_text,
        )

        reference_files = sorted({
            chunk.file_name for chunk in chunks if chunk.file_name
        })

        return VerifyReport(
            group_name=name,
            instruction=instruction,
            filename=file.filename or tmp_path.name,
            satisfied=report["satisfied"],
            missing=report["missing"],
            unclear=report["unclear"],
            summary=report["summary"],
            reference_files=reference_files,
        )
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
