from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.postgres import get_db
from app.models.database import DownloadedNotice
from app.models.schemas import DownloadedNoticeOut, ScrapeRequest, ScrapeResponse
from app.services.scraper import run_scraper

logger = logging.getLogger(__name__)

router = APIRouter(tags=["scraper"])

_task_status: dict[str, dict] = {}


def _run_scrape_task(task_id: str, department: str, max_pages: int):
    _task_status[task_id] = {"status": "running"}
    try:
        result = run_scraper(department=department, max_pages=max_pages)
        _task_status[task_id] = result
    except Exception as exc:
        logger.exception("Scrape task %s failed: %s", task_id, exc)
        _task_status[task_id] = {"status": "failed", "error": str(exc)}


@router.post("/scraper/run", response_model=ScrapeResponse)
def start_scraper(
    request: ScrapeRequest,
    background_tasks: BackgroundTasks,
):
    """Trigger NRB notice scraping as a background task."""
    task_id = str(uuid.uuid4())
    _task_status[task_id] = {"status": "queued"}
    background_tasks.add_task(
        _run_scrape_task, task_id, request.department, request.max_pages,
    )
    return ScrapeResponse(
        message="Scraping started in background",
        task_id=task_id,
    )


@router.get("/scraper/status/{task_id}")
def scrape_status(task_id: str):
    """Check the status of a scraping task."""
    status = _task_status.get(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return status


@router.get("/notices", response_model=list[DownloadedNoticeOut])
def list_notices(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    status: str | None = Query(default=None, description="Filter by status (e.g. downloaded, error)"),
    db: Session = Depends(get_db),
):
    """List all downloaded notices."""
    query = db.query(DownloadedNotice)
    if status:
        query = query.filter(DownloadedNotice.status == status)
    rows = (
        query
        .order_by(DownloadedNotice.downloaded_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return rows


@router.get("/notices/{notice_id}", response_model=DownloadedNoticeOut)
def get_notice(notice_id: int, db: Session = Depends(get_db)):
    """Get a single notice by ID."""
    notice = db.query(DownloadedNotice).filter_by(id=notice_id).first()
    if not notice:
        raise HTTPException(status_code=404, detail="Notice not found")
    return notice


@router.delete("/notices/{notice_id}")
def delete_notice(
    notice_id: int,
    delete_file: bool = Query(default=False, description="Also delete the PDF file from disk"),
    db: Session = Depends(get_db),
):
    """Remove a notice record. Optionally delete the file from disk."""
    notice = db.query(DownloadedNotice).filter_by(id=notice_id).first()
    if not notice:
        raise HTTPException(status_code=404, detail="Notice not found")

    filepath = notice.filepath
    filename = notice.filename

    if delete_file:
        p = Path(filepath)
        if p.exists():
            p.unlink()
            logger.info("Deleted file: %s", filepath)

    db.delete(notice)
    db.commit()

    return {"message": f"Deleted notice: {filename}", "filepath": filepath}
