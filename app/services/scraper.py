from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Dict
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.postgres import get_session_factory
from app.models.database import DownloadedNotice

logger = logging.getLogger(__name__)

BASE_LISTING = "https://www.nrb.org.np/category/notices/"
DOMAIN = "https://www.nrb.org.np"
PDF_PREFIX = "https://www.nrb.org.np/ofg/"


def _page_url(department: str, page: int) -> str:
    if page <= 1:
        return f"{BASE_LISTING}?department={department}"
    return f"{BASE_LISTING}page/{page}/?department={department}"


def _sanitize_filename(name: str, max_bytes: int = 180) -> str:
    name = (name or "").strip()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", name)
    name = name.strip(" ._")
    if not name:
        name = "notice"
    while len(name.encode("utf-8")) > max_bytes:
        name = name[:-1].strip(" ._")
        if not name:
            name = "notice"
            break
    return name


def _extract_pdf_links(html: str) -> Dict[str, str]:
    soup = BeautifulSoup(html, "lxml")
    out: Dict[str, str] = {}
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full = urljoin(DOMAIN, href)
        if not full.startswith(PDF_PREFIX):
            continue
        title = a.get_text(" ", strip=True) or ""
        if title.lower() in {"next", "previous", "first", "last"}:
            continue
        if title.isdigit():
            continue
        out[full] = title or full.rstrip("/").split("/")[-1]
    return out


def _download_file(session: requests.Session, url: str, out_path: Path) -> tuple[bool, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type") or "").lower()
        if "pdf" not in ctype:
            return (False, 0)
        tmp = out_path.with_suffix(out_path.suffix + ".part")
        bytes_written = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 64):
                if not chunk:
                    continue
                f.write(chunk)
                bytes_written += len(chunk)
        tmp.replace(out_path)
        return (True, bytes_written)


def _get_seen_urls(db: Session) -> set[str]:
    rows = db.query(DownloadedNotice.url).all()
    return {r[0] for r in rows}


def run_scraper(
    department: str = "ofg",
    max_pages: int = 200,
    polite_delay_sec: float = 0.6,
) -> dict:
    """
    Scrape NRB notices, download PDFs, and record in Postgres.
    Returns a summary dict.
    """
    settings = get_settings()
    out_dir = Path(settings.DOCS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    db: Session = get_session_factory()()

    http = requests.Session()
    http.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; NRB-PDF-Downloader/1.0; +https://www.nrb.org.np/)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })

    try:
        seen = _get_seen_urls(db)
        downloaded_count = 0
        skipped_count = 0
        error_count = 0

        for page in range(1, max_pages + 1):
            url = _page_url(department, page)
            logger.info("[page %d] Fetching: %s", page, url)

            resp = http.get(url, timeout=60)
            if resp.status_code == 404:
                logger.info("Page %d returned 404, stopping", page)
                break
            resp.raise_for_status()

            links = _extract_pdf_links(resp.text)
            if not links:
                logger.info("Page %d has no /ofg/ links, stopping", page)
                break

            for pdf_url, title in links.items():
                if pdf_url in seen:
                    continue
                seen.add(pdf_url)

                safe_title = _sanitize_filename(title)
                filename = f"{safe_title}.pdf"
                out_path = out_dir / filename

                # Handle duplicate filenames
                if out_path.exists():
                    i = 2
                    while True:
                        candidate = out_dir / f"{safe_title} ({i}).pdf"
                        if not candidate.exists():
                            out_path = candidate
                            filename = candidate.name
                            break
                        i += 1

                try:
                    ok, nbytes = _download_file(http, pdf_url, out_path)

                    if ok:
                        downloaded_count += 1
                        status = "downloaded"
                        logger.info("  downloaded: %s (%d bytes)", filename, nbytes)
                    else:
                        skipped_count += 1
                        status = "skipped_non_pdf"
                        logger.info("  skipped (not PDF): %s", pdf_url)

                    db.add(DownloadedNotice(
                        url=pdf_url,
                        title=title,
                        filename=filename,
                        filepath=str(out_path),
                        page=page,
                        bytes=nbytes,
                        status=status,
                    ))
                    db.commit()

                except Exception as e:
                    error_count += 1
                    logger.error("  error downloading %s: %s", pdf_url, e)
                    db.add(DownloadedNotice(
                        url=pdf_url,
                        title=title,
                        filename=filename,
                        filepath=str(out_path),
                        page=page,
                        bytes=0,
                        status=f"error: {e}",
                    ))
                    db.commit()

                time.sleep(polite_delay_sec)

        return {
            "status": "completed",
            "downloaded": downloaded_count,
            "skipped": skipped_count,
            "errors": error_count,
        }
    finally:
        db.close()
