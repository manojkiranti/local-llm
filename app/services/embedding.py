from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path

from importlib.resources import files as pkg_files

import npttf2utf
from docx import Document as DocxDocument
from pypdf import PdfReader
from qdrant_client.models import Distance, PointStruct, VectorParams
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.postgres import get_session_factory
from app.db.qdrant import get_qdrant_client
from app.models.database import EmbeddedFile
from app.services.retrieval import get_retrieval_service

logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".txt", ".md", ".pdf", ".docx"}

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
ENGLISH_HINT_RE = re.compile(
    r"\b(the|and|for|with|from|that|this|are|was|were|you|your|have|has|not|will|can)\b",
    re.IGNORECASE,
)

RULES_FILE = str(pkg_files("npttf2utf").joinpath("map.json"))
PREETI_MAPPER = npttf2utf.FontMapper(RULES_FILE)


# ── File reading ─────────────────────────────────────────────────────

def iter_files(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            yield path


def read_txt_or_md(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_docx(path: Path) -> str:
    doc = DocxDocument(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text)


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        return read_txt_or_md(path)
    if ext == ".docx":
        return read_docx(path)
    if ext == ".pdf":
        return read_pdf(path)
    return ""


# ── Text normalisation ───────────────────────────────────────────────

def looks_like_english(text: str) -> bool:
    return bool(ENGLISH_HINT_RE.search(text[:2000]))


def needs_preeti_conversion(text: str) -> bool:
    sample = " ".join(text.split())[:2000]
    if not sample:
        return False
    if DEVANAGARI_RE.search(sample):
        return False
    if looks_like_english(sample):
        return False
    ascii_chars = sum(1 for ch in sample if 32 <= ord(ch) <= 126)
    ascii_ratio = ascii_chars / max(len(sample), 1)
    suspicious_symbols = sum(ch in r"[]{};'/\|`~@#$%^&*_" for ch in sample)
    return ascii_ratio > 0.85 and suspicious_symbols >= 3


def normalize_text(raw: str) -> str:
    text = raw.replace("\x00", " ").strip()
    if needs_preeti_conversion(text):
        text = PREETI_MAPPER.map_to_unicode(
            text, from_font="Preeti",
            unescape_html_input=False, escape_html_output=False,
        )
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Chunking ─────────────────────────────────────────────────────────

def chunk_text(path: Path, text: str) -> list[dict]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "।", ".", "?", "!", " ", ""],
    )
    chunks = splitter.split_text(text)

    docs = []
    for i, chunk in enumerate(chunks):
        cleaned = chunk.strip()
        if len(cleaned) < 30:
            continue
        docs.append({
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{path.as_posix()}::{i}")),
            "text": cleaned,
            "metadata": {
                "source": str(path),
                "filename": path.name,
                "chunk_index": i,
                "total_chunks": len(chunks),
            },
        })
    return docs


# ── Main embed pipeline ─────────────────────────────────────────────

def _get_processed_set(db: Session) -> set[str]:
    rows = db.query(EmbeddedFile.filepath).all()
    return {r[0] for r in rows}


def _upsert_processed(db: Session, filepath: str, filename: str, ext: str, chunk_count: int):
    existing = db.query(EmbeddedFile).filter_by(filepath=filepath).first()
    if existing:
        existing.chunk_count = chunk_count
    else:
        db.add(EmbeddedFile(
            filepath=filepath, filename=filename,
            extension=ext, chunk_count=chunk_count,
        ))
    db.commit()


def run_embed_pipeline(filepaths: list[str] | None = None) -> dict:
    """
    Embed documents and upsert into Qdrant.
    If filepaths is empty/None, processes all new files in DOCS_DIR.
    Returns a summary dict.
    """
    settings = get_settings()
    docs_dir = Path(settings.DOCS_DIR)
    retrieval = get_retrieval_service()
    client = get_qdrant_client()
    db: Session = get_session_factory()()

    try:
        already_done = _get_processed_set(db)

        # Determine which files to process
        if filepaths:
            candidates = [Path(fp) for fp in filepaths]
        else:
            candidates = list(iter_files(docs_dir))

        all_docs = []
        processed_paths = []

        for path in candidates:
            if not path.is_file():
                logger.warning("SKIP  %s  -> file not found", path)
                continue
            if path.suffix.lower() not in SUPPORTED_EXTS:
                logger.warning("SKIP  %s  -> unsupported extension", path)
                continue
            if str(path) in already_done:
                logger.info("DONE  %s  -> already processed, skipping", path)
                continue

            raw = extract_text(path)
            if not raw.strip():
                logger.info("SKIP  %s  -> no text extracted", path)
                continue

            text = normalize_text(raw)
            if not text:
                logger.info("SKIP  %s  -> empty after normalization", path)
                continue

            docs = chunk_text(path, text)
            if not docs:
                logger.info("SKIP  %s  -> no usable chunks", path)
                continue

            all_docs.extend(docs)
            processed_paths.append((path, len(docs)))
            logger.info("LOAD  %s  -> %d chunks", path, len(docs))

        if not all_docs:
            return {"status": "no_new_files", "embedded": 0, "files": []}

        # Encode
        texts = [d["text"] for d in all_docs]
        vectors = retrieval.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=settings.EMBEDDING_NORMALIZE,
        )

        # Ensure collection exists
        if not client.collection_exists(settings.QDRANT_COLLECTION):
            client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=settings.EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )

        # Upsert to Qdrant
        points = [
            PointStruct(
                id=doc["id"],
                vector=vec.tolist(),
                payload={**doc["metadata"], "text": doc["text"]},
            )
            for doc, vec in zip(all_docs, vectors)
        ]

        batch_size = 128
        for start in range(0, len(points), batch_size):
            client.upsert(
                collection_name=settings.QDRANT_COLLECTION,
                points=points[start : start + batch_size],
                wait=True,
            )

        # Mark as processed in Postgres
        file_names = []
        for path, count in processed_paths:
            _upsert_processed(db, str(path), path.name, path.suffix.lower(), count)
            file_names.append(path.name)

        logger.info("Indexed %d chunks from %d files", len(points), len(processed_paths))
        return {
            "status": "completed",
            "embedded": len(points),
            "files": file_names,
        }
    finally:
        db.close()
