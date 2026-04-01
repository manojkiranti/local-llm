from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.postgres import Base


class EmbeddedFile(Base):
    __tablename__ = "embedded_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filepath: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    extension: Mapped[str | None] = mapped_column(String, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    processed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class DownloadedNotice(Base):
    __tablename__ = "downloaded_notices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    url: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    filepath: Mapped[str] = mapped_column(String, nullable=False)
    page: Mapped[int] = mapped_column(Integer, nullable=False)
    bytes: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String, nullable=False, default="downloaded")
    downloaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
