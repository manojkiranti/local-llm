from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

_engine = None
_SessionLocal = None


class Base(DeclarativeBase):
    pass


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionLocal


def get_db():
    """Yield a DB session for FastAPI dependency injection."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    """Create all tables."""
    from app.models.database import DownloadedNotice, EmbeddedFile  # noqa: F401

    Base.metadata.create_all(bind=get_engine())


def close_db() -> None:
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
        _engine = None
        _SessionLocal = None
