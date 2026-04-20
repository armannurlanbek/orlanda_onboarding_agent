"""Engine and session factory. Use only when DATABASE_URL is configured."""
from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from rag_agent.config import DATABASE_URL

_engine = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine():
    """Return a singleton Engine, or raise if DATABASE_URL is not set."""
    global _engine
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set. Configure PostgreSQL connection in the environment.")
    if _engine is None:
        _engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            echo=False,
        )
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return a configured sessionmaker."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=get_engine(),
        )
    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI-style dependency generator: yields a session and closes it.
    """
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
