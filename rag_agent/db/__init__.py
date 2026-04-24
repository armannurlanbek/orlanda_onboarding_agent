"""PostgreSQL persistence layer (SQLAlchemy + Alembic)."""
from rag_agent.db.base import Base
from rag_agent.db.models import (
    AuthSession,
    ChatLogEntry,
    DocumentChunk,
    DocumentIndexRecord,
    KnowledgeItemRecord,
    PdfMetadataRecord,
    User,
)
from rag_agent.db.session import get_db, get_engine, get_session_factory

__all__ = [
    "AuthSession",
    "Base",
    "ChatLogEntry",
    "DocumentChunk",
    "DocumentIndexRecord",
    "KnowledgeItemRecord",
    "PdfMetadataRecord",
    "User",
    "get_db",
    "get_engine",
    "get_session_factory",
]
