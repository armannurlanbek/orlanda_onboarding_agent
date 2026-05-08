"""ORM models."""
from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from rag_agent.db.base import Base


class User(Base):
    """
    Application user. Password hash will use Argon2 (or similar) once auth is migrated off JSON.
    """

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    username: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    # "admin" | "user" — keep as string for simple migrations when adding roles later.
    role: Mapped[str] = mapped_column(String(16), nullable=False, default="user")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    must_change_password: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    password_changed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    temp_password_issued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class AuthSession(Base):
    """Opaque bearer token session (only SHA-256(secret:token) is stored, not the raw token)."""

    __tablename__ = "auth_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class MondayConnectionState(Base):
    """Ephemeral OAuth challenge state (PKCE) for monday auth callback validation."""

    __tablename__ = "monday_connection_states"
    __table_args__ = (
        Index("ix_monday_connection_states_user_id", "user_id"),
        Index("ix_monday_connection_states_expires_at", "expires_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    state: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    code_verifier: Mapped[str] = mapped_column(Text, nullable=False)
    redirect_uri: Mapped[str] = mapped_column(String(1024), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class MondayUserConnection(Base):
    """Stored per-user monday OAuth tokens (encrypted)."""

    __tablename__ = "monday_user_connections"
    __table_args__ = (
        Index("ix_monday_user_connections_user_id", "user_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    monday_user_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    monday_account_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    token_type: Mapped[str] = mapped_column(String(64), nullable=False, default="Bearer")
    access_token_encrypted: Mapped[str] = mapped_column(Text, nullable=False)
    refresh_token_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    scope: Mapped[str | None] = mapped_column(Text, nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class DocumentIndexRecord(Base):
    """Document-level index state for incremental RAG updates."""

    __tablename__ = "rag_documents"

    doc_id: Mapped[str] = mapped_column(String(512), primary_key=True)
    # Supported types: "pdf" | "knowledge_item".
    doc_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    # Relative pdf path or knowledge item id.
    source_ref: Mapped[str] = mapped_column(String(1024), nullable=False, unique=True, index=True)
    source_name: Mapped[str] = mapped_column(String(512), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class DocumentChunk(Base):
    """One embedded chunk for pgvector retrieval."""

    __tablename__ = "rag_document_chunks"
    __table_args__ = (
        UniqueConstraint("doc_id", "chunk_no", name="uq_rag_document_chunks_doc_chunkno"),
        Index("ix_rag_document_chunks_doc_id", "doc_id"),
        Index("ix_rag_document_chunks_source_file", "source_file"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    doc_id: Mapped[str] = mapped_column(
        String(512),
        ForeignKey("rag_documents.doc_id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_no: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    search_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    source_file: Mapped[str] = mapped_column(String(512), nullable=False)
    page: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    metadata_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    # OpenAI text-embedding-3-small dimension.
    embedding: Mapped[list[float]] = mapped_column(Vector(1536), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class KnowledgeItemRecord(Base):
    """Text knowledge item stored in PostgreSQL."""

    __tablename__ = "knowledge_items"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    last_updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    update_period_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    responsible: Mapped[str] = mapped_column(String(255), nullable=False, default="")


class PdfMetadataRecord(Base):
    """Metadata for one PDF file (path is relative to knowledge_base)."""

    __tablename__ = "pdf_metadata"

    path: Mapped[str] = mapped_column(String(1024), primary_key=True)
    last_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    update_period_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    responsible: Mapped[str] = mapped_column(String(255), nullable=False, default="")


class ChatLogEntry(Base):
    """Persistent chat/admin review log entry."""

    __tablename__ = "chat_logs"
    __table_args__ = (
        Index("ix_chat_logs_timestamp", "timestamp"),
        Index("ix_chat_logs_username", "username"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    username: Mapped[str] = mapped_column(String(255), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False, default="")
    sources: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    correct_answer: Mapped[str] = mapped_column(Text, nullable=False, default="")
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AdminAuditLog(Base):
    """Immutable record of state-changing admin actions."""

    __tablename__ = "admin_audit_logs"
    __table_args__ = (
        Index("ix_admin_audit_logs_admin_username", "admin_username"),
        Index("ix_admin_audit_logs_timestamp", "timestamp"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    admin_username: Mapped[str] = mapped_column(String(255), nullable=False)
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    target: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    details_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    ip_address: Mapped[str] = mapped_column(String(128), nullable=False, default="")
