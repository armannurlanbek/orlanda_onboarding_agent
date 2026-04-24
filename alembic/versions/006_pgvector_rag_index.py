"""Add pgvector-backed RAG indexing tables.

Revision ID: 006_pgvector_rag_index
Revises: 004_user_password_lifecycle
Create Date: 2026-04-22
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "006_pgvector_rag_index"
down_revision: Union[str, None] = "004_user_password_lifecycle"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # pgvector must be installed on PostgreSQL host package level first.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "rag_documents",
        sa.Column("doc_id", sa.String(length=512), nullable=False),
        sa.Column("doc_type", sa.String(length=32), nullable=False),
        sa.Column("source_ref", sa.String(length=1024), nullable=False),
        sa.Column("source_name", sa.String(length=512), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("doc_id"),
    )
    op.create_index(op.f("ix_rag_documents_doc_type"), "rag_documents", ["doc_type"], unique=False)
    op.create_index(op.f("ix_rag_documents_source_ref"), "rag_documents", ["source_ref"], unique=True)

    op.create_table(
        "rag_document_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("doc_id", sa.String(length=512), nullable=False),
        sa.Column("chunk_no", sa.Integer(), nullable=False),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("search_text", sa.Text(), nullable=False, server_default=""),
        sa.Column("source_file", sa.String(length=512), nullable=False),
        sa.Column("page", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("embedding", Vector(1536), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["doc_id"], ["rag_documents.doc_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("doc_id", "chunk_no", name="uq_rag_document_chunks_doc_chunkno"),
    )
    op.create_index("ix_rag_document_chunks_doc_id", "rag_document_chunks", ["doc_id"], unique=False)
    op.create_index("ix_rag_document_chunks_source_file", "rag_document_chunks", ["source_file"], unique=False)
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_rag_document_chunks_search_text_tsv "
        "ON rag_document_chunks USING GIN (to_tsvector('simple', search_text))"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_rag_document_chunks_embedding_hnsw "
        "ON rag_document_chunks USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_rag_document_chunks_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_rag_document_chunks_search_text_tsv")
    op.drop_index("ix_rag_document_chunks_source_file", table_name="rag_document_chunks")
    op.drop_index("ix_rag_document_chunks_doc_id", table_name="rag_document_chunks")
    op.drop_table("rag_document_chunks")
    op.drop_index(op.f("ix_rag_documents_source_ref"), table_name="rag_documents")
    op.drop_index(op.f("ix_rag_documents_doc_type"), table_name="rag_documents")
    op.drop_table("rag_documents")