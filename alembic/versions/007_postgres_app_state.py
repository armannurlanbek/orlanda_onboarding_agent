"""Add PostgreSQL tables for app mutable state.

Revision ID: 007_postgres_app_state
Revises: 006_pgvector_rag_index
Create Date: 2026-04-22
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "007_postgres_app_state"
down_revision: Union[str, None] = "006_pgvector_rag_index"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "knowledge_items",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("content", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("last_updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("update_period_days", sa.Integer(), nullable=True),
        sa.Column("responsible", sa.String(length=255), nullable=False, server_default=""),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "pdf_metadata",
        sa.Column("path", sa.String(length=1024), nullable=False),
        sa.Column("last_updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("update_period_days", sa.Integer(), nullable=True),
        sa.Column("responsible", sa.String(length=255), nullable=False, server_default=""),
        sa.PrimaryKeyConstraint("path"),
    )

    op.create_table(
        "chat_logs",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("username", sa.String(length=255), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False, server_default=""),
        sa.Column("sources", sa.JSON(), nullable=False, server_default=sa.text("'[]'::json")),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("score", sa.Integer(), nullable=True),
        sa.Column("correct_answer", sa.Text(), nullable=False, server_default=""),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_chat_logs_timestamp", "chat_logs", ["timestamp"], unique=False)
    op.create_index("ix_chat_logs_username", "chat_logs", ["username"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_chat_logs_username", table_name="chat_logs")
    op.drop_index("ix_chat_logs_timestamp", table_name="chat_logs")
    op.drop_table("chat_logs")
    op.drop_table("pdf_metadata")
    op.drop_table("knowledge_items")