"""Add user_memories table and users.memory_enabled (long-term cross-conversation memory).

Revision ID: 011_user_memories
Revises: 010_user_monday_oauth
Create Date: 2026-06-15

Per-user long-term memories the agent writes via tools and that get injected into the
system prompt on every turn, so knowledge persists across separate conversations. The
``embedding`` column is nullable and unused in v1 (upgrade hook for semantic retrieval).
The pgvector extension is already created by revision 006_pgvector_rag_index.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "011_user_memories"
down_revision: Union[str, None] = "010_user_monday_oauth"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "memory_enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
    )

    op.create_table(
        "user_memories",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("category", sa.String(length=16), nullable=False, server_default="fact"),
        sa.Column("source", sa.String(length=16), nullable=False, server_default="agent"),
        sa.Column("source_thread_id", sa.String(length=512), nullable=True),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_user_memories_user_id", "user_memories", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_user_memories_user_id", table_name="user_memories")
    op.drop_table("user_memories")
    op.drop_column("users", "memory_enabled")
