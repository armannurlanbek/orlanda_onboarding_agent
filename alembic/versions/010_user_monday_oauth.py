"""Add user_monday_oauth table (per-user monday.com OAuth tokens).

Revision ID: 010_user_monday_oauth
Revises: 009_drop_monday_tables
Create Date: 2026-06-12

Stores one encrypted monday.com OAuth access token per user for the per-user MCP
integration. The access token is encrypted at rest (see rag_agent/crypto.py). monday OAuth
tokens do not expire and have no refresh token, so only the access token + granted scope
are kept. One row per user (unique user_id), deleted when the user is deleted.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "010_user_monday_oauth"
down_revision: Union[str, None] = "009_drop_monday_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "user_monday_oauth",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("access_token_encrypted", sa.Text(), nullable=False),
        sa.Column("scope", sa.Text(), nullable=False, server_default=""),
        sa.Column("token_type", sa.String(length=32), nullable=False, server_default="Bearer"),
        sa.Column("monday_account_id", sa.String(length=64), nullable=True),
        sa.Column("monday_user_name", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    # Unique index matches the ORM model (user_id unique=True, index=True): one row/user.
    op.create_index(
        "ix_user_monday_oauth_user_id",
        "user_monday_oauth",
        ["user_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_user_monday_oauth_user_id", table_name="user_monday_oauth")
    op.drop_table("user_monday_oauth")
