"""Add client_invites table (client portal invite links).

Revision ID: 012_client_invites
Revises: 011_user_memories
Create Date: 2026-07-09

Invite links for external-client registration: each link bakes in the
OrlandaBot project ids the client gets access to. Registering through a link
creates a users row with role="client" (no schema change needed — role is a
free string column) and provisions project access in orlanda-api.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "012_client_invites"
down_revision: Union[str, None] = "011_user_memories"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "client_invites",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("token", sa.String(length=64), nullable=False),
        sa.Column("company_name", sa.String(length=255), nullable=False, server_default=""),
        sa.Column("project_ids", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("project_names", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("max_uses", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("used_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by", sa.String(length=255), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_client_invites_token", "client_invites", ["token"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_client_invites_token", table_name="client_invites")
    op.drop_table("client_invites")
