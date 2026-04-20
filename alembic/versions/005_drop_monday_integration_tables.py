"""Drop Monday integration tables.

Revision ID: 005_drop_monday_integration
Revises: 004_monday_integration
Create Date: 2026-04-20
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "005_drop_monday_integration"
down_revision: Union[str, None] = "004_monday_integration"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_index(op.f("ix_monday_oauth_states_user_id"), table_name="monday_oauth_states")
    op.drop_table("monday_oauth_states")
    op.drop_index(op.f("ix_monday_connections_user_id"), table_name="monday_connections")
    op.drop_table("monday_connections")


def downgrade() -> None:
    op.create_table(
        "monday_connections",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False, server_default="hosted_mcp"),
        sa.Column("encrypted_payload", sa.Text(), nullable=False),
        sa.Column("allow_mutations", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("connected_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_monday_connections_user_id"), "monday_connections", ["user_id"], unique=True)

    op.create_table(
        "monday_oauth_states",
        sa.Column("state", sa.String(length=255), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("expires_at_epoch", sa.BigInteger(), nullable=False),
        sa.Column("consumed_at_epoch", sa.BigInteger(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("state"),
    )
    op.create_index(op.f("ix_monday_oauth_states_user_id"), "monday_oauth_states", ["user_id"], unique=False)
