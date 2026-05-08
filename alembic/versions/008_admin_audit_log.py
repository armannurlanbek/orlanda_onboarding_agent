"""Add admin_audit_logs table.

Revision ID: 008_admin_audit_log
Revises: 007_postgres_app_state
Create Date: 2026-05-07
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "008_admin_audit_log"
down_revision: Union[str, None] = "007_postgres_app_state"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "admin_audit_logs",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("admin_username", sa.String(length=255), nullable=False),
        sa.Column("action", sa.String(length=128), nullable=False),
        sa.Column("target", sa.String(length=512), nullable=False, server_default=""),
        sa.Column(
            "details_json",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'::json"),
        ),
        sa.Column("ip_address", sa.String(length=128), nullable=False, server_default=""),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_admin_audit_logs_admin_username",
        "admin_audit_logs",
        ["admin_username"],
        unique=False,
    )
    op.create_index(
        "ix_admin_audit_logs_timestamp",
        "admin_audit_logs",
        ["timestamp"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_admin_audit_logs_timestamp", table_name="admin_audit_logs")
    op.drop_index("ix_admin_audit_logs_admin_username", table_name="admin_audit_logs")
    op.drop_table("admin_audit_logs")
