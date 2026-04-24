"""Add password lifecycle fields to users

Revision ID: 004_user_password_lifecycle
Revises: 005_drop_monday_integration
Create Date: 2026-04-21

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "004_user_password_lifecycle"
down_revision: Union[str, None] = "005_drop_monday_integration"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("users", sa.Column("must_change_password", sa.Boolean(), nullable=False, server_default=sa.text("false")))
    op.add_column("users", sa.Column("password_changed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("users", sa.Column("temp_password_issued_at", sa.DateTime(timezone=True), nullable=True))

    # Existing users keep normal access until explicitly provisioned with temporary passwords.
    op.execute("UPDATE users SET must_change_password = false WHERE must_change_password IS NULL")

    # Keep DB defaults simple; application controls runtime value for new rows.
    op.alter_column("users", "must_change_password", server_default=None)


def downgrade() -> None:
    op.drop_column("users", "temp_password_issued_at")
    op.drop_column("users", "password_changed_at")
    op.drop_column("users", "must_change_password")
