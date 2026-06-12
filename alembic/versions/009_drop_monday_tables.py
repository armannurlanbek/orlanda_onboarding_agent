"""Drop all Monday integration tables (integration fully removed).

Revision ID: 009_drop_monday_tables
Revises: 008_admin_audit_log
Create Date: 2026-06-12

Removes the per-user Monday OAuth/token tables. `monday_user_connections` and
`monday_connection_states` are the tables the (now-deleted) integration used at
runtime; they were created out-of-band on the server and have no committed
migration creating them, so DROP ... IF EXISTS is required. The legacy
`monday_connections` / `monday_oauth_states` (created by 004, dropped by 005) are
included defensively in case any environment still has them.

This is a one-way teardown: the ORM models and integration code no longer exist,
so downgrade() intentionally does nothing.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "009_drop_monday_tables"
down_revision: Union[str, None] = "008_admin_audit_log"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("DROP TABLE IF EXISTS monday_user_connections CASCADE")
    op.execute("DROP TABLE IF EXISTS monday_connection_states CASCADE")
    # Legacy tables (already dropped by 005 in normal histories); safe no-ops otherwise.
    op.execute("DROP TABLE IF EXISTS monday_oauth_states CASCADE")
    op.execute("DROP TABLE IF EXISTS monday_connections CASCADE")


def downgrade() -> None:
    # Monday integration was fully removed; there is nothing to recreate.
    pass
