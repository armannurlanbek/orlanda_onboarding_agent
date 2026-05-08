"""Admin action audit log — write/query helpers."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import desc, func, select

from rag_agent.db.models import AdminAuditLog
from rag_agent.db.session import get_session_factory

logger = logging.getLogger(__name__)


def _to_public(row: AdminAuditLog) -> dict:
    return {
        "id": row.id,
        "timestamp": row.timestamp.isoformat() if row.timestamp else "",
        "admin_username": row.admin_username,
        "action": row.action,
        "target": row.target or "",
        "details": row.details_json or {},
        "ip_address": row.ip_address or "",
    }


def write_audit(
    action: str,
    admin_username: str,
    target: str | None = None,
    details: dict[str, Any] | None = None,
    ip_address: str | None = None,
) -> None:
    """Record one admin action. Never raises — audit failures must not block the caller."""
    try:
        with get_session_factory()() as db:
            db.add(
                AdminAuditLog(
                    id=str(uuid.uuid4()),
                    timestamp=datetime.now(tz=timezone.utc),
                    admin_username=admin_username,
                    action=action,
                    target=target or "",
                    details_json=details or {},
                    ip_address=ip_address or "",
                )
            )
            db.commit()
    except Exception as exc:
        logger.warning("audit_log write failed (non-fatal): %s", exc)


def list_audit(limit: int = 100, offset: int = 0) -> list[dict]:
    """Return audit log entries newest-first."""
    try:
        with get_session_factory()() as db:
            rows = db.execute(
                select(AdminAuditLog)
                .order_by(desc(AdminAuditLog.timestamp))
                .offset(max(0, int(offset)))
                .limit(max(1, int(limit)))
            ).scalars().all()
        return [_to_public(r) for r in rows]
    except Exception as exc:
        logger.warning("audit_log list failed: %s", exc)
        return []


def count_audit() -> int:
    """Total number of audit log entries."""
    try:
        with get_session_factory()() as db:
            return int(db.scalar(select(func.count()).select_from(AdminAuditLog)) or 0)
    except Exception:
        return 0
