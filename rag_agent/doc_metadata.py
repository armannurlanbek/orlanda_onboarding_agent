"""
Document metadata storage for admin workflows.

We store PDF metadata (last update, update period, responsible) in PostgreSQL.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from rag_agent.db.models import PdfMetadataRecord
from rag_agent.db.session import get_session_factory


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _to_public(row: PdfMetadataRecord | None) -> dict[str, Any]:
    if row is None:
        return {"last_updated_at": "", "update_period_days": None, "responsible": ""}
    return {
        "last_updated_at": row.last_updated_at.isoformat() if row.last_updated_at else "",
        "update_period_days": row.update_period_days,
        "responsible": row.responsible or "",
    }


def _normalize_rel_path(rel_path: str) -> str:
    rel_path = (rel_path or "").strip().replace("\\", "/").lstrip("/")
    return rel_path


def get_pdf_metadata(rel_path: str) -> dict[str, Any]:
    """
    Return PDF metadata with defaults:
    - last_updated_at: ISO string or "" (unknown)
    - update_period_days: int | None
    - responsible: username or "" (unknown)
    """
    rel_path = _normalize_rel_path(rel_path)
    with get_session_factory()() as db:
        row = db.get(PdfMetadataRecord, rel_path)
    return _to_public(row)


def record_pdf_upload(rel_path: str, *, responsible: str, update_period_days: int | None) -> dict[str, Any]:
    """Set last_updated_at=now, responsible=caller, and update_period_days (can be null)."""
    rel_path = _normalize_rel_path(rel_path)
    now = _now_iso()
    dt = datetime.fromisoformat(now)
    with get_session_factory()() as db:
        row = db.get(PdfMetadataRecord, rel_path)
        if row is None:
            row = PdfMetadataRecord(path=rel_path)
            db.add(row)
        row.last_updated_at = dt
        row.update_period_days = update_period_days
        row.responsible = responsible or ""
        db.commit()
        db.refresh(row)
    return _to_public(row)


def set_pdf_update_period(rel_path: str, *, update_period_days: int | None) -> dict[str, Any]:
    """Update only update_period_days (does not touch last_updated_at/responsible)."""
    rel_path = _normalize_rel_path(rel_path)
    with get_session_factory()() as db:
        row = db.get(PdfMetadataRecord, rel_path)
        if row is None:
            row = PdfMetadataRecord(
                path=rel_path,
                last_updated_at=None,
                update_period_days=update_period_days,
                responsible="",
            )
            db.add(row)
        else:
            row.update_period_days = update_period_days
        db.commit()
        db.refresh(row)
    return _to_public(row)


def delete_pdf_metadata(rel_path: str) -> None:
    rel_path = _normalize_rel_path(rel_path)
    with get_session_factory()() as db:
        row = db.get(PdfMetadataRecord, rel_path)
        if row is not None:
            db.delete(row)
            db.commit()


def rename_pdf_metadata(old_rel: str, new_rel: str) -> None:
    """Move metadata entry when a PDF file is renamed on disk (e.g. to *_changed.pdf)."""
    old_rel = _normalize_rel_path(old_rel)
    new_rel = _normalize_rel_path(new_rel)
    if old_rel == new_rel:
        return
    with get_session_factory()() as db:
        row = db.get(PdfMetadataRecord, old_rel)
        if row is None:
            return
        existing_new = db.get(PdfMetadataRecord, new_rel)
        if existing_new is not None:
            db.delete(existing_new)
            db.flush()
        row.path = new_rel
        db.commit()


def _parse_iso_maybe(iso: str) -> datetime | None:
    if not iso:
        return None
    try:
        # datetime.fromisoformat can parse timezone-aware ISO strings
        return datetime.fromisoformat(iso)
    except ValueError:
        return None


def compute_expiry(last_updated_at: str | None, update_period_days: int | None) -> dict[str, Any]:
    """
    Compute expiry:
    - expires_at: ISO string or "".
    - expired: bool.
    Expired is only meaningful when both last_updated_at and update_period_days are present.
    """
    dt = _parse_iso_maybe(last_updated_at or "")
    if dt is None or update_period_days is None:
        return {"expires_at": "", "expired": False}
    from datetime import timedelta

    expires = dt + timedelta(days=int(update_period_days))
    return {"expires_at": expires.isoformat(), "expired": datetime.now(tz=timezone.utc) > expires}


