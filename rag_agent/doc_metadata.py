"""
Document metadata storage for admin workflows.

We store PDF metadata (last update, update period, responsible) in a JSON file
because PDFs themselves live on disk and we want editable extra fields.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any

from rag_agent.config import RAG_AGENT_DIR


DATA_DIR = RAG_AGENT_DIR / "data"
PDF_METADATA_FILE = DATA_DIR / "pdf_metadata.json"


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _load_raw() -> dict[str, Any]:
    if not PDF_METADATA_FILE.is_file():
        return {"pdfs": {}}
    try:
        data = json.loads(PDF_METADATA_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"pdfs": {}}
    except (json.JSONDecodeError, OSError):
        return {"pdfs": {}}


def _save_raw(data: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PDF_METADATA_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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
    raw = _load_raw()
    meta = (raw.get("pdfs") or {}).get(rel_path) or {}
    return {
        "last_updated_at": meta.get("last_updated_at") or "",
        "update_period_days": meta.get("update_period_days", None),
        "responsible": meta.get("responsible") or "",
    }


def record_pdf_upload(rel_path: str, *, responsible: str, update_period_days: int | None) -> dict[str, Any]:
    """Set last_updated_at=now, responsible=caller, and update_period_days (can be null)."""
    rel_path = _normalize_rel_path(rel_path)
    raw = _load_raw()
    raw.setdefault("pdfs", {})
    now = _now_iso()
    meta = {
        "last_updated_at": now,
        "update_period_days": update_period_days,
        "responsible": responsible or "",
    }
    raw["pdfs"][rel_path] = meta
    _save_raw(raw)
    return meta


def set_pdf_update_period(rel_path: str, *, update_period_days: int | None) -> dict[str, Any]:
    """Update only update_period_days (does not touch last_updated_at/responsible)."""
    rel_path = _normalize_rel_path(rel_path)
    raw = _load_raw()
    raw.setdefault("pdfs", {})
    meta = raw["pdfs"].get(rel_path) or {"last_updated_at": "", "update_period_days": None, "responsible": ""}
    meta["update_period_days"] = update_period_days
    raw["pdfs"][rel_path] = meta
    _save_raw(raw)
    return meta


def delete_pdf_metadata(rel_path: str) -> None:
    rel_path = _normalize_rel_path(rel_path)
    raw = _load_raw()
    pdfs = raw.get("pdfs") or {}
    if rel_path in pdfs:
        pdfs.pop(rel_path, None)
        raw["pdfs"] = pdfs
        _save_raw(raw)


def rename_pdf_metadata(old_rel: str, new_rel: str) -> None:
    """Move metadata entry when a PDF file is renamed on disk (e.g. to *_changed.pdf)."""
    old_rel = _normalize_rel_path(old_rel)
    new_rel = _normalize_rel_path(new_rel)
    if old_rel == new_rel:
        return
    raw = _load_raw()
    raw.setdefault("pdfs", {})
    pdfs = raw.get("pdfs") or {}
    if old_rel not in pdfs:
        return
    meta = pdfs.pop(old_rel)
    pdfs[new_rel] = meta
    raw["pdfs"] = pdfs
    _save_raw(raw)


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


