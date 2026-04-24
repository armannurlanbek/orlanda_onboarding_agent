"""
Text knowledge items: name + content, stored in PostgreSQL and included in RAG index.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from langchain_core.documents import Document
from sqlalchemy import select

from rag_agent.db.models import KnowledgeItemRecord
from rag_agent.db.session import get_session_factory

UNSET = object()


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _defaults() -> dict:
    return {
        "created_at": "",
        "last_updated_at": "",
        "update_period_days": None,
        "responsible": "",
    }


def _normalize_item(it: dict) -> dict:
    if not isinstance(it, dict):
        return dict(_defaults())
    out = dict(it)
    for k, v in _defaults().items():
        out.setdefault(k, v)
    return out


def _to_public(it: KnowledgeItemRecord) -> dict:
    return {
        "id": it.id,
        "name": it.name,
        "content": it.content or "",
        "created_at": it.created_at.isoformat() if it.created_at else "",
        "last_updated_at": it.last_updated_at.isoformat() if it.last_updated_at else "",
        "update_period_days": it.update_period_days,
        "responsible": it.responsible or "",
    }


def list_items() -> list[dict]:
    """Return all items with metadata fields (backward compatible)."""
    with get_session_factory()() as db:
        rows = db.execute(select(KnowledgeItemRecord).order_by(KnowledgeItemRecord.created_at.asc())).scalars().all()
    return [_normalize_item(_to_public(it)) for it in rows]


def get_item(item_id: str) -> dict | None:
    """Return one item by id or None."""
    with get_session_factory()() as db:
        row = db.get(KnowledgeItemRecord, item_id)
    return _normalize_item(_to_public(row)) if row else None


def add_item(name: str, content: str, update_period_days: int | None = None, responsible: str = "") -> dict:
    """Create item and append to store."""
    name = (name or "").strip() or "Без названия"
    now = _now_iso()
    item_id = str(uuid.uuid4())
    dt = datetime.fromisoformat(now)
    row = KnowledgeItemRecord(
        id=item_id,
        name=name,
        content=(content or "").strip(),
        created_at=dt,
        last_updated_at=dt,
        update_period_days=update_period_days,
        responsible=responsible or "",
    )
    with get_session_factory()() as db:
        db.add(row)
        db.commit()
    return _to_public(row)


def update_item(
    item_id: str,
    name: str | None = None,
    content: str | None = None,
    *,
    update_period_days: int | None | object = UNSET,
    touch_last_updated_at: bool = False,
) -> dict | None:
    """
    Update item by id.
    - `name` / `content` are None => leave unchanged
    - `update_period_days` is UNSET => leave unchanged, otherwise set (including null)
    - `touch_last_updated_at` controls whether last_updated_at is updated (content/name edits)
    """
    with get_session_factory()() as db:
        row = db.get(KnowledgeItemRecord, item_id)
        if row is None:
            return None
        if name is not None:
            row.name = (name or "").strip() or "Без названия"
        if content is not None:
            row.content = (content or "").strip()
        if update_period_days is not UNSET:
            row.update_period_days = update_period_days
        if touch_last_updated_at:
            row.last_updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(row)
    return _to_public(row)


def delete_item(item_id: str) -> bool:
    """Remove item by id. Return True if found and deleted."""
    with get_session_factory()() as db:
        row = db.get(KnowledgeItemRecord, item_id)
        if row is None:
            return False
        db.delete(row)
        db.commit()
    return True


def items_to_documents() -> list[Document]:
    """Convert all stored items to LangChain Documents for indexing (source_file = name for RAG display)."""
    docs = []
    for it in list_items():
        name = it.get("name") or "Без названия"
        content = (it.get("content") or "").strip()
        if not content:
            continue
        doc = Document(
            page_content=content,
            metadata={"source_file": name, "page": 0, "type": "knowledge_item", "id": it.get("id", "")},
        )
        docs.append(doc)
    return docs
