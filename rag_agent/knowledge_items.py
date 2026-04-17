"""
Text knowledge items: name + content, stored in JSON and included in RAG index.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from datetime import datetime, timezone

from langchain_core.documents import Document

from rag_agent.config import RAG_AGENT_DIR

ITEMS_FILE = RAG_AGENT_DIR / "data" / "knowledge_items.json"

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


def _load_raw() -> list[dict]:
    if not ITEMS_FILE.is_file():
        return []
    try:
        data = json.loads(ITEMS_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []
        return data
    except (json.JSONDecodeError, OSError):
        return []


def _save_raw(items: list[dict]) -> None:
    ITEMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    ITEMS_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def list_items() -> list[dict]:
    """Return all items with metadata fields (backward compatible)."""
    return [_normalize_item(it) for it in _load_raw()]


def get_item(item_id: str) -> dict | None:
    """Return one item by id or None."""
    for it in _load_raw():
        if it.get("id") == item_id:
            return _normalize_item(it)
    return None


def add_item(name: str, content: str, update_period_days: int | None = None, responsible: str = "") -> dict:
    """Create item and append to store."""
    name = (name or "").strip() or "Без названия"
    items = _load_raw()
    now = _now_iso()
    item = {
        "id": str(uuid.uuid4()),
        "name": name,
        "content": (content or "").strip(),
        "created_at": now,
        "last_updated_at": now,
        "update_period_days": update_period_days,
        "responsible": responsible or "",
    }
    items.append(item)
    _save_raw(items)
    return item


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
    items = _load_raw()
    for i, it in enumerate(items):
        if it.get("id") == item_id:
            if name is not None:
                items[i]["name"] = (name or "").strip() or "Без названия"
            if content is not None:
                items[i]["content"] = (content or "").strip()
            if update_period_days is not UNSET:
                items[i]["update_period_days"] = update_period_days
            if touch_last_updated_at:
                items[i]["last_updated_at"] = _now_iso()
            # Backward compatible: ensure metadata fields exist for older items.
            for k, v in _defaults().items():
                items[i].setdefault(k, v)
            _save_raw(items)
            return items[i]
    return None


def delete_item(item_id: str) -> bool:
    """Remove item by id. Return True if found and deleted."""
    items = _load_raw()
    for i, it in enumerate(items):
        if it.get("id") == item_id:
            items.pop(i)
            _save_raw(items)
            return True
    return False


def items_to_documents() -> list[Document]:
    """Convert all stored items to LangChain Documents for indexing (source_file = name for RAG display)."""
    docs = []
    for it in _load_raw():
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
