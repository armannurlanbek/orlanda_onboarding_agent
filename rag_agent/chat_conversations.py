"""Server-side storage for per-user chat conversation metadata (title + timestamps)."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from rag_agent.config import RAG_AGENT_DIR

_DATA_PATH = RAG_AGENT_DIR / "data" / "chat_conversations.json"
_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load() -> dict:
    if not _DATA_PATH.is_file():
        return {}
    try:
        raw = _DATA_PATH.read_text(encoding="utf-8")
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _save(payload: dict) -> None:
    _DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    _DATA_PATH.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def list_for_user(username: str) -> dict[str, dict]:
    user_key = str(username or "").strip().lower()
    if not user_key:
        return {}
    with _LOCK:
        data = _load()
        by_user = data.get("by_user", {})
        if not isinstance(by_user, dict):
            return {}
        conversations = by_user.get(user_key, {})
        if not isinstance(conversations, dict):
            return {}
        out: dict[str, dict] = {}
        for conv_id, meta in conversations.items():
            if not isinstance(meta, dict):
                continue
            conv_key = str(conv_id or "").strip()
            if not conv_key:
                continue
            out[conv_key] = {
                "title": str(meta.get("title") or conv_key).strip() or conv_key,
                "created_at": str(meta.get("created_at") or ""),
                "updated_at": str(meta.get("updated_at") or ""),
            }
        return out


def upsert_for_user(username: str, conversation_id: str, title: str) -> dict:
    user_key = str(username or "").strip().lower()
    conv_id = str(conversation_id or "").strip()
    conv_title = str(title or "").strip() or conv_id
    if not user_key or not conv_id:
        raise ValueError("username and conversation_id are required")
    with _LOCK:
        data = _load()
        by_user = data.setdefault("by_user", {})
        if not isinstance(by_user, dict):
            by_user = {}
            data["by_user"] = by_user
        user_conversations = by_user.setdefault(user_key, {})
        if not isinstance(user_conversations, dict):
            user_conversations = {}
            by_user[user_key] = user_conversations
        existing = user_conversations.get(conv_id, {})
        created_at = str(existing.get("created_at") or _now_iso()) if isinstance(existing, dict) else _now_iso()
        updated_at = _now_iso()
        user_conversations[conv_id] = {
            "title": conv_title,
            "created_at": created_at,
            "updated_at": updated_at,
        }
        _save(data)
        return {
            "id": conv_id,
            "title": conv_title,
            "created_at": created_at,
            "updated_at": updated_at,
        }


def delete_for_user(username: str, conversation_id: str) -> None:
    user_key = str(username or "").strip().lower()
    conv_id = str(conversation_id or "").strip()
    if not user_key or not conv_id:
        return
    with _LOCK:
        data = _load()
        by_user = data.get("by_user", {})
        if not isinstance(by_user, dict):
            return
        user_conversations = by_user.get(user_key, {})
        if not isinstance(user_conversations, dict):
            return
        if conv_id in user_conversations:
            user_conversations.pop(conv_id, None)
            _save(data)
