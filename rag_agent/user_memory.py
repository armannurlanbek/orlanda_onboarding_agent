"""Long-term, per-user memory: data access + system-prompt rendering.

Memories persist across all of a user's conversations (unlike the per-thread checkpointer).
The agent writes them via tools (see rag_agent.memory_tools); users and admins manage them via
the REST API. Every turn, a user's memories are rendered into a block injected into the system
prompt (build_memory_block), gated by the global RAG_MEMORY_ENABLED kill-switch AND the
per-user users.memory_enabled toggle.

v1 stores plain text and injects wholesale (cap + token budget). The UserMemory.embedding column
is the upgrade hook for semantic retrieval later — not used here.
"""
from __future__ import annotations

import logging
import re
import uuid

from sqlalchemy import delete, func, select

from rag_agent.config import (
    RAG_MAX_USER_MEMORIES,
    RAG_MEMORY_INJECT_TOKEN_BUDGET,
    memory_enabled,
)
from rag_agent.db.models import User, UserMemory
from rag_agent.db.session import get_session_factory

log = logging.getLogger(__name__)

VALID_CATEGORIES = ("fact", "preference", "task_recipe")
VALID_SOURCES = ("agent", "user", "admin")
_DEFAULT_CATEGORY = "fact"
# Rough chars-per-token used to honor the injection token budget without a tokenizer dep.
_CHARS_PER_TOKEN = 4


def _handle(mem_id: uuid.UUID) -> str:
    """Short, stable per-user handle the model/UI uses to reference a memory."""
    return mem_id.hex[:8]


def _normalize_category(category: str | None) -> str:
    c = (category or "").strip().lower()
    return c if c in VALID_CATEGORIES else _DEFAULT_CATEGORY


def _normalize_content(content: str) -> str:
    """Collapse whitespace + lowercase for the exact-duplicate guard."""
    return re.sub(r"\s+", " ", (content or "").strip()).lower()


def _coerce_user_id(user_id) -> uuid.UUID | None:
    if isinstance(user_id, uuid.UUID):
        return user_id
    try:
        return uuid.UUID(str(user_id))
    except (ValueError, TypeError, AttributeError):
        return None


def _to_public(row: UserMemory) -> dict:
    return {
        "id": _handle(row.id),
        "content": row.content,
        "category": row.category,
        "source": row.source,
        "created_at": row.created_at.isoformat() if row.created_at else "",
        "updated_at": row.updated_at.isoformat() if row.updated_at else "",
    }


def _resolve(db, uid: uuid.UUID, handle: str) -> UserMemory | None:
    """Find a user's memory by short handle or full UUID (always scoped to the user)."""
    h = (handle or "").replace("-", "").strip().lower()
    if len(h) < 4:
        return None
    rows = db.execute(select(UserMemory).where(UserMemory.user_id == uid)).scalars().all()
    for r in rows:
        if r.id.hex == h or r.id.hex.startswith(h):
            return r
    return None


# ── Per-user toggle ──────────────────────────────────────────────────────────
def get_memory_enabled(user_id) -> bool:
    """True if this user has memory enabled (independent of the global kill-switch)."""
    uid = _coerce_user_id(user_id)
    if uid is None:
        return False
    with get_session_factory()() as db:
        user = db.get(User, uid)
        return bool(user.memory_enabled) if user else False


def set_memory_enabled(user_id, enabled: bool) -> bool:
    """Set this user's memory toggle. Returns the new value (False if user not found)."""
    uid = _coerce_user_id(user_id)
    if uid is None:
        return False
    with get_session_factory()() as db:
        user = db.get(User, uid)
        if user is None:
            return False
        user.memory_enabled = bool(enabled)
        db.commit()
        return bool(user.memory_enabled)


# ── CRUD ─────────────────────────────────────────────────────────────────────
def list_memories(user_id) -> list[dict]:
    """All of a user's memories, most-recently-updated first."""
    uid = _coerce_user_id(user_id)
    if uid is None:
        return []
    with get_session_factory()() as db:
        rows = db.execute(
            select(UserMemory)
            .where(UserMemory.user_id == uid)
            .order_by(UserMemory.updated_at.desc())
        ).scalars().all()
        return [_to_public(r) for r in rows]


def count_memories(user_id) -> int:
    uid = _coerce_user_id(user_id)
    if uid is None:
        return 0
    with get_session_factory()() as db:
        return int(
            db.scalar(
                select(func.count()).select_from(UserMemory).where(UserMemory.user_id == uid)
            )
            or 0
        )


def add_memory(
    user_id,
    content: str,
    category: str = _DEFAULT_CATEGORY,
    source: str = "agent",
    thread_id: str | None = None,
) -> dict:
    """Add a memory.

    Returns {"status": "added"|"duplicate"|"full"|"invalid", "memory": <public|None>}.
    Enforces the per-user cap (RAG_MAX_USER_MEMORIES) and an exact-normalized dedup guard.
    """
    uid = _coerce_user_id(user_id)
    text = (content or "").strip()
    if uid is None or not text:
        return {"status": "invalid", "memory": None}

    cat = _normalize_category(category)
    src = (source or "agent").strip().lower()
    if src not in VALID_SOURCES:
        src = "agent"
    norm = _normalize_content(text)

    with get_session_factory()() as db:
        existing = db.execute(
            select(UserMemory).where(UserMemory.user_id == uid)
        ).scalars().all()
        for r in existing:
            if _normalize_content(r.content) == norm:
                return {"status": "duplicate", "memory": _to_public(r)}
        if len(existing) >= RAG_MAX_USER_MEMORIES:
            return {"status": "full", "memory": None}
        row = UserMemory(
            id=uuid.uuid4(),
            user_id=uid,
            content=text,
            category=cat,
            source=src,
            source_thread_id=(thread_id or None),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return {"status": "added", "memory": _to_public(row)}


def update_memory(user_id, handle: str, content: str) -> dict | None:
    """Edit a memory's content. Returns the updated public dict, or None if not found."""
    uid = _coerce_user_id(user_id)
    text = (content or "").strip()
    if uid is None or not text:
        return None
    with get_session_factory()() as db:
        row = _resolve(db, uid, handle)
        if row is None:
            return None
        row.content = text
        db.commit()
        db.refresh(row)
        return _to_public(row)


def delete_memory(user_id, handle: str) -> bool:
    """Delete one memory. Returns True if found and deleted."""
    uid = _coerce_user_id(user_id)
    if uid is None:
        return False
    with get_session_factory()() as db:
        row = _resolve(db, uid, handle)
        if row is None:
            return False
        db.delete(row)
        db.commit()
        return True


def clear_memories(user_id) -> int:
    """Delete all of a user's memories. Returns the number deleted."""
    uid = _coerce_user_id(user_id)
    if uid is None:
        return 0
    with get_session_factory()() as db:
        result = db.execute(delete(UserMemory).where(UserMemory.user_id == uid))
        db.commit()
        return int(result.rowcount or 0)


# ── System-prompt rendering ────────────────────────────────────────────────────
def build_memory_block(user_id) -> str | None:
    """Render a user's memories into a system-prompt block, or None if memory is off/empty.

    Honors the global kill-switch, the per-user toggle, the count cap, and the token budget
    (most-recently-updated memories win when the budget is exceeded).
    """
    if not memory_enabled():
        return None
    uid = _coerce_user_id(user_id)
    if uid is None:
        return None

    with get_session_factory()() as db:
        user = db.get(User, uid)
        if user is None or not user.memory_enabled:
            return None
        rows = db.execute(
            select(UserMemory)
            .where(UserMemory.user_id == uid)
            .order_by(UserMemory.updated_at.desc())
            .limit(RAG_MAX_USER_MEMORIES)
        ).scalars().all()

    if not rows:
        return None

    char_budget = RAG_MEMORY_INJECT_TOKEN_BUDGET * _CHARS_PER_TOKEN
    lines: list[str] = []
    used = 0
    for r in rows:
        line = f"[mem_{_handle(r.id)}] ({r.category}) {r.content}".strip()
        if used + len(line) > char_budget and lines:
            break
        lines.append(line)
        used += len(line) + 1

    header = (
        "## Долгосрочная память о пользователе\n"
        "Факты, предпочтения и проверенные рецепты задач, которые ты сохранил ранее. "
        "Используй их, чтобы не переспрашивать и сразу применять рабочие решения. "
        "Ссылайся на запись по её [mem_xxxx] при обновлении или удалении."
    )
    return header + "\n" + "\n".join(lines)
