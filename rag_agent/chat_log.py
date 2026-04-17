"""
Persistent log of chat interactions: question, answer, sources, user, timestamp.
Used by the admin panel to show all questions, answers, and documents used.
"""
import json
from datetime import datetime, timezone
import uuid

from rag_agent.config import RAG_AGENT_DIR

LOG_FILE = RAG_AGENT_DIR / "data" / "chat_log.json"


def _load_entries() -> list[dict]:
    if not LOG_FILE.is_file():
        return []
    try:
        data = json.loads(LOG_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _ensure_entry_fields(entries: list[dict]) -> tuple[list[dict], bool]:
    """
    Ensure each log entry has stable id and review fields.
    Returns (entries, changed).
    """
    changed = False
    for e in entries:
        if not isinstance(e, dict):
            continue
        if not e.get("id"):
            e["id"] = str(uuid.uuid4())
            changed = True
        if "score" not in e:
            e["score"] = None
            changed = True
        if "correct_answer" not in e:
            e["correct_answer"] = ""
            changed = True
        if "reviewed_at" not in e:
            e["reviewed_at"] = ""
            changed = True
    return entries, changed


def _save_entries(entries: list[dict]) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def append(
    username: str,
    question: str,
    answer: str,
    sources: list[dict],
    error: str | None = None,
) -> None:
    """Append one chat interaction to the log."""
    entries = _load_entries()
    entries.append({
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "username": username,
        "question": question,
        "answer": answer,
        "sources": sources,
        "error": error,
        "score": None,
        "correct_answer": "",
        "reviewed_at": "",
    })
    _save_entries(entries)


def list_entries(limit: int = 500, offset: int = 0) -> list[dict]:
    """Return log entries, newest first. limit/offset for pagination."""
    entries = _load_entries()
    entries, changed = _ensure_entry_fields(entries)
    if changed:
        _save_entries(entries)
    entries.reverse()
    return entries[offset : offset + limit]


def count() -> int:
    """Total number of log entries."""
    return len(_load_entries())


def update_review(entry_id: str, score: int | None, correct_answer: str | None) -> dict | None:
    """Update admin review fields for one log entry by id. Returns updated entry or None."""
    entries = _load_entries()
    entries, changed = _ensure_entry_fields(entries)
    if changed:
        _save_entries(entries)

    for i, entry in enumerate(entries):
        if entry.get("id") != entry_id:
            continue
        if score is not None:
            entries[i]["score"] = int(score)
        if correct_answer is not None:
            entries[i]["correct_answer"] = str(correct_answer)
        entries[i]["reviewed_at"] = datetime.now(tz=timezone.utc).isoformat()
        _save_entries(entries)
        return entries[i]
    return None
