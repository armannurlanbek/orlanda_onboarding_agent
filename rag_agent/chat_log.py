"""
Persistent log of chat interactions: question, answer, sources, user, timestamp.
Used by the admin panel to show all questions, answers, and documents used.
"""
from datetime import datetime, timezone
import uuid

from sqlalchemy import desc, func, select

from rag_agent.db.models import ChatLogEntry
from rag_agent.db.session import get_session_factory


def _to_public(entry: ChatLogEntry) -> dict:
    return {
        "id": entry.id,
        "timestamp": entry.timestamp.isoformat() if entry.timestamp else "",
        "username": entry.username,
        "question": entry.question,
        "answer": entry.answer or "",
        "sources": entry.sources or [],
        "error": entry.error,
        "score": entry.score,
        "correct_answer": entry.correct_answer or "",
        "reviewed_at": entry.reviewed_at.isoformat() if entry.reviewed_at else "",
    }


def append(
    username: str,
    question: str,
    answer: str,
    sources: list[dict],
    error: str | None = None,
) -> None:
    """Append one chat interaction to the log."""
    with get_session_factory()() as db:
        db.add(
            ChatLogEntry(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(tz=timezone.utc),
                username=username,
                question=question,
                answer=answer,
                sources=sources or [],
                error=error,
                score=None,
                correct_answer="",
                reviewed_at=None,
            )
        )
        db.commit()


def list_entries(limit: int = 500, offset: int = 0) -> list[dict]:
    """Return log entries, newest first. limit/offset for pagination."""
    with get_session_factory()() as db:
        rows = db.execute(
            select(ChatLogEntry)
            .order_by(desc(ChatLogEntry.timestamp))
            .offset(max(0, int(offset)))
            .limit(max(1, int(limit)))
        ).scalars().all()
    return [_to_public(r) for r in rows]


def count() -> int:
    """Total number of log entries."""
    with get_session_factory()() as db:
        return int(db.scalar(select(func.count()).select_from(ChatLogEntry)) or 0)


def update_review(entry_id: str, score: int | None, correct_answer: str | None) -> dict | None:
    """Update admin review fields for one log entry by id. Returns updated entry or None."""
    with get_session_factory()() as db:
        row = db.get(ChatLogEntry, entry_id)
        if row is None:
            return None
        if score is not None:
            row.score = int(score)
        if correct_answer is not None:
            row.correct_answer = str(correct_answer)
        row.reviewed_at = datetime.now(tz=timezone.utc)
        db.commit()
        db.refresh(row)
    return _to_public(row)
