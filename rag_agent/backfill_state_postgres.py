"""
Backfill legacy JSON app state into PostgreSQL tables.

Covers:
- rag_agent/data/knowledge_items.json -> knowledge_items
- rag_agent/data/pdf_metadata.json -> pdf_metadata
- rag_agent/data/chat_log.json -> chat_logs
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import select

from rag_agent.config import RAG_AGENT_DIR
from rag_agent.db.models import ChatLogEntry, KnowledgeItemRecord, PdfMetadataRecord
from rag_agent.db.session import get_session_factory


DATA_DIR = RAG_AGENT_DIR / "data"
KNOWLEDGE_ITEMS_FILE = DATA_DIR / "knowledge_items.json"
PDF_METADATA_FILE = DATA_DIR / "pdf_metadata.json"
CHAT_LOG_FILE = DATA_DIR / "chat_log.json"


def _parse_dt(value: str | None):
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _load_json(path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def backfill_knowledge_items() -> tuple[int, int]:
    raw = _load_json(KNOWLEDGE_ITEMS_FILE)
    if not isinstance(raw, list):
        return 0, 0
    added = 0
    skipped = 0
    with get_session_factory()() as db:
        existing_ids = set(db.execute(select(KnowledgeItemRecord.id)).scalars().all())
        for it in raw:
            if not isinstance(it, dict):
                continue
            item_id = str(it.get("id") or "").strip() or str(uuid.uuid4())
            if item_id in existing_ids:
                skipped += 1
                continue
            db.add(
                KnowledgeItemRecord(
                    id=item_id,
                    name=(str(it.get("name") or "").strip() or "Без названия"),
                    content=str(it.get("content") or ""),
                    created_at=_parse_dt(it.get("created_at")) or datetime.now(timezone.utc),
                    last_updated_at=_parse_dt(it.get("last_updated_at")) or datetime.now(timezone.utc),
                    update_period_days=it.get("update_period_days"),
                    responsible=str(it.get("responsible") or ""),
                )
            )
            added += 1
        db.commit()
    return added, skipped


def backfill_pdf_metadata() -> tuple[int, int]:
    raw = _load_json(PDF_METADATA_FILE)
    pdfs = (raw or {}).get("pdfs") if isinstance(raw, dict) else None
    if not isinstance(pdfs, dict):
        return 0, 0
    added = 0
    skipped = 0
    with get_session_factory()() as db:
        existing = set(db.execute(select(PdfMetadataRecord.path)).scalars().all())
        for rel, meta in pdfs.items():
            path = str(rel or "").strip().replace("\\", "/").lstrip("/")
            if not path:
                continue
            if path in existing:
                skipped += 1
                continue
            meta = meta if isinstance(meta, dict) else {}
            db.add(
                PdfMetadataRecord(
                    path=path,
                    last_updated_at=_parse_dt(meta.get("last_updated_at")),
                    update_period_days=meta.get("update_period_days"),
                    responsible=str(meta.get("responsible") or ""),
                )
            )
            added += 1
        db.commit()
    return added, skipped


def backfill_chat_logs() -> tuple[int, int]:
    raw = _load_json(CHAT_LOG_FILE)
    if not isinstance(raw, list):
        return 0, 0
    added = 0
    skipped = 0
    with get_session_factory()() as db:
        existing = set(db.execute(select(ChatLogEntry.id)).scalars().all())
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            entry_id = str(entry.get("id") or "").strip() or str(uuid.uuid4())
            if entry_id in existing:
                skipped += 1
                continue
            sources = entry.get("sources")
            if not isinstance(sources, list):
                sources = []
            db.add(
                ChatLogEntry(
                    id=entry_id,
                    timestamp=_parse_dt(entry.get("timestamp")) or datetime.now(timezone.utc),
                    username=str(entry.get("username") or ""),
                    question=str(entry.get("question") or ""),
                    answer=str(entry.get("answer") or ""),
                    sources=sources,
                    error=(str(entry.get("error")) if entry.get("error") is not None else None),
                    score=(int(entry.get("score")) if entry.get("score") is not None else None),
                    correct_answer=str(entry.get("correct_answer") or ""),
                    reviewed_at=_parse_dt(entry.get("reviewed_at")),
                )
            )
            added += 1
        db.commit()
    return added, skipped


def main() -> None:
    ki_added, ki_skipped = backfill_knowledge_items()
    pdf_added, pdf_skipped = backfill_pdf_metadata()
    log_added, log_skipped = backfill_chat_logs()
    print(
        "State backfill complete: "
        f"knowledge_items added={ki_added} skipped={ki_skipped}; "
        f"pdf_metadata added={pdf_added} skipped={pdf_skipped}; "
        f"chat_logs added={log_added} skipped={log_skipped}"
    )


if __name__ == "__main__":
    main()
