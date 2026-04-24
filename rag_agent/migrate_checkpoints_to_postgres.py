"""
Migrate LangGraph checkpoints from SQLite to PostgreSQL saver.

Usage:
    python -m rag_agent.migrate_checkpoints_to_postgres
"""
from __future__ import annotations

from collections import defaultdict

from rag_agent.config import CHECKPOINT_DB, CHECKPOINT_POSTGRES_URL, DATABASE_URL


def _postgres_checkpoint_dsn(raw_url: str) -> str:
    url = (raw_url or "").strip()
    if url.startswith("postgresql+psycopg://"):
        return "postgresql://" + url[len("postgresql+psycopg://") :]
    return url


def main() -> None:
    sqlite_path = (CHECKPOINT_DB or "").strip()
    pg_url = _postgres_checkpoint_dsn(CHECKPOINT_POSTGRES_URL or DATABASE_URL or "")
    if not sqlite_path:
        raise RuntimeError("CHECKPOINT_DB (sqlite path) is not set.")
    if not pg_url:
        raise RuntimeError("CHECKPOINT_POSTGRES_URL or DATABASE_URL is required.")

    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.checkpoint.sqlite import SqliteSaver

    src_cm = SqliteSaver.from_conn_string(sqlite_path)
    dst_cm = PostgresSaver.from_conn_string(pg_url)
    src = src_cm.__enter__()
    dst = dst_cm.__enter__()
    setup_fn = getattr(dst, "setup", None)
    if callable(setup_fn):
        setup_fn()

    copied = 0
    skipped = 0
    writes_copied = 0
    try:
        for item in src.list(None, limit=1_000_000):
            checkpoint = item.checkpoint or {}
            channel_versions = checkpoint.get("channel_versions", {})
            if not isinstance(channel_versions, dict):
                channel_versions = {}
            try:
                new_cfg = dst.put(item.config, checkpoint, item.metadata or {}, channel_versions)
                copied += 1
            except Exception:
                skipped += 1
                continue

            pending = list(item.pending_writes or [])
            if not pending:
                continue
            grouped: dict[tuple[str, str], list[tuple[str, object]]] = defaultdict(list)
            for w in pending:
                # Expected shape in current langgraph sqlite saver:
                # (task_id, channel, value, task_path)
                if not isinstance(w, tuple):
                    continue
                if len(w) >= 4:
                    task_id = str(w[0])
                    channel = str(w[1])
                    value = w[2]
                    task_path = str(w[3] or "")
                elif len(w) == 3:
                    task_id = str(w[0])
                    channel = str(w[1])
                    value = w[2]
                    task_path = ""
                else:
                    continue
                grouped[(task_id, task_path)].append((channel, value))

            for (task_id, task_path), writes in grouped.items():
                try:
                    dst.put_writes(new_cfg, writes, task_id=task_id, task_path=task_path)
                    writes_copied += len(writes)
                except Exception:
                    continue
    finally:
        dst_cm.__exit__(None, None, None)
        src_cm.__exit__(None, None, None)

    print(
        f"Checkpoint migration complete. copied={copied} skipped={skipped} writes_copied={writes_copied}"
    )


if __name__ == "__main__":
    main()
