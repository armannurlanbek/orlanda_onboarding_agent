"""
One-time migration: copy legacy rag_agent/data/users.json rows into PostgreSQL.

Runtime auth no longer reads this file — import once, verify accounts, then delete or archive users.json.

Run after DATABASE_URL is set and Alembic migrations are applied.
Skips usernames that already exist (case-insensitive).

Usage (from project root):

    python -m rag_agent.import_json_users
"""
from __future__ import annotations

import json
import sys

from sqlalchemy import func, select

from rag_agent.config import ADMIN_USERNAMES, DATABASE_URL, USERS_FILE
from rag_agent.db.models import User
from rag_agent.db.session import get_session_factory


def main() -> int:
    if not DATABASE_URL:
        print("DATABASE_URL is not set. Configure PostgreSQL first.", file=sys.stderr)
        return 1
    if not USERS_FILE.is_file():
        print(f"No file at {USERS_FILE}; nothing to import.", file=sys.stderr)
        return 0

    try:
        raw = json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"Could not read users file: {e}", file=sys.stderr)
        return 1

    if not isinstance(raw, dict) or not raw:
        print("users.json is empty or not an object.")
        return 0

    session = get_session_factory()()
    added = 0
    skipped = 0
    try:
        for username, payload in raw.items():
            if not isinstance(username, str) or not username.strip():
                continue
            un = username.strip()
            exists = session.scalar(select(User.id).where(func.lower(User.username) == un.lower()))
            if exists:
                skipped += 1
                continue
            if isinstance(payload, dict):
                password_hash = str(payload.get("password_hash") or payload.get("password") or "")
                role = str(payload.get("role") or "user")
            else:
                password_hash = str(payload)
                role = "admin" if un.lower() in ADMIN_USERNAMES else "user"
            if not password_hash:
                print(f"Skip {un!r}: no password hash", file=sys.stderr)
                continue
            if role not in ("admin", "user"):
                role = "user"
            session.add(
                User(
                    username=un,
                    password_hash=password_hash,
                    role=role,
                    is_active=True,
                )
            )
            added += 1
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    print(f"Imported {added} user(s); skipped {skipped} already present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
