"""
Per-user Monday OAuth credential storage with encryption-at-rest.
"""
from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from rag_agent.config import (
    MONDAY_CLIENT_ID,
    MONDAY_CREDENTIALS_FILE,
    MONDAY_ENCRYPTION_KEY,
    MONDAY_OAUTH_STATE_TTL_SECONDS,
)

_oauth_states: dict[str, tuple[str, float]] = {}


def _derive_fernet(secret: str) -> Fernet:
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    key = base64.urlsafe_b64encode(digest)
    return Fernet(key)


def _fernet() -> Fernet | None:
    if not MONDAY_ENCRYPTION_KEY:
        return None
    return _derive_fernet(MONDAY_ENCRYPTION_KEY)


def _load_raw(path: Path = MONDAY_CREDENTIALS_FILE) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_raw(data: dict[str, Any], path: Path = MONDAY_CREDENTIALS_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_store_ready() -> None:
    # If Monday integration is configured, encryption key is mandatory.
    if MONDAY_CLIENT_ID and not MONDAY_ENCRYPTION_KEY:
        raise RuntimeError("MONDAY_ENCRYPTION_KEY must be set when Monday integration is enabled.")


def _encrypt_payload(payload: dict[str, Any]) -> str:
    _ensure_store_ready()
    f = _fernet()
    serialized = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if f is None:
        # Local-dev fallback when Monday is disabled.
        return base64.urlsafe_b64encode(serialized).decode("ascii")
    return f.encrypt(serialized).decode("utf-8")


def _decrypt_payload(token: str) -> dict[str, Any] | None:
    _ensure_store_ready()
    if not token:
        return None
    f = _fernet()
    try:
        if f is None:
            raw = base64.urlsafe_b64decode(token.encode("ascii"))
        else:
            raw = f.decrypt(token.encode("utf-8"))
        data = json.loads(raw.decode("utf-8"))
        return data if isinstance(data, dict) else None
    except (InvalidToken, ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def save_user_credentials(username: str, payload: dict[str, Any]) -> None:
    if not username:
        return
    raw = _load_raw()
    encrypted = _encrypt_payload(payload)
    raw[username] = {"encrypted_payload": encrypted, "updated_at": int(time.time())}
    _save_raw(raw)


def get_user_credentials(username: str) -> dict[str, Any] | None:
    if not username:
        return None
    raw = _load_raw()
    entry = raw.get(username)
    if not isinstance(entry, dict):
        return None
    token = str(entry.get("encrypted_payload") or "")
    return _decrypt_payload(token)


def delete_user_credentials(username: str) -> None:
    if not username:
        return
    raw = _load_raw()
    if username in raw:
        raw.pop(username, None)
        _save_raw(raw)


def create_oauth_state(username: str) -> str:
    state = secrets.token_urlsafe(24)
    _oauth_states[state] = (username, time.time() + MONDAY_OAUTH_STATE_TTL_SECONDS)
    return state


def consume_oauth_state(state: str) -> str | None:
    if not state:
        return None
    now = time.time()
    # Purge stale states lazily.
    for key, (_, expiry) in list(_oauth_states.items()):
        if now > expiry:
            _oauth_states.pop(key, None)
    data = _oauth_states.pop(state, None)
    if not data:
        return None
    username, expiry = data
    if now > expiry:
        return None
    return username
