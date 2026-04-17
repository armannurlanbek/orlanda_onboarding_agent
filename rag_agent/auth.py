"""
Simple auth for local website: username/password, token-based sessions.
Users stored in data/users.json; tokens in memory (re-login after server restart).
thread_id = username, so conversation history is per user when CHECKPOINT_DB is set.
"""
import hashlib
import json
import secrets
import re
from pathlib import Path

from rag_agent.config import SECRET_KEY, USERS_FILE, ADMIN_USERNAMES

# token -> (username, expiry_ts). In-memory: sessions lost on restart.
_sessions: dict[str, tuple[str, float]] = {}
TOKEN_EXPIRY_DAYS = 7


def _hash_password(password: str) -> str:
    return hashlib.sha256((SECRET_KEY + password).encode()).hexdigest()


def _load_users() -> dict:
    """username -> {password_hash, role} or legacy: username -> hashed_password (str)."""
    if not USERS_FILE.is_file():
        return {}
    try:
        raw = json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out = {}
    needs_save = False
    for k, v in raw.items():
        if isinstance(v, dict):
            out[k] = {"password_hash": v.get("password_hash", v.get("password", "")), "role": v.get("role", "user")}
        else:
            out[k] = {"password_hash": str(v), "role": "admin" if k.lower() in ADMIN_USERNAMES else "user"}
            needs_save = True
    if needs_save:
        _save_users(out)
    return out


def _save_users(users: dict) -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(json.dumps(users, indent=2), encoding="utf-8")


def _get_password_hash(users: dict, username: str) -> str | None:
    u = users.get(username)
    if not u:
        return None
    if isinstance(u, dict):
        return u.get("password_hash") or u.get("password")
    return str(u)


def get_user_role(username: str) -> str:
    """Return 'admin' or 'user'. Admin if in ADMIN_USERNAMES env or stored role is admin."""
    if not username:
        return "user"
    if username.lower() in ADMIN_USERNAMES:
        return "admin"
    users = _load_users()
    u = users.get(username)
    if isinstance(u, dict) and u.get("role") == "admin":
        return "admin"
    return "user"


def _username_valid(username: str) -> bool:
    return bool(username and re.match(r"^[a-zA-Z0-9_-]{2,64}$", username))


def register(username: str, password: str) -> tuple[bool, str]:
    """Register a new user. Returns (success, token_or_error_message)."""
    if not _username_valid(username):
        return False, "Логин: 2–64 символа, только буквы, цифры, _ и -"
    if not password or len(password) < 4:
        return False, "Пароль: минимум 4 символа"
    users = _load_users()
    if username.lower() in {u.lower() for u in users}:
        return False, "Такой пользователь уже есть"
    role = "admin" if username.lower() in ADMIN_USERNAMES else "user"
    users[username] = {"password_hash": _hash_password(password), "role": role}
    _save_users(users)
    token = _create_token(username)
    return True, token


def login(username: str, password: str) -> tuple[bool, str]:
    """Check credentials, return (success, token_or_error_message)."""
    if not username or not password:
        return False, "Введите логин и пароль"
    users = _load_users()
    stored_hash = _get_password_hash(users, username)
    if not stored_hash or stored_hash != _hash_password(password):
        return False, "Неверный логин или пароль"
    token = _create_token(username)
    return True, token


def _create_token(username: str) -> str:
    import time
    token = secrets.token_urlsafe(32)
    expiry = time.time() + TOKEN_EXPIRY_DAYS * 86400
    _sessions[token] = (username, expiry)
    return token


def resolve_token(token: str) -> str | None:
    """Return username if token is valid and not expired, else None."""
    import time
    if not token:
        return None
    data = _sessions.get(token)
    if not data:
        return None
    username, expiry = data
    if time.time() > expiry:
        _sessions.pop(token, None)
        return None
    return username
