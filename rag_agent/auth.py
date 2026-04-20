"""
Website auth: PostgreSQL only (users + auth_sessions).

- Users: Argon2 password hashes; roles admin/user; identities per RAG_AGENT_ADMIN_USERNAMES
  and RAG_ALLOWED_EMAIL_DOMAIN (@orlanda.info by default).
- Sessions: opaque bearer tokens stored hashed in auth_sessions (survive restarts).

Legacy SHA-256 password hashes from imports are still verified once and upgraded to Argon2.

users.json is no longer read or written — use `python -m rag_agent.import_json_users` once to migrate.
"""
from __future__ import annotations

import hashlib
import secrets
import time
import uuid
from datetime import datetime, timezone

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from sqlalchemy import delete, func, select

from rag_agent.config import (
    ADMIN_USERNAMES,
    DATABASE_URL,
    RAG_ALLOWED_EMAIL_DOMAIN,
    RAG_MAX_PASSWORD_LENGTH,
    RAG_MIN_PASSWORD_LENGTH,
    RAG_SESSION_EXPIRY_DAYS,
    RAG_USERNAME_MAX_LEN,
    SECRET_KEY,
)

_hasher = PasswordHasher()


def _hash_legacy(password: str) -> str:
    return hashlib.sha256((SECRET_KEY + password).encode()).hexdigest()


def _hash_argon2(password: str) -> str:
    return _hasher.hash(password)


def _hash_session_token(raw_token: str) -> str:
    """Store only a derived hash of the bearer token (not reversible to token without brute force)."""
    return hashlib.sha256((SECRET_KEY + ":" + raw_token).encode("utf-8")).hexdigest()


def _verify_and_maybe_upgrade_hash(stored: str, password: str) -> tuple[bool, str | None]:
    """
    Verify password against Argon2 or legacy SHA-256 hash.
    Returns (ok, new_hash) where new_hash is set when the stored hash should be replaced (upgrade/rehash).
    """
    stored = stored or ""
    if stored.startswith("$argon2"):
        try:
            _hasher.verify(stored, password)
            if _hasher.check_needs_rehash(stored):
                return True, _hasher.hash(password)
            return True, None
        except VerifyMismatchError:
            return False, None
        except InvalidHashError:
            return False, None
    if secrets.compare_digest(stored, _hash_legacy(password)):
        return True, _hasher.hash(password)
    return False, None


def _password_policy_error(password: str) -> str | None:
    """Return Russian error message for invalid new password, or None if ok (registration only)."""
    if len(password) > RAG_MAX_PASSWORD_LENGTH:
        return f"Пароль не длиннее {RAG_MAX_PASSWORD_LENGTH} символов"
    if len(password) < RAG_MIN_PASSWORD_LENGTH:
        return f"Пароль: минимум {RAG_MIN_PASSWORD_LENGTH} символов"
    if not any(ch.isalpha() for ch in password):
        return "Пароль должен содержать хотя бы одну букву"
    if not any(ch.isdigit() for ch in password):
        return "Пароль должен содержать хотя бы одну цифру (0–9)"
    return None


def get_user_role(username: str) -> str:
    """Return 'admin' or 'user'. Admin if in ADMIN_USERNAMES env or stored role is admin."""
    if not username:
        return "user"
    if username.lower() in ADMIN_USERNAMES:
        return "admin"
    if not DATABASE_URL:
        return "user"
    from rag_agent.db.models import User
    from rag_agent.db.session import get_session_factory

    session = get_session_factory()()
    try:
        row = session.scalar(select(User).where(func.lower(User.username) == username.lower()))
        if row and row.role == "admin":
            return "admin"
    finally:
        session.close()
    return "user"


def _username_rule_error_message() -> str:
    return (
        "Разрешены только логины из списка администратора (переменная RAG_AGENT_ADMIN_USERNAMES) "
        f"или адрес электронной почты вида имя@{RAG_ALLOWED_EMAIL_DOMAIN}"
    )


def _valid_admin_short_username(username: str) -> bool:
    """Admin-only short login: no '@', 2–64 chars, letters/digits/_/-."""
    if "@" in username or len(username) < 2 or len(username) > 64:
        return False
    for ch in username:
        if ch in "_-":
            continue
        if ch.isalnum():
            continue
        return False
    return True


def _valid_orlanda_email_username(username: str) -> bool:
    """Exactly one '@', domain is RAG_ALLOWED_EMAIL_DOMAIN, common email local-part chars."""
    if username.count("@") != 1:
        return False
    local, _, domain = username.partition("@")
    if not local:
        return False
    if domain.lower() != RAG_ALLOWED_EMAIL_DOMAIN.lower():
        return False
    if len(username) > RAG_USERNAME_MAX_LEN:
        return False
    for ch in local:
        if ch in "._%+-" or ch.isalnum():
            continue
        return False
    return True


def _username_valid(username: str) -> bool:
    ul = username.lower()
    if ul in ADMIN_USERNAMES and ADMIN_USERNAMES:
        if "@" not in username:
            return _valid_admin_short_username(username)
        return _valid_orlanda_email_username(username)
    if "@" in username:
        return _valid_orlanda_email_username(username)
    return False


def register(username: str, password: str) -> tuple[bool, str]:
    """Register a new user. Returns (success, token_or_error_message)."""
    if not DATABASE_URL:
        return False, "Сервер не настроен: задайте DATABASE_URL (PostgreSQL)."
    username = (username or "").strip()
    password = (password or "").strip()
    if not _username_valid(username):
        return False, _username_rule_error_message()
    policy_err = _password_policy_error(password)
    if policy_err:
        return False, policy_err
    return _register_db(username, password)


def _register_db(username: str, password: str) -> tuple[bool, str]:
    from rag_agent.db.models import User
    from rag_agent.db.session import get_session_factory

    role = "admin" if username.lower() in ADMIN_USERNAMES else "user"
    ph = _hash_argon2(password)
    session = get_session_factory()()
    try:
        exists = session.scalar(select(User.id).where(func.lower(User.username) == username.lower()))
        if exists:
            return False, "Такой пользователь уже есть"
        u = User(
            username=username,
            password_hash=ph,
            role=role,
            is_active=True,
        )
        session.add(u)
        session.flush()
        uid = u.id
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
    token = _create_token(username, user_id=uid)
    return True, token


def login(username: str, password: str) -> tuple[bool, str]:
    """Check credentials, return (success, token_or_error_message)."""
    if not DATABASE_URL:
        return False, "Сервер не настроен: задайте DATABASE_URL (PostgreSQL)."
    username = (username or "").strip()
    password = (password or "").strip()
    if not username or not password:
        return False, "Введите логин и пароль"
    if not _username_valid(username):
        return False, _username_rule_error_message()
    return _login_db(username, password)


def _login_db(username: str, password: str) -> tuple[bool, str]:
    from rag_agent.db.models import User
    from rag_agent.db.session import get_session_factory

    session = get_session_factory()()
    try:
        user = session.scalar(select(User).where(func.lower(User.username) == username.lower()))
        if not user or not user.is_active:
            session.rollback()
            return False, "Неверный логин или пароль"
        ok, new_hash = _verify_and_maybe_upgrade_hash(user.password_hash, password)
        if not ok:
            session.rollback()
            return False, "Неверный логин или пароль"
        if new_hash:
            user.password_hash = new_hash
        session.commit()
        uid = user.id
        canonical = user.username
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
    token = _create_token(canonical, user_id=uid)
    return True, token


def _create_token(username: str, *, user_id: uuid.UUID) -> str:
    """Issue bearer token; persist to auth_sessions."""
    raw = secrets.token_urlsafe(32)
    expiry_ts = time.time() + RAG_SESSION_EXPIRY_DAYS * 86400
    expires_at = datetime.fromtimestamp(expiry_ts, tz=timezone.utc)

    from rag_agent.db.models import AuthSession
    from rag_agent.db.session import get_session_factory

    db = get_session_factory()()
    try:
        db.add(
            AuthSession(
                token_hash=_hash_session_token(raw),
                user_id=user_id,
                expires_at=expires_at,
            )
        )
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
    return raw


def _resolve_token_db(token: str) -> str | None:
    from rag_agent.db.models import AuthSession, User
    from rag_agent.db.session import get_session_factory

    th = _hash_session_token(token)
    db = get_session_factory()()
    try:
        now = datetime.now(timezone.utc)
        db.execute(delete(AuthSession).where(AuthSession.expires_at < now))
        username = db.scalar(
            select(User.username)
            .join(AuthSession, AuthSession.user_id == User.id)
            .where(AuthSession.token_hash == th, AuthSession.expires_at > now),
        )
        db.commit()
        return username
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def resolve_token(token: str) -> str | None:
    """Return username if token is valid and not expired, else None."""
    if not token or not DATABASE_URL:
        return None
    return _resolve_token_db(token)


def invalidate_token(token: str) -> None:
    """Remove session so this bearer token stops working (logout)."""
    if not token or not DATABASE_URL:
        return
    from rag_agent.db.models import AuthSession
    from rag_agent.db.session import get_session_factory

    th = _hash_session_token(token)
    db = get_session_factory()()
    try:
        db.execute(delete(AuthSession).where(AuthSession.token_hash == th))
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
