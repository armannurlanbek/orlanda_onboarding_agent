"""
Website auth: PostgreSQL only (users + auth_sessions).

- Users: Argon2 password hashes; roles admin/user; identities per RAG_AGENT_ADMIN_USERNAMES
  and RAG_ALLOWED_EMAIL_DOMAIN (@orlanda.info by default).
- Sessions: opaque bearer tokens stored hashed in auth_sessions (survive restarts).

Legacy SHA-256 password hashes from imports are still verified once and upgraded
to Argon2 on successful login.
"""
from __future__ import annotations

import hashlib
import secrets
import string
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
    """Return Russian error message for invalid new password, or None if ok."""
    if len(password) > RAG_MAX_PASSWORD_LENGTH:
        return f"Пароль не длиннее {RAG_MAX_PASSWORD_LENGTH} символов"
    if len(password) < RAG_MIN_PASSWORD_LENGTH:
        return f"Пароль: минимум {RAG_MIN_PASSWORD_LENGTH} символов"
    if not any(ch.isalpha() for ch in password):
        return "Пароль должен содержать хотя бы одну букву"
    if not any(ch.isdigit() for ch in password):
        return "Пароль должен содержать хотя бы одну цифру (0–9)"
    return None


def _random_temp_password(length: int = 16) -> str:
    """Generate a random temporary password compliant with current policy."""
    length = max(RAG_MIN_PASSWORD_LENGTH, min(RAG_MAX_PASSWORD_LENGTH, int(length or 16)))
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*_-+=?"
    while True:
        pwd = "".join(secrets.choice(alphabet) for _ in range(length))
        if _password_policy_error(pwd) is None:
            return pwd


def get_user_role(username: str) -> str:
    """Return 'admin' or 'user'. Admin if in ADMIN_USERNAMES env or stored role is admin."""
    if not username:
        return "user"
    canonical = _canonical_username(username)
    if canonical in ADMIN_USERNAMES:
        return "admin"
    if not DATABASE_URL:
        return "user"
    from rag_agent.db.models import User
    from rag_agent.db.session import get_session_factory

    session = get_session_factory()()
    try:
        row = session.scalar(select(User).where(func.lower(User.username).in_(_identity_candidates(username))))
        if row and row.role == "admin":
            return "admin"
    finally:
        session.close()
    return "user"


def _username_rule_error_message() -> str:
    return (
        "Разрешены короткие логины (буквы/цифры/_/-) "
        f"или адрес электронной почты вида имя@{RAG_ALLOWED_EMAIL_DOMAIN}"
    )


def _register_username_rule_error_message() -> str:
    return f"Регистрация возможна только с корпоративной почтой вида имя@{RAG_ALLOWED_EMAIL_DOMAIN}"


def _valid_short_username(username: str) -> bool:
    """Short login: no '@', 2–64 chars, letters/digits/_/-."""
    if "@" in username or len(username) < 2 or len(username) > 64:
        return False
    for ch in username:
        if ch in "_-":
            continue
        if ch.isalnum():
            continue
        return False
    return True


def _valid_company_email_username(username: str) -> bool:
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
    if "@" in username:
        return _valid_company_email_username(username)
    return _valid_short_username(username)


def _localpart_if_company_email(identity: str) -> str | None:
    text = (identity or "").strip().lower()
    if not _valid_company_email_username(text):
        return None
    local, _, _ = text.partition("@")
    return local or None


def _canonical_username(identity: str) -> str:
    text = (identity or "").strip().lower()
    local = _localpart_if_company_email(text)
    return local or text


def _identity_candidates(identity: str) -> list[str]:
    text = (identity or "").strip().lower()
    if not text:
        return []
    local = _localpart_if_company_email(text)
    if local:
        return [text, local]
    return [text]


def get_user_auth_flags(username: str) -> dict[str, bool]:
    """Return auth-related flags for UI and route guards."""
    if not DATABASE_URL or not username:
        return {"must_change_password": False}
    from rag_agent.db.models import User
    from rag_agent.db.session import get_session_factory

    db = get_session_factory()()
    try:
        user = db.scalar(select(User).where(func.lower(User.username).in_(_identity_candidates(username))))
        return {"must_change_password": bool(getattr(user, "must_change_password", False))} if user else {"must_change_password": False}
    finally:
        db.close()


def is_password_change_required(username: str) -> bool:
    return bool(get_user_auth_flags(username).get("must_change_password"))


def register(username: str, password: str) -> tuple[bool, str]:
    """Register a new user. Returns (success, token_or_error_message)."""
    if not DATABASE_URL:
        return False, "Сервер не настроен: задайте DATABASE_URL (PostgreSQL)."
    username = (username or "").strip()
    password = (password or "").strip()
    if not _valid_company_email_username(username):
        return False, _register_username_rule_error_message()
    policy_err = _password_policy_error(password)
    if policy_err:
        return False, policy_err
    return _register_db(_canonical_username(username), password)


def _register_db(username: str, password: str) -> tuple[bool, str]:
    from rag_agent.db.models import User
    from rag_agent.db.session import get_session_factory

    role = "admin" if username.lower() in ADMIN_USERNAMES else "user"
    ph = _hash_argon2(password)
    now = datetime.now(timezone.utc)
    session = get_session_factory()()
    try:
        exists = session.scalar(select(User.id).where(func.lower(User.username).in_(_identity_candidates(username))))
        if exists:
            return False, "Такой пользователь уже есть"
        u = User(
            username=username,
            password_hash=ph,
            role=role,
            is_active=True,
            must_change_password=False,
            password_changed_at=now,
            temp_password_issued_at=None,
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


def provision_user_with_temp_password(
    *,
    created_by_username: str,
    username: str,
    role: str = "user",
) -> tuple[bool, dict | str]:
    """Admin helper: create employee account with temporary password and forced first-login rotation."""
    if not DATABASE_URL:
        return False, "Сервер не настроен: задайте DATABASE_URL (PostgreSQL)."
    creator = (created_by_username or "").strip()
    username = _canonical_username((username or "").strip())
    if get_user_role(creator) != "admin":
        return False, "Доступ только для администратора"
    if not _username_valid(username):
        return False, _username_rule_error_message()

    role_norm = str(role or "user").strip().lower()
    if role_norm not in {"admin", "user"}:
        return False, "role должен быть 'admin' или 'user'"

    from rag_agent.db.models import User
    from rag_agent.db.session import get_session_factory

    temp_password = _random_temp_password()
    now = datetime.now(timezone.utc)
    db = get_session_factory()()
    try:
        exists = db.scalar(select(User.id).where(func.lower(User.username).in_(_identity_candidates(username))))
        if exists:
            return False, "Такой пользователь уже есть"
        u = User(
            username=username,
            password_hash=_hash_argon2(temp_password),
            role=role_norm,
            is_active=True,
            must_change_password=True,
            password_changed_at=None,
            temp_password_issued_at=now,
        )
        db.add(u)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    return True, {
        "username": username,
        "role": role_norm,
        "must_change_password": True,
        "temporary_password": temp_password,
    }


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
        candidates = _identity_candidates(username)
        users = session.execute(
            select(User).where(func.lower(User.username).in_(candidates))
        ).scalars().all()
        user = None
        for candidate in candidates:
            user = next((u for u in users if (u.username or "").lower() == candidate), None)
            if user:
                break
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


def _invalidate_all_user_sessions(db, user_id: uuid.UUID) -> None:
    from rag_agent.db.models import AuthSession

    db.execute(delete(AuthSession).where(AuthSession.user_id == user_id))


def change_password(
    *,
    username: str,
    new_password: str,
    repeat_password: str,
    current_password: str | None = None,
) -> tuple[bool, str]:
    """
    Change password for current user.
    - If must_change_password=true, current_password is optional.
    - Otherwise current_password is required and validated.
    Returns (ok, new_token_or_error).
    """
    if not DATABASE_URL:
        return False, "Сервер не настроен: задайте DATABASE_URL (PostgreSQL)."
    username = (username or "").strip()
    new_password = (new_password or "").strip()
    repeat_password = (repeat_password or "").strip()
    current_password = (current_password or "").strip() or None

    if not username:
        return False, "Не удалось определить пользователя"
    if new_password != repeat_password:
        return False, "Новый пароль и подтверждение не совпадают"
    policy_err = _password_policy_error(new_password)
    if policy_err:
        return False, policy_err

    from rag_agent.db.models import User
    from rag_agent.db.session import get_session_factory

    db = get_session_factory()()
    try:
        user = db.scalar(select(User).where(func.lower(User.username) == username.lower()))
        if not user or not user.is_active:
            db.rollback()
            return False, "Пользователь не найден или отключен"

        if user.must_change_password:
            # First-login flow: user already authenticated by token, current password can be omitted.
            pass
        else:
            if not current_password:
                db.rollback()
                return False, "Введите текущий пароль"
            ok, _ = _verify_and_maybe_upgrade_hash(user.password_hash, current_password)
            if not ok:
                db.rollback()
                return False, "Текущий пароль неверный"

        same_as_old, _ = _verify_and_maybe_upgrade_hash(user.password_hash, new_password)
        if same_as_old:
            db.rollback()
            return False, "Новый пароль должен отличаться от текущего"

        user.password_hash = _hash_argon2(new_password)
        user.must_change_password = False
        user.password_changed_at = datetime.now(timezone.utc)
        user.temp_password_issued_at = None

        _invalidate_all_user_sessions(db, user.id)
        db.flush()
        db.commit()

        new_token = _create_token(user.username, user_id=user.id)
        return True, new_token
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


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
