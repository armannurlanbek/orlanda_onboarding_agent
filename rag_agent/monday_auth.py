"""
Per-user monday OAuth + MCP tool integration helpers.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import secrets
import threading
import time
import contextvars
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib import error as urllib_error, parse, request

from langchain_core.tools import StructuredTool
from sqlalchemy import delete, func, select

from rag_agent.config import (
    MONDAY_ENCRYPTION_KEY,
    RAG_MONDAY_MCP_MAX_OPTIONAL_PARAMS,
    RAG_MONDAY_MCP_MAX_RETRIES,
    RAG_MONDAY_MCP_MAX_TOOLS,
    RAG_MONDAY_MCP_PERSISTENT_SESSION,
    RAG_MONDAY_MCP_RETRY_BACKOFF_SECONDS,
    RAG_MONDAY_MCP_SUPPRESS_TERMINATION_500_WARNINGS,
    RAG_MONDAY_MCP_TOOLS_CACHE_TTL_SECONDS,
    RAG_MONDAY_MCP_TIMEOUT_SECONDS,
    RAG_MONDAY_MCP_TOOL_ALLOWLIST,
    RAG_MONDAY_MCP_TRANSPORT,
    RAG_MONDAY_MCP_URL,
    RAG_MONDAY_OAUTH_AUTHORIZE_URL,
    RAG_MONDAY_OAUTH_CLIENT_ID,
    RAG_MONDAY_OAUTH_CLIENT_SECRET,
    RAG_MONDAY_OAUTH_REDIRECT_URI,
    RAG_MONDAY_OAUTH_SCOPES,
    RAG_MONDAY_OAUTH_STATE_TTL_SECONDS,
    RAG_MONDAY_OAUTH_TOKEN_URL,
    RAG_MONDAY_TOKEN_MAX_AGE_SECONDS,
    RAG_MONDAY_TOKEN_REFRESH_LEEWAY_SECONDS,
)
from rag_agent.db.models import MondayConnectionState, MondayUserConnection, User
from rag_agent.db.session import get_engine, get_session_factory

logger = logging.getLogger(__name__)


_monday_call_stats_ctx: contextvars.ContextVar[dict[str, int] | None] = contextvars.ContextVar(
    "monday_call_stats_ctx",
    default=None,
)


def begin_monday_call_stats() -> None:
    _monday_call_stats_ctx.set({"read_calls": 0, "write_calls": 0, "all_calls": 0})


def _bump_monday_call(tool_name: str) -> None:
    stats = _monday_call_stats_ctx.get()
    if stats is None:
        return
    stats["all_calls"] = int(stats.get("all_calls", 0)) + 1
    name_l = str(tool_name or "").lower()
    write_markers = {"create_", "update", "delete", "change_", "move_", "set_", "edit", "write", "add_"}
    if any(marker in name_l for marker in write_markers):
        stats["write_calls"] = int(stats.get("write_calls", 0)) + 1
    else:
        stats["read_calls"] = int(stats.get("read_calls", 0)) + 1


def get_monday_call_stats() -> dict[str, int]:
    stats = _monday_call_stats_ctx.get() or {}
    return {
        "all_calls": int(stats.get("all_calls", 0)),
        "read_calls": int(stats.get("read_calls", 0)),
        "write_calls": int(stats.get("write_calls", 0)),
    }


def _as_mcp_guard_result(tool_name: str, message: str, *, missing: list[str] | None = None) -> str:
    payload = {
        "ok": False,
        "source": "monday_mcp_guard",
        "tool": tool_name,
        "error_type": "validation",
        "message": message,
    }
    if missing:
        payload["missing"] = missing
    return json.dumps(payload, ensure_ascii=False)


def _format_tool_error_for_model(tool_name: str, err: BaseException) -> str:
    """Return a guard JSON the model can read and react to.

    Critically, this NEVER raises — every tool exception becomes a tool message
    so the agent loop can self-correct instead of crashing the chat. Pattern-matches
    common Monday GraphQL errors and appends a remediation hint to nudge the model
    toward the right next step.
    """
    txt = str(err or "")
    lower = txt.lower()
    hint = ""
    if "column not found" in lower or "missing_column" in lower:
        hint = (
            "Column id was wrong. Call `get_board_info` first to fetch the real "
            "column ids for this board, then retry with the correct id."
        )
    elif "request_max_complexity_exceeded" in lower or "complexityexception" in lower:
        hint = (
            "Monday query was too expensive. Retry with `limit<=25` and an explicit "
            "`columnIds` list of only the columns needed for the answer."
        )
    elif "userunauthorizedexception" in lower or "boards permission" in lower:
        hint = (
            "User has no permission for this resource. Ask the user to grant access "
            "or pick a different board."
        )
    elif "resourcenotfoundexception" in lower or "not found" in lower:
        hint = (
            "Resource does not exist or is not visible. Verify the id with `search` "
            "or `list_workspaces`/`get_board_info` before retrying."
        )
    elif "mcp error -32602" in lower or "input validation error" in lower:
        hint = (
            "Tool arguments failed Monday validation. Re-read the schema and retry "
            "with corrected fields (check required keys and value types)."
        )
    elif "timeout" in lower or "timed out" in lower:
        hint = "Monday API timed out. Retry with a smaller request or ask the user to wait and retry."
    msg = f"Monday tool `{tool_name}` failed: {txt}"
    if hint:
        msg += f" Recovery hint: {hint}"
    return _as_mcp_guard_result(tool_name, msg)


def _required_args_from_schema(args_schema) -> set[str]:
    if args_schema is None:
        return set()
    fields = getattr(args_schema, "model_fields", None)
    if isinstance(fields, dict):
        required: set[str] = set()
        for name, field in fields.items():
            if getattr(field, "is_required", None) and field.is_required():
                required.add(str(name))
        return required
    return set()


def _missing_required_args(args_schema, kwargs: dict[str, Any]) -> list[str]:
    required = _required_args_from_schema(args_schema)
    if not required:
        return []
    missing: list[str] = []
    for key in sorted(required):
        if key not in kwargs:
            missing.append(key)
            continue
        val = kwargs.get(key)
        if val is None:
            missing.append(key)
            continue
        if isinstance(val, str) and not val.strip():
            missing.append(key)
    return missing


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_tables() -> None:
    bind = get_engine()
    MondayConnectionState.__table__.create(bind=bind, checkfirst=True)
    MondayUserConnection.__table__.create(bind=bind, checkfirst=True)


def _fernet_or_none():
    raw = (MONDAY_ENCRYPTION_KEY or "").strip()
    if not raw:
        return None
    try:
        from cryptography.fernet import Fernet
    except ImportError:
        logger.warning("cryptography is not installed; monday token encryption unavailable")
        return None
    digest = hashlib.sha256(raw.encode("utf-8")).digest()
    key = base64.urlsafe_b64encode(digest)
    return Fernet(key)


def _encrypt_token(value: str) -> str:
    f = _fernet_or_none()
    if not f:
        raise RuntimeError("MONDAY_ENCRYPTION_KEY must be set and cryptography installed")
    return f.encrypt((value or "").encode("utf-8")).decode("utf-8")


def _decrypt_token(value: str | None) -> str:
    if not value:
        return ""
    f = _fernet_or_none()
    if not f:
        return ""
    try:
        return f.decrypt(value.encode("utf-8")).decode("utf-8")
    except Exception:
        return ""


def _code_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _resolve_user(db, username: str) -> User | None:
    u = (username or "").strip().lower()
    if not u:
        return None
    return db.scalar(select(User).where(func.lower(User.username) == u))


def _cleanup_expired_states(db) -> None:
    db.execute(delete(MondayConnectionState).where(MondayConnectionState.expires_at < _utcnow()))


@dataclass
class MondayStatus:
    connected: bool
    monday_user_id: str | None = None
    monday_account_id: str | None = None
    scope: str | None = None
    expires_at: str | None = None
    revoked: bool = False


def get_monday_status(username: str) -> MondayStatus:
    _ensure_tables()
    db = get_session_factory()()
    try:
        user = _resolve_user(db, username)
        if not user:
            return MondayStatus(connected=False)
        conn = db.scalar(select(MondayUserConnection).where(MondayUserConnection.user_id == user.id))
        if not conn:
            return MondayStatus(connected=False)
        revoked = conn.revoked_at is not None
        token = _decrypt_token(conn.access_token_encrypted)
        # A stored access token (even if near/at expiry) is considered connected as
        # long as we hold a refresh token to mint a fresh one on the next chat turn.
        has_refresh = bool(_decrypt_token(conn.refresh_token_encrypted))
        connected = bool((token or has_refresh) and not revoked)
        return MondayStatus(
            connected=connected,
            monday_user_id=conn.monday_user_id,
            monday_account_id=conn.monday_account_id,
            scope=conn.scope,
            expires_at=conn.expires_at.isoformat() if conn.expires_at else None,
            revoked=revoked,
        )
    finally:
        db.close()


def start_monday_oauth(username: str, redirect_uri: str | None = None) -> dict[str, str]:
    _ensure_tables()
    if not RAG_MONDAY_OAUTH_CLIENT_ID:
        raise RuntimeError("RAG_MONDAY_OAUTH_CLIENT_ID is required")
    redirect = (redirect_uri or RAG_MONDAY_OAUTH_REDIRECT_URI or "").strip()
    if not redirect:
        raise RuntimeError("RAG_MONDAY_OAUTH_REDIRECT_URI is required")

    verifier = secrets.token_urlsafe(48)
    state = secrets.token_urlsafe(32)
    challenge = _code_challenge(verifier)
    expires_at = _utcnow() + timedelta(seconds=RAG_MONDAY_OAUTH_STATE_TTL_SECONDS)

    db = get_session_factory()()
    try:
        user = _resolve_user(db, username)
        if not user:
            raise RuntimeError("User not found")
        _cleanup_expired_states(db)
        db.add(
            MondayConnectionState(
                state=state,
                user_id=user.id,
                code_verifier=verifier,
                redirect_uri=redirect,
                expires_at=expires_at,
            )
        )
        db.commit()
    finally:
        db.close()

    params = {
        "client_id": RAG_MONDAY_OAUTH_CLIENT_ID,
        "redirect_uri": redirect,
        "response_type": "code",
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    scopes = (RAG_MONDAY_OAUTH_SCOPES or "").strip()
    if scopes:
        params["scope"] = scopes
    auth_url = f"{RAG_MONDAY_OAUTH_AUTHORIZE_URL}?{parse.urlencode(params)}"
    return {"authorization_url": auth_url, "state": state}


def _post_form(url: str, payload: dict[str, str]) -> dict[str, Any]:
    body = parse.urlencode(payload).encode("utf-8")
    req = request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    with request.urlopen(req, timeout=20) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        raise RuntimeError("Invalid OAuth token response")


def complete_monday_oauth_callback(
    *,
    state: str,
    code: str | None,
    error: str | None,
    error_description: str | None,
) -> dict[str, Any]:
    _ensure_tables()
    db = get_session_factory()()
    try:
        _cleanup_expired_states(db)
        row = db.scalar(select(MondayConnectionState).where(MondayConnectionState.state == (state or "").strip()))
        if not row or row.expires_at < _utcnow():
            raise RuntimeError("OAuth state is invalid or expired")
        user = db.scalar(select(User).where(User.id == row.user_id))
        if not user:
            raise RuntimeError("User for OAuth state was not found")
        db.delete(row)
        db.commit()

        if error:
            raise RuntimeError(error_description or error)
        if not code:
            raise RuntimeError("OAuth callback is missing code")

        if not RAG_MONDAY_OAUTH_CLIENT_ID:
            raise RuntimeError("RAG_MONDAY_OAUTH_CLIENT_ID is required")
        if not RAG_MONDAY_OAUTH_CLIENT_SECRET:
            raise RuntimeError("RAG_MONDAY_OAUTH_CLIENT_SECRET is required")
        token_response = _post_form(
            RAG_MONDAY_OAUTH_TOKEN_URL,
            {
                "grant_type": "authorization_code",
                "client_id": RAG_MONDAY_OAUTH_CLIENT_ID,
                "client_secret": RAG_MONDAY_OAUTH_CLIENT_SECRET,
                "code": code,
                "redirect_uri": row.redirect_uri or RAG_MONDAY_OAUTH_REDIRECT_URI,
                "code_verifier": row.code_verifier,
            },
        )

        access_token = str(token_response.get("access_token") or "").strip()
        if not access_token:
            raise RuntimeError("OAuth token exchange did not return access_token")
        refresh_token = str(token_response.get("refresh_token") or "").strip()
        token_type = str(token_response.get("token_type") or "Bearer").strip() or "Bearer"
        scope = str(token_response.get("scope") or "").strip() or None

        expires_at = None
        expires_in = token_response.get("expires_in")
        if isinstance(expires_in, (int, float)) and float(expires_in) > 0:
            expires_at = _utcnow() + timedelta(seconds=float(expires_in))

        conn = db.scalar(select(MondayUserConnection).where(MondayUserConnection.user_id == user.id))
        if conn is None:
            conn = MondayUserConnection(
                user_id=user.id,
                access_token_encrypted="",
                token_type="Bearer",
            )
            db.add(conn)
        conn.access_token_encrypted = _encrypt_token(access_token)
        conn.refresh_token_encrypted = _encrypt_token(refresh_token) if refresh_token else None
        conn.token_type = token_type
        conn.scope = scope
        conn.expires_at = expires_at
        conn.revoked_at = None
        conn.updated_at = _utcnow()
        db.commit()
        return {"ok": True, "username": user.username}
    finally:
        db.close()


def disconnect_monday(username: str) -> None:
    _ensure_tables()
    db = get_session_factory()()
    try:
        user = _resolve_user(db, username)
        if not user:
            return
        conn = db.scalar(select(MondayUserConnection).where(MondayUserConnection.user_id == user.id))
        if not conn:
            return
        conn.revoked_at = _utcnow()
        conn.updated_at = _utcnow()
        db.commit()
    finally:
        db.close()


# Seconds before `expires_at` at which we proactively refresh the access token,
# so an in-flight chat turn never starts with a token about to expire. Widened
# from 60s and made configurable so even a long multi-step turn refreshes first.
_TOKEN_REFRESH_LEEWAY_SECONDS = RAG_MONDAY_TOKEN_REFRESH_LEEWAY_SECONDS
# Fallback freshness window when Monday omits `expires_in` (no `expires_at`).
_TOKEN_MAX_AGE_SECONDS = RAG_MONDAY_TOKEN_MAX_AGE_SECONDS

# Per-user locks to avoid a refresh stampede when concurrent requests for the same
# user all observe an expired token at once. Keyed by lowercased username.
_refresh_locks_guard = threading.Lock()
_refresh_locks: dict[str, threading.Lock] = {}


def _refresh_lock_for(username: str) -> threading.Lock:
    key = (username or "").strip().lower()
    with _refresh_locks_guard:
        lock = _refresh_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _refresh_locks[key] = lock
        return lock


def _token_needs_refresh(conn: MondayUserConnection, token: str) -> bool:
    """True when the stored access token is missing, near expiry, or (when expiry
    is unknown) older than the max-age fallback so it still rotates."""
    if not token:
        return True
    if conn.expires_at is None:
        # Monday may omit `expires_in`; don't treat the token as eternal.
        updated = getattr(conn, "updated_at", None)
        if updated is None:
            return True
        return updated <= (_utcnow() - timedelta(seconds=_TOKEN_MAX_AGE_SECONDS))
    return conn.expires_at <= (_utcnow() + timedelta(seconds=_TOKEN_REFRESH_LEEWAY_SECONDS))


def _refresh_access_token_for_user(username: str) -> str:
    """Refresh the per-user monday access token via the OAuth refresh-token grant.

    Returns the fresh (decrypted) access token on success, or "" on any failure
    (no refresh token, network/HTTP error, error payload) so the caller reports
    the user as disconnected. Never logs token values — only counts/reasons.

    A per-user lock serializes concurrent refreshes; after acquiring the lock we
    re-read the row so a token refreshed by another waiter is reused instead of
    triggering a second refresh.
    """
    if not RAG_MONDAY_OAUTH_CLIENT_ID or not RAG_MONDAY_OAUTH_CLIENT_SECRET:
        logger.warning("Monday token refresh skipped: OAuth client id/secret not configured")
        return ""

    lock = _refresh_lock_for(username)
    with lock:
        db = get_session_factory()()
        try:
            user = _resolve_user(db, username)
            if not user:
                return ""
            conn = db.scalar(select(MondayUserConnection).where(MondayUserConnection.user_id == user.id))
            if not conn or conn.revoked_at is not None:
                return ""

            # Another waiter may have already refreshed while we blocked on the lock.
            current = _decrypt_token(conn.access_token_encrypted)
            if not _token_needs_refresh(conn, current):
                return current

            refresh_token = _decrypt_token(conn.refresh_token_encrypted)
            if not refresh_token:
                logger.warning("Monday token refresh unavailable: no stored refresh token for user")
                return ""

            try:
                token_response = _post_form(
                    RAG_MONDAY_OAUTH_TOKEN_URL,
                    {
                        "grant_type": "refresh_token",
                        "client_id": RAG_MONDAY_OAUTH_CLIENT_ID,
                        "client_secret": RAG_MONDAY_OAUTH_CLIENT_SECRET,
                        "refresh_token": refresh_token,
                    },
                )
            except urllib_error.HTTPError as exc:
                logger.warning("Monday token refresh failed: HTTP %s", getattr(exc, "code", "?"))
                return ""
            except (urllib_error.URLError, TimeoutError, OSError) as exc:
                logger.warning("Monday token refresh failed: network error (%s)", type(exc).__name__)
                return ""
            except Exception:
                logger.warning("Monday token refresh failed: unexpected error parsing response")
                return ""

            new_access = str(token_response.get("access_token") or "").strip()
            if not new_access:
                err = str(token_response.get("error") or "no access_token in response")
                logger.warning("Monday token refresh failed: %s", err)
                return ""

            # Monday may or may not rotate the refresh token; keep the old one if absent.
            new_refresh = str(token_response.get("refresh_token") or "").strip()
            scope = str(token_response.get("scope") or "").strip() or conn.scope
            token_type = str(token_response.get("token_type") or "").strip() or conn.token_type or "Bearer"

            new_expires_at = None
            expires_in = token_response.get("expires_in")
            if isinstance(expires_in, (int, float)) and float(expires_in) > 0:
                new_expires_at = _utcnow() + timedelta(seconds=float(expires_in))

            conn.access_token_encrypted = _encrypt_token(new_access)
            if new_refresh:
                conn.refresh_token_encrypted = _encrypt_token(new_refresh)
            conn.token_type = token_type
            conn.scope = scope
            conn.expires_at = new_expires_at
            conn.revoked_at = None
            conn.updated_at = _utcnow()
            db.commit()
            logger.info("Monday access token refreshed successfully for user")
            return new_access
        except Exception:
            logger.warning("Monday token refresh failed: unexpected error persisting new token")
            try:
                db.rollback()
            except Exception:
                pass
            return ""
        finally:
            db.close()


def _get_access_token_for_user(username: str) -> str:
    _ensure_tables()
    db = get_session_factory()()
    try:
        user = _resolve_user(db, username)
        if not user:
            return ""
        conn = db.scalar(select(MondayUserConnection).where(MondayUserConnection.user_id == user.id))
        if not conn or conn.revoked_at is not None:
            return ""
        token = _decrypt_token(conn.access_token_encrypted)
        needs_refresh = _token_needs_refresh(conn, token)
    finally:
        db.close()

    # Refresh outside the read session so the refresh path owns its own DB session
    # and per-user lock (avoids holding a session open across a network call).
    if needs_refresh:
        return _refresh_access_token_for_user(username)
    return token


_tools_cache: dict[str, tuple[float, list]] = {}
_TOOLS_CACHE_TTL_SECONDS = float(RAG_MONDAY_MCP_TOOLS_CACHE_TTL_SECONDS)
_mcp_logging_configured = False


def _configure_mcp_logging() -> None:
    """Suppress known non-fatal MCP session termination warnings."""
    global _mcp_logging_configured
    if _mcp_logging_configured:
        return
    _mcp_logging_configured = True
    if not RAG_MONDAY_MCP_SUPPRESS_TERMINATION_500_WARNINGS:
        return
    logging.getLogger("mcp.client.streamable_http").setLevel(logging.ERROR)


def _run_coro_on_background_loop(make_coro):
    """Run a coroutine to completion on a dedicated background thread + loop.

    Used to bridge sync→async when the caller is already inside a running event
    loop (where `asyncio.run` would raise). Accepts a zero-arg factory so the
    coroutine is created on the background loop's thread, which avoids
    "attached to a different loop" errors. Blocks the calling thread until done.
    """
    result_box: dict[str, Any] = {}
    error_box: dict[str, BaseException] = {}

    def _worker() -> None:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result_box["value"] = loop.run_until_complete(make_coro())
        except BaseException as exc:  # noqa: BLE001 - propagate to caller thread
            error_box["error"] = exc
        finally:
            try:
                loop.close()
            finally:
                asyncio.set_event_loop(None)

    thread = threading.Thread(target=_worker, name="monday-mcp-loop", daemon=True)
    thread.start()
    thread.join()
    if "error" in error_box:
        raise error_box["error"]
    return result_box.get("value")


def _run_async_sync(make_coro):
    """Execute an async coroutine factory from sync code, regardless of context.

    `make_coro` MUST be a zero-arg callable returning a fresh coroutine (not a
    coroutine object) so it can be (re)created on whichever loop ends up running
    it. When no event loop is running we use `asyncio.run`; when one IS running
    (e.g. a future async caller) we offload to a dedicated background loop instead
    of silently returning None and yielding an empty toolset.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(make_coro())
    # A loop is already running on this thread; run on an isolated background loop.
    return _run_coro_on_background_loop(make_coro)


_TRANSIENT_MCP_MARKERS = (
    "connection reset",
    "connection aborted",
    "connection closed",
    "server disconnected",
    "remote end closed",
    "broken pipe",
    "temporarily unavailable",
    "timed out",
    "timeout",
    "read timeout",
    "502",
    "503",
    "504",
    "bad gateway",
    "service unavailable",
    "gateway timeout",
    "internal server error",
    "session terminated",
    "terminate",  # the suppressed "termination 500" on MCP session teardown
)


def _is_transient_mcp_error(exc: BaseException) -> bool:
    """Heuristic: True for blips worth retrying (resets, 5xx, session teardown).

    Validation / not-found / permission errors are NOT transient — retrying them
    just wastes a turn, so those fall through to the model-readable guard message.
    """
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    if isinstance(exc, (urllib_error.URLError,)):
        return True
    txt = str(exc or "").lower()
    # Don't treat a 500 that is actually a GraphQL validation echo as transient.
    if "input validation error" in txt or "-32602" in txt:
        return False
    return any(marker in txt for marker in _TRANSIENT_MCP_MARKERS)


async def _aretry(call_once, tool_name: str):
    """Run an async no-arg call with a bounded retry on transient MCP failures.

    Re-raises the last exception so the existing per-tool guard formatting turns it
    into a model-readable message. Non-transient errors are raised immediately.
    """
    attempts = RAG_MONDAY_MCP_MAX_RETRIES + 1
    last_exc: BaseException | None = None
    for attempt in range(attempts):
        try:
            return await call_once()
        except Exception as exc:  # noqa: BLE001 - re-raised below for guard formatting
            last_exc = exc
            if attempt >= attempts - 1 or not _is_transient_mcp_error(exc):
                raise
            backoff = RAG_MONDAY_MCP_RETRY_BACKOFF_SECONDS * (attempt + 1)
            logger.warning(
                "Transient Monday MCP failure invoking `%s` (attempt %d/%d, %s); retrying in %.2fs",
                tool_name,
                attempt + 1,
                attempts,
                type(exc).__name__,
                backoff,
            )
            if backoff > 0:
                await asyncio.sleep(backoff)
    if last_exc is not None:
        raise last_exc


async def _ainvoke_with_retry(async_tool, kwargs: dict[str, Any], tool_name: str):
    """Bounded-retry wrapper around a LangChain tool's `.ainvoke`."""
    return await _aretry(lambda: async_tool.ainvoke(kwargs), tool_name)


async def _aretry_coroutine(coro_tool, tool_name: str, kwargs: dict[str, Any]):
    """Bounded-retry wrapper around a raw MCP tool coroutine callable."""
    return await _aretry(lambda: coro_tool(**kwargs), tool_name)


def _load_tools_with_retry(make_get_tools_coro):
    """Load MCP tools with a bounded retry on transient connection failures.

    `make_get_tools_coro` is a zero-arg coroutine factory (so it can be re-run on a
    fresh attempt / background loop). Returns the loaded tools list, or re-raises
    the last error after exhausting retries (caller treats that as "no tools").
    """
    attempts = RAG_MONDAY_MCP_MAX_RETRIES + 1
    last_exc: BaseException | None = None
    for attempt in range(attempts):
        try:
            return _run_async_sync(make_get_tools_coro)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= attempts - 1 or not _is_transient_mcp_error(exc):
                raise
            backoff = RAG_MONDAY_MCP_RETRY_BACKOFF_SECONDS * (attempt + 1)
            logger.warning(
                "Transient Monday MCP failure loading tools (attempt %d/%d, %s); retrying in %.2fs",
                attempt + 1,
                attempts,
                type(exc).__name__,
                backoff,
            )
            if backoff > 0:
                time.sleep(backoff)
    if last_exc is not None:
        raise last_exc
    return []


def _ensure_sync_callable_tools(tools: list) -> list:
    wrapped: list = []
    for tool in tools:
        name = str(getattr(tool, "name", "")).strip()
        description = str(getattr(tool, "description", "") or "")
        args_schema = getattr(tool, "args_schema", None)
        func = getattr(tool, "func", None)
        coroutine = getattr(tool, "coroutine", None)

        def _guard_and_format_missing(kwargs: dict[str, Any]) -> str | None:
            missing = _missing_required_args(args_schema, kwargs or {})
            if not missing:
                return None
            return _as_mcp_guard_result(
                name or "monday_tool",
                f"Missing required args for `{name or 'monday_tool'}`: {', '.join(missing)}",
                missing=missing,
            )

        if callable(func):
            def _safe_sync_func(**kwargs):
                missing_res = _guard_and_format_missing(kwargs)
                if missing_res:
                    return missing_res
                try:
                    _bump_monday_call(name or "monday_tool")
                    return func(**kwargs)
                except Exception as e:
                    return _format_tool_error_for_model(name or "monday_tool", e)

            async def _safe_existing_coroutine(**kwargs):
                missing_res = _guard_and_format_missing(kwargs)
                if missing_res:
                    return missing_res
                if not callable(coroutine):
                    return _safe_sync_func(**kwargs)
                try:
                    _bump_monday_call(name or "monday_tool")
                    return await _aretry_coroutine(coroutine, name or "monday_tool", kwargs)
                except Exception as e:
                    return _format_tool_error_for_model(name or "monday_tool", e)

            wrapped.append(
                StructuredTool.from_function(
                    name=name or "monday_tool",
                    description=description or f"MCP tool: {name or 'monday_tool'}",
                    args_schema=args_schema,
                    func=_safe_sync_func,
                    coroutine=_safe_existing_coroutine,
                )
            )
            continue

        if not callable(coroutine):
            wrapped.append(tool)
            continue

        def _make_sync_runner(async_tool, tool_name: str, schema):
            def _sync_runner(**kwargs):
                missing = _missing_required_args(schema, kwargs or {})
                if missing:
                    return _as_mcp_guard_result(
                        tool_name,
                        f"Missing required args for `{tool_name}`: {', '.join(missing)}",
                        missing=missing,
                    )
                try:
                    _bump_monday_call(tool_name)
                    # Robust sync→async bridge (works even under a running loop) with
                    # a bounded retry on transient MCP failures.
                    return _run_async_sync(
                        lambda: _ainvoke_with_retry(async_tool, dict(kwargs), tool_name)
                    )
                except Exception as e:
                    return _format_tool_error_for_model(tool_name, e)

            return _sync_runner

        async def _safe_coroutine_runner(coro_tool, tool_name: str, schema, **kwargs):
            missing = _missing_required_args(schema, kwargs or {})
            if missing:
                return _as_mcp_guard_result(
                    tool_name,
                    f"Missing required args for `{tool_name}`: {', '.join(missing)}",
                    missing=missing,
                )
            try:
                _bump_monday_call(tool_name)
                return await _aretry_coroutine(coro_tool, tool_name, kwargs)
            except Exception as e:
                return _format_tool_error_for_model(tool_name, e)

        def _make_safe_coroutine(coro_tool, tool_name: str, schema):
            async def _runner(**kwargs):
                return await _safe_coroutine_runner(coro_tool, tool_name, schema, **kwargs)

            return _runner

        wrapped.append(
            StructuredTool.from_function(
                name=name or "monday_tool",
                description=description or f"MCP tool: {name or 'monday_tool'}",
                args_schema=args_schema,
                func=_make_sync_runner(tool, name or "monday_tool", args_schema),
                coroutine=_make_safe_coroutine(coroutine, name or "monday_tool", args_schema) if callable(coroutine) else None,
            )
        )
    return wrapped


# Tools the agent must never lose to a cap. Reads/discovery come first because the
# assistant is read-heavy and EVERY column-filtered or write call must be preceded
# by get_board_info (to resolve board-specific column ids). The old ranking gave
# write tools a higher score than reads and omitted get_board_info/list_workspaces
# entirely, so a tight cap evicted exactly the discovery tools the agent needs.
# Ordered most load-bearing first, so even an aggressively low cap keeps the
# tools the agent literally cannot navigate without (entry point -> search ->
# board metadata -> item reads -> the GraphQL escape hatch).
_MONDAY_READ_TOOLS = (
    "get_user_context",
    "search",
    "get_board_info",
    "get_board_items_page",
    "all_monday_api",
    "list_workspaces",
    "workspace_info",
    "get_column_type_info",
    "get_graphql_schema",
    "get_type_details",
    "list_users_and_teams",
    "get_updates",
    "get_board_activity",
    "board_insights",
    "read_docs",
)
_MONDAY_CORE_WRITE_TOOLS = (
    "create_item",
    "change_item_column_values",
    "create_update",
    "create_group",
    "create_column",
    "create_board",
    "create_doc",
    "update_doc",
)
_MONDAY_ESSENTIAL_TOOLS = frozenset(_MONDAY_READ_TOOLS + _MONDAY_CORE_WRITE_TOOLS)

# UI-internal / agent-unsafe tools to drop even if the server advertises them.
# get_full_board_data is explicitly marked "INTERNAL USE ONLY - DO NOT CALL DIRECTLY".
_MONDAY_TOOL_DENYLIST = frozenset({"get_full_board_data"})


def _cap_monday_tools(tools: list, max_tools: int) -> list:
    """Cap tool count while GUARANTEEING the essential read/write tools survive.

    Ranking (lower kept first): essential reads -> essential core writes ->
    other reads (get_*/list_*/search/*info*) -> everything else. With the default
    cap (25) and Monday's ~30-40 tool surface, all essentials plus the common
    writes are retained and only rare/niche tools are dropped.
    """
    cap = max(1, int(max_tools or 1))
    if len(tools) <= cap:
        return tools

    read_index = {name: i for i, name in enumerate(_MONDAY_READ_TOOLS)}
    write_index = {name: i for i, name in enumerate(_MONDAY_CORE_WRITE_TOOLS)}
    write_markers = ("create_", "update", "delete", "change_", "move_", "set_", "edit", "write", "add_")

    def _rank(tool_obj):
        name = str(getattr(tool_obj, "name", "") or "").strip()
        if name in read_index:
            return (0, read_index[name], name)
        if name in write_index:
            return (1, write_index[name], name)
        lname = name.lower()
        is_write = any(marker in lname for marker in write_markers)
        is_read = (
            lname.startswith("get_")
            or lname.startswith("list_")
            or "search" in lname
            or "info" in lname
        )
        if is_read and not is_write:
            return (2, 0, name)
        return (3, 0, name)

    ranked = sorted(list(tools), key=_rank)
    kept = ranked[:cap]
    dropped = [str(getattr(t, "name", "") or "") for t in ranked[cap:]]
    logger.warning(
        "Capped monday MCP tools from %d to %d (cap=%d); dropped: %s",
        len(tools),
        len(kept),
        cap,
        ", ".join(d for d in dropped if d) or "(none)",
    )
    return kept


def _count_optional_props_in_schema(schema_obj: Any) -> int:
    """Count optional object properties recursively in JSON schema dict."""
    if not isinstance(schema_obj, dict):
        return 0

    total = 0
    schema_type = str(schema_obj.get("type") or "").strip().lower()
    if schema_type == "object" and isinstance(schema_obj.get("properties"), dict):
        props = schema_obj["properties"]
        required = schema_obj.get("required", [])
        req_set = {str(x) for x in required} if isinstance(required, list) else set()
        for prop_name, prop_schema in props.items():
            if str(prop_name) not in req_set:
                total += 1
            total += _count_optional_props_in_schema(prop_schema)

    for key in ("items", "additionalProperties", "not"):
        child = schema_obj.get(key)
        if isinstance(child, dict):
            total += _count_optional_props_in_schema(child)

    for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
        children = schema_obj.get(key)
        if isinstance(children, list):
            for child in children:
                total += _count_optional_props_in_schema(child)

    defs = schema_obj.get("$defs")
    if isinstance(defs, dict):
        for child in defs.values():
            total += _count_optional_props_in_schema(child)

    return total


def _optional_param_count(tool_obj) -> int:
    schema_cls = getattr(tool_obj, "args_schema", None)
    if schema_cls is None:
        return 0
    try:
        model_json_schema = getattr(schema_cls, "model_json_schema", None)
        if callable(model_json_schema):
            schema_dict = model_json_schema()
            return _count_optional_props_in_schema(schema_dict)
    except Exception:
        return 0
    return 0


def _cap_optional_params(tools: list, max_optional_params: int) -> list:
    """Trim NON-essential tools to keep optional-parameter complexity under budget.

    Essential tools (``_MONDAY_ESSENTIAL_TOOLS``) are ALWAYS kept regardless of the
    budget, so this can never re-introduce the original "agent lost its discovery
    tools" bug. The old version walked all tools, summed optional params, and broke
    out of the loop once the budget was hit — a single heavy tool
    (get_board_items_page, ~15 optional params) exhausted it and dropped every tool
    after, including search / get_board_info / all_monday_api. Only runs when
    explicitly enabled (budget > 0); disabled by default.
    """
    budget = max(1, int(max_optional_params or 1))
    if not tools:
        return []

    kept: list = []
    used = 0
    for tool_obj in tools:
        name = str(getattr(tool_obj, "name", "") or "").strip()
        if name in _MONDAY_ESSENTIAL_TOOLS:
            kept.append(tool_obj)  # never spend budget on essentials
            continue
        opt = _optional_param_count(tool_obj)
        if (used + opt) > budget:
            continue  # skip this one but keep scanning for smaller tools
        kept.append(tool_obj)
        used += opt

    if len(kept) < len(tools):
        logger.warning(
            "Capped monday tool optional params (budget=%d): kept %d/%d tools (essentials always retained)",
            budget,
            len(kept),
            len(tools),
        )
    return kept


def _build_monday_mcp_client(token: str):
    """Construct a MultiServerMCPClient for the per-user monday connection.

    Raises ImportError if langchain-mcp-adapters is not installed.
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    return MultiServerMCPClient(
        {
            "monday": {
                "transport": RAG_MONDAY_MCP_TRANSPORT,
                "url": RAG_MONDAY_MCP_URL,
                "headers": {"Authorization": f"Bearer {token}"},
                "timeout": RAG_MONDAY_MCP_TIMEOUT_SECONDS,
            }
        }
    )


def _prepare_monday_tools(raw_tools: list) -> list:
    """Wrap + filter + cap raw MCP tools identically for every load path.

    Shared by the per-call loader and the persistent-session loader so both apply
    the same guards, denylist, allowlist and essential-tool-preserving caps.
    """
    tools = _ensure_sync_callable_tools(raw_tools)
    # Drop UI-internal / agent-unsafe tools the server may advertise.
    tools = [t for t in tools if str(getattr(t, "name", "")).strip() not in _MONDAY_TOOL_DENYLIST]
    if RAG_MONDAY_MCP_TOOL_ALLOWLIST:
        tools = [t for t in tools if str(getattr(t, "name", "")).strip() in RAG_MONDAY_MCP_TOOL_ALLOWLIST]
    tools = _cap_monday_tools(tools, RAG_MONDAY_MCP_MAX_TOOLS)
    if RAG_MONDAY_MCP_MAX_OPTIONAL_PARAMS > 0:
        tools = _cap_optional_params(tools, RAG_MONDAY_MCP_MAX_OPTIONAL_PARAMS)
    return tools


def get_monday_mcp_tools_for_user(username: str) -> list:
    _configure_mcp_logging()
    token = _get_access_token_for_user(username)
    if not token:
        return []
    key_digest = hashlib.sha256(f"{username}:{token}".encode("utf-8")).hexdigest()
    cached = _tools_cache.get(key_digest)
    now = time.time()
    if cached and (now - cached[0]) < _TOOLS_CACHE_TTL_SECONDS:
        return list(cached[1])
    try:
        client = _build_monday_mcp_client(token)
    except ImportError:
        logger.warning("langchain-mcp-adapters is not installed; monday tools unavailable")
        return []
    try:
        # Factory (not a coroutine object) so it can be re-created per retry/loop.
        # NOTE: get_tools() binds each tool to a *connection* (session=None), so every
        # invocation opens its own short-lived HTTP session. The async streaming path
        # uses `monday_session_tools(...)` instead, which reuses ONE session per turn.
        raw_tools = _load_tools_with_retry(lambda: client.get_tools()) or []
    except Exception:
        # Transient retries exhausted (or a hard failure). Do NOT cache the empty
        # result — a later turn should re-attempt rather than appear toolless for
        # the whole cache TTL.
        logger.warning("Failed to load Monday MCP tools after retries; reporting no tools this turn")
        return []
    tools = _prepare_monday_tools(raw_tools)
    _tools_cache[key_digest] = (now, list(tools))
    return tools


@asynccontextmanager
async def monday_session_tools(username: str, *, enabled: bool = True):
    """Yield monday tools bound to ONE MCP session kept open for the whole turn.

    Reuses a single ``client.session(...)`` for every tool call in the turn rather
    than opening a fresh HTTP session per call (the default get_tools() behavior,
    which makes a multi-step turn open 15-30 sessions). Use on the async streaming
    path::

        async with monday_session_tools(username, enabled=...) as tools:
            agent = build_agent(extra_tools=tools, include_monday_tools=False, ...)
            async for ev in agent.astream_events(...):
                ...

    Safety: yields ``[]`` when disabled / not connected; on ANY failure establishing
    the persistent session it falls back to the per-call tools
    (``get_monday_mcp_tools_for_user``) so behavior never regresses below today's.
    The session is created and torn down on the caller's event loop (a ClientSession
    is bound to its loop and is not concurrency-safe) — do not offload individual
    tool calls to other loops/threads while it is open.
    """
    if not enabled:
        yield []
        return

    token = await asyncio.to_thread(_get_access_token_for_user, username)
    if not token:
        yield []
        return

    if not RAG_MONDAY_MCP_PERSISTENT_SESSION:
        # Kill-switch: use the per-call path (its own short-lived sessions).
        yield await asyncio.to_thread(get_monday_mcp_tools_for_user, username)
        return

    _configure_mcp_logging()
    session_cm = None
    tools = None
    try:
        from langchain_mcp_adapters.tools import load_mcp_tools

        client = _build_monday_mcp_client(token)
        session_cm = client.session("monday")
        session = await session_cm.__aenter__()
        raw_tools = await load_mcp_tools(session)
        tools = _prepare_monday_tools(raw_tools)
    except Exception:
        # Setup failed: tear down any partial session, fall back to per-call tools.
        if session_cm is not None:
            try:
                await session_cm.__aexit__(None, None, None)
            except Exception:
                pass
        logger.warning("Persistent Monday MCP session unavailable; falling back to per-call tools")
        try:
            fallback = await asyncio.to_thread(get_monday_mcp_tools_for_user, username)
        except Exception:
            fallback = []
        yield fallback
        return

    # Session established — keep it open for the whole turn. A consumer exception
    # propagates out of `yield` (NOT caught here) and the session is torn down in
    # `finally`, so this can never double-yield.
    try:
        yield tools
    finally:
        try:
            await session_cm.__aexit__(None, None, None)
        except Exception:
            pass


_MONDAY_KEYWORDS = {
    "monday",
    "monday.com",
    "board",
    "boards",
    "todo",
    "to-do",
    "item",
    "items",
    "task",
    "tasks",
    "workspace",
    "pulse",
    "group",
    "status",
    "assignee",
    "assigned",
    "sprint",
    "crm",
    # Russian vocabulary
    "понедельник",
    "доска",
    "доски",
    "задача",
    "задачи",
    "таск",
    "таски",
    "статус",
    "воркспейс",
    "рабочее пространство",
    "назначено",
    "назначены",
    "assigned to me",
    "назначено мне",
}
_MONDAY_ACTIONS = {
    "create",
    "update",
    "delete",
    "list",
    "find",
    "show",
    "move",
    "assign",
    "summarize",
    "open",
    "check",
    # Russian verbs
    "создай",
    "создать",
    "обнови",
    "обновить",
    "удали",
    "удалить",
    "покажи",
    "посмотри",
    "найди",
    "список",
    "получи",
    "какие",
    "какая",
    "какой",
    "назначь",
    "назначить",
    "перемести",
    "перенеси",
    "суммируй",
    "сделай",
}


def detect_monday_intent(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return False
    words = {w.strip(".,!?():;[]{}\"'") for w in normalized.split()}
    has_keyword = any(k in normalized for k in _MONDAY_KEYWORDS) or bool(words & _MONDAY_KEYWORDS)
    has_action = bool(words & _MONDAY_ACTIONS)
    # Accept common "my tasks / board list" intents even if "monday" is omitted.
    if any(p in normalized for p in {"мои задачи", "мои таски", "my tasks", "какие задачи", "какие таски"}):
        return True
    return has_keyword and (has_action or "monday" in normalized or "доска" in normalized or "задач" in normalized)


def detect_monday_write_intent(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return False
    write_markers = {
        "добавь",
        "добавить",
        "создай",
        "создать",
        "обнови",
        "обновить",
        "измени",
        "изменить",
        "напиши",
        "create",
        "update",
        "edit",
        "write",
        "add",
        "post",
        "comment",
    }
    monday_object_markers = {
        "monday",
        "дос",
        "задач",
        "item",
        "board",
        "апдейт",
        "update",
    }
    return any(m in normalized for m in write_markers) and any(m in normalized for m in monday_object_markers)
