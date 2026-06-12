"""monday.com OAuth 2.0 flow + per-user token storage.

The platform owns one monday App (client id/secret). Each user runs the Authorization
Code flow once; we store *their* access token (encrypted) and call monday on their behalf,
so monday enforces that user's own permissions. monday tokens do not expire and have no
refresh token, so we only ever store the access token + granted scope.

CSRF is handled with a stateless HMAC-signed ``state`` (no server-side state table): the
state encodes the username + a nonce + a short expiry, signed with ``SECRET_KEY``.
"""
from __future__ import annotations

import base64
import hmac
import logging
import secrets
import time
from hashlib import sha256
from urllib.parse import urlencode

import httpx

from rag_agent.config import (
    RAG_MONDAY_API_VERSION,
    RAG_MONDAY_OAUTH_AUTHORIZE_URL,
    RAG_MONDAY_OAUTH_CLIENT_ID,
    RAG_MONDAY_OAUTH_CLIENT_SECRET,
    RAG_MONDAY_OAUTH_SCOPES,
    RAG_MONDAY_OAUTH_TOKEN_URL,
    SECRET_KEY,
    monday_redirect_uri,
)
from rag_agent.crypto import decrypt_secret, encrypt_secret

log = logging.getLogger(__name__)

# Signed OAuth state lifetime (seconds). The user must finish consent within this window.
_STATE_TTL_SECONDS = 600
_STATE_SEP = "\x1f"
_MONDAY_API_URL = "https://api.monday.com/v2"


# ── CSRF state (stateless, HMAC-signed) ──────────────────────────────────────
def _sign(payload_b64: str) -> str:
    return hmac.new(SECRET_KEY.encode("utf-8"), payload_b64.encode("utf-8"), sha256).hexdigest()


def make_state(username: str) -> str:
    """Return a signed, expiring state token binding the OAuth round-trip to ``username``."""
    payload = _STATE_SEP.join([username, secrets.token_urlsafe(16), str(int(time.time()) + _STATE_TTL_SECONDS)])
    payload_b64 = base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii")
    return f"{payload_b64}.{_sign(payload_b64)}"


def verify_state(state: str | None) -> str | None:
    """Validate a state token; return the bound username, or None if invalid/expired."""
    if not state or "." not in state:
        return None
    payload_b64, sig = state.rsplit(".", 1)
    if not hmac.compare_digest(sig, _sign(payload_b64)):
        return None
    try:
        username, _nonce, exp = base64.urlsafe_b64decode(payload_b64.encode("ascii")).decode("utf-8").split(_STATE_SEP)
    except Exception:  # noqa: BLE001 - any decode failure means the state is invalid
        return None
    if int(exp) < int(time.time()):
        return None
    return username or None


# ── Authorize URL + token exchange ───────────────────────────────────────────
def build_authorize_url(username: str) -> str:
    """Build the monday consent URL for ``username`` (caller redirects the browser here)."""
    params = {
        "client_id": RAG_MONDAY_OAUTH_CLIENT_ID,
        "redirect_uri": monday_redirect_uri(),
        "scope": RAG_MONDAY_OAUTH_SCOPES,
        "state": make_state(username),
    }
    return f"{RAG_MONDAY_OAUTH_AUTHORIZE_URL}?{urlencode(params)}"


def exchange_code_for_token(code: str) -> dict:
    """Exchange an authorization code for an access token.

    monday's token endpoint takes form-urlencoded params (no ``grant_type``) and returns
    ``{access_token, token_type, scope}``. Raises on HTTP error.
    """
    resp = httpx.post(
        RAG_MONDAY_OAUTH_TOKEN_URL,
        data={
            "client_id": RAG_MONDAY_OAUTH_CLIENT_ID,
            "client_secret": RAG_MONDAY_OAUTH_CLIENT_SECRET,
            "redirect_uri": monday_redirect_uri(),
            "code": code,
        },
        timeout=20.0,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_monday_identity(access_token: str) -> dict:
    """Best-effort: return ``{"name", "account_id"}`` for display. Never raises."""
    headers = {"Authorization": access_token}
    if RAG_MONDAY_API_VERSION:
        headers["API-Version"] = RAG_MONDAY_API_VERSION
    try:
        resp = httpx.post(
            _MONDAY_API_URL,
            json={"query": "query { me { name account { id } } }"},
            headers=headers,
            timeout=15.0,
        )
        resp.raise_for_status()
        me = (resp.json().get("data") or {}).get("me") or {}
        return {"name": me.get("name"), "account_id": str((me.get("account") or {}).get("id") or "") or None}
    except Exception as exc:  # noqa: BLE001 - identity is cosmetic; failures must not block connect
        log.warning("monday identity fetch failed (non-fatal): %s", exc)
        return {}


# ── Per-user token storage (encrypted at rest) ───────────────────────────────
def store_token(
    username: str,
    access_token: str,
    *,
    scope: str = "",
    token_type: str = "Bearer",
    account_id: str | None = None,
    user_name: str | None = None,
) -> bool:
    """Upsert the encrypted monday token for ``username``. Returns True on success."""
    from rag_agent.auth import get_user_id
    from rag_agent.db.models import UserMondayToken
    from rag_agent.db.session import get_session_factory

    user_id = get_user_id(username)
    if user_id is None:
        log.error("store_token: no user row for username=%r", username)
        return False

    session = get_session_factory()()
    try:
        row = session.query(UserMondayToken).filter(UserMondayToken.user_id == user_id).one_or_none()
        if row is None:
            row = UserMondayToken(user_id=user_id)
            session.add(row)
        row.access_token_encrypted = encrypt_secret(access_token)
        row.scope = scope or ""
        row.token_type = token_type or "Bearer"
        row.monday_account_id = account_id
        row.monday_user_name = user_name
        session.commit()
        return True
    finally:
        session.close()


def get_access_token(username: str) -> str | None:
    """Return the decrypted monday access token for ``username``, or None if not connected."""
    from rag_agent.auth import get_user_id
    from rag_agent.db.models import UserMondayToken
    from rag_agent.db.session import get_session_factory

    user_id = get_user_id(username)
    if user_id is None:
        return None
    session = get_session_factory()()
    try:
        row = session.query(UserMondayToken).filter(UserMondayToken.user_id == user_id).one_or_none()
        if row is None:
            return None
        try:
            return decrypt_secret(row.access_token_encrypted)
        except Exception as exc:  # noqa: BLE001 - tampered/rotated key: treat as not connected
            log.error("get_access_token: decrypt failed for username=%r: %s", username, exc)
            return None
    finally:
        session.close()


def get_connection_status(username: str) -> dict:
    """Return connection status for the settings UI (no token material)."""
    from rag_agent.auth import get_user_id
    from rag_agent.db.models import UserMondayToken
    from rag_agent.db.session import get_session_factory

    user_id = get_user_id(username)
    if user_id is None:
        return {"connected": False}
    session = get_session_factory()()
    try:
        row = session.query(UserMondayToken).filter(UserMondayToken.user_id == user_id).one_or_none()
        if row is None:
            return {"connected": False}
        return {
            "connected": True,
            "scope": row.scope or "",
            "account_id": row.monday_account_id,
            "monday_user_name": row.monday_user_name,
            "connected_at": row.created_at.isoformat() if row.created_at else None,
        }
    finally:
        session.close()


def delete_token(username: str) -> bool:
    """Delete the stored token for ``username``. Returns True if a row was removed."""
    from rag_agent.auth import get_user_id
    from rag_agent.db.models import UserMondayToken
    from rag_agent.db.session import get_session_factory

    user_id = get_user_id(username)
    if user_id is None:
        return False
    session = get_session_factory()()
    try:
        row = session.query(UserMondayToken).filter(UserMondayToken.user_id == user_id).one_or_none()
        if row is None:
            return False
        session.delete(row)
        session.commit()
        return True
    finally:
        session.close()
