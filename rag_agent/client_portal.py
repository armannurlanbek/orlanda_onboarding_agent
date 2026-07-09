"""Client portal: invite links + bridge to orlanda-api (the OrlandaBot brain).

External clients register through invite links that bake in the OrlandaBot
project ids they get access to. Their cabinet (tasks table, assistant chat,
progress pages) is served by orlanda-api over an internal HTTP channel
authenticated with the X-Internal-Key header (ORLANDA_API_KEY here must equal
CLIENT_PORTAL_API_KEY in orlanda-api's .env). The platform never stores
project access itself — orlanda-api's ProjectMember table is the single
source of truth.
"""
from __future__ import annotations

import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone

import httpx
from sqlalchemy import select

from rag_agent.config import (
    CLIENT_INVITE_EXPIRY_DAYS,
    ORLANDA_API_BASE_URL,
    ORLANDA_API_KEY,
    PROGRESS_API_SECRET,
    PROGRESS_BASE_URL,
    client_portal_enabled,
)

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = httpx.Timeout(180.0, connect=10.0)  # chat can run long tool loops


class OrlandaApiError(Exception):
    """orlanda-api is unreachable or returned an error."""


# ── orlanda-api HTTP bridge ──────────────────────────────────────────────────

def _orlanda_headers() -> dict[str, str]:
    if not client_portal_enabled():
        raise OrlandaApiError("Клиентский портал не настроен (ORLANDA_API_KEY)")
    return {"X-Internal-Key": ORLANDA_API_KEY or ""}


async def _orlanda_request(method: str, path: str, **kwargs) -> dict:
    url = f"{ORLANDA_API_BASE_URL}{path}"
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.request(method, url, headers=_orlanda_headers(), **kwargs)
    except httpx.HTTPError as exc:
        logger.error("orlanda-api %s %s failed: %s", method, path, exc)
        raise OrlandaApiError("Сервис данных временно недоступен") from exc
    if resp.status_code != 200:
        logger.error("orlanda-api %s %s -> %s: %s", method, path, resp.status_code, resp.text[:500])
        raise OrlandaApiError("Сервис данных временно недоступен")
    return resp.json()


async def orlanda_all_projects() -> list[dict]:
    """All OrlandaBot projects (for the invite admin UI picker)."""
    data = await _orlanda_request("GET", "/client/projects")
    return data.get("projects", [])


async def orlanda_get_access(user: str) -> list[dict]:
    data = await _orlanda_request("GET", "/client/access", params={"user": user})
    return data.get("projects", [])


async def orlanda_sync_access(user: str, project_ids: list[int]) -> list[int]:
    data = await _orlanda_request(
        "PUT", "/client/access", json={"user": user, "project_ids": project_ids}
    )
    return data.get("project_ids", [])


async def orlanda_tasks_table(user: str) -> dict:
    return await _orlanda_request("GET", "/client/tasks-table", params={"user": user})


async def orlanda_chat(user: str, question: str) -> dict:
    return await _orlanda_request(
        "POST", "/client/chat", json={"user": user, "question": question}
    )


async def orlanda_chat_reset(user: str) -> None:
    await _orlanda_request("POST", "/client/chat/reset", json={"user": user})


# ── Progress-tracking links ──────────────────────────────────────────────────

async def progress_links(user: str) -> list[dict]:
    """Progress-page URLs for the client's projects.

    Combines orlanda-api access (project -> monday_item_id) with the
    progress-tracking token map. Projects without a token get url=None.
    """
    projects = await orlanda_get_access(user)
    token_map: dict[str, str] = {}
    if PROGRESS_BASE_URL and PROGRESS_API_SECRET:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
                resp = await client.get(
                    f"{PROGRESS_BASE_URL}/api/project-tokens",
                    params={"secret": PROGRESS_API_SECRET},
                )
            if resp.status_code == 200:
                token_map = resp.json().get("tokens", {})
            else:
                logger.warning("progress-tracking token lookup -> %s", resp.status_code)
        except httpx.HTTPError as exc:
            logger.warning("progress-tracking token lookup failed: %s", exc)

    result = []
    for p in projects:
        token = token_map.get(str(p.get("monday_item_id") or ""))
        result.append(
            {
                "id": p["id"],
                "name": p["name"],
                "url": f"{PROGRESS_BASE_URL}/p/{token}" if token else None,
            }
        )
    return result


# ── Invites ──────────────────────────────────────────────────────────────────

def create_invite(
    *,
    created_by: str,
    project_ids: list[int],
    project_names: list[str] | None = None,
    company_name: str = "",
    max_uses: int = 1,
    expires_days: int | None = None,
) -> dict:
    """Create an invite link; returns its public dict (incl. the token)."""
    from rag_agent.db.models import ClientInvite
    from rag_agent.db.session import get_session_factory

    days = expires_days if expires_days and expires_days > 0 else CLIENT_INVITE_EXPIRY_DAYS
    invite = ClientInvite(
        id=uuid.uuid4(),
        token=secrets.token_urlsafe(24),
        company_name=(company_name or "").strip(),
        project_ids=[int(pid) for pid in project_ids],
        project_names=list(project_names or []),
        max_uses=max(1, int(max_uses or 1)),
        used_count=0,
        expires_at=datetime.now(timezone.utc) + timedelta(days=days),
        created_by=created_by,
    )
    db = get_session_factory()()
    try:
        db.add(invite)
        db.commit()
        return _invite_public(invite)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _invite_public(invite) -> dict:
    return {
        "token": invite.token,
        "company_name": invite.company_name,
        "project_ids": list(invite.project_ids or []),
        "project_names": list(invite.project_names or []),
        "max_uses": invite.max_uses,
        "used_count": invite.used_count,
        "expires_at": invite.expires_at.isoformat() if invite.expires_at else None,
        "created_by": invite.created_by,
        "created_at": invite.created_at.isoformat() if invite.created_at else None,
    }


def list_invites() -> list[dict]:
    from rag_agent.db.models import ClientInvite
    from rag_agent.db.session import get_session_factory

    db = get_session_factory()()
    try:
        rows = db.execute(
            select(ClientInvite).order_by(ClientInvite.created_at.desc())
        ).scalars().all()
        return [_invite_public(r) for r in rows]
    finally:
        db.close()


def delete_invite(token: str) -> bool:
    from rag_agent.db.models import ClientInvite
    from rag_agent.db.session import get_session_factory

    db = get_session_factory()()
    try:
        invite = db.scalar(select(ClientInvite).where(ClientInvite.token == token))
        if not invite:
            return False
        db.delete(invite)
        db.commit()
        return True
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _invite_error(invite) -> str | None:
    """Return a user-facing error if the invite cannot be used, else None."""
    if invite is None:
        return "Приглашение не найдено или отозвано"
    if invite.expires_at is not None:
        expires = invite.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        if expires < datetime.now(timezone.utc):
            return "Срок действия приглашения истёк"
    if invite.used_count >= invite.max_uses:
        return "Приглашение уже использовано"
    return None


def check_invite(token: str) -> dict:
    """Public invite preview for the registration page."""
    from rag_agent.db.models import ClientInvite
    from rag_agent.db.session import get_session_factory

    db = get_session_factory()()
    try:
        invite = db.scalar(select(ClientInvite).where(ClientInvite.token == token))
        error = _invite_error(invite)
        if error:
            return {"valid": False, "error": error}
        return {
            "valid": True,
            "company_name": invite.company_name,
            "project_names": list(invite.project_names or []),
        }
    finally:
        db.close()


async def register_client_via_invite(token: str, email: str, password: str) -> tuple[bool, str]:
    """Register a client through an invite link.

    Order matters: validate invite -> create the user -> provision access in
    orlanda-api -> consume the invite. If provisioning fails, the freshly
    created account is rolled back so the client can simply retry the link.
    Returns (ok, session_token_or_error).
    """
    from rag_agent.auth import delete_user_account, register_client
    from rag_agent.db.models import ClientInvite
    from rag_agent.db.session import get_session_factory

    email = (email or "").strip().lower()

    db = get_session_factory()()
    try:
        invite = db.scalar(select(ClientInvite).where(ClientInvite.token == token))
        error = _invite_error(invite)
        if error:
            return False, error
        project_ids = [int(pid) for pid in (invite.project_ids or [])]
    finally:
        db.close()

    ok, result = register_client(email, password)
    if not ok:
        return False, result
    session_token = result

    try:
        await orlanda_sync_access(email, project_ids)
    except OrlandaApiError as exc:
        logger.error("Access provisioning failed for %s; rolling back account", email)
        try:
            delete_user_account(email)
        except Exception:
            logger.exception("Rollback of client account %s failed", email)
        return False, f"{exc}. Попробуйте ещё раз через пару минут."

    # Consume the invite only after everything else succeeded.
    db = get_session_factory()()
    try:
        invite = db.scalar(select(ClientInvite).where(ClientInvite.token == token))
        if invite:
            invite.used_count = (invite.used_count or 0) + 1
            db.commit()
    except Exception:
        db.rollback()
        logger.exception("Failed to mark invite %s as used", token)
    finally:
        db.close()

    return True, session_token
