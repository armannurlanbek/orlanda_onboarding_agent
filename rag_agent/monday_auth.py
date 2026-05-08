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
import time
import contextvars
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib import parse, request

from langchain_core.tools import StructuredTool
from sqlalchemy import delete, func, select

from rag_agent.config import (
    MONDAY_ENCRYPTION_KEY,
    RAG_MONDAY_MCP_MAX_OPTIONAL_PARAMS,
    RAG_MONDAY_MCP_MAX_TOOLS,
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
)
from rag_agent.db.models import MondayConnectionState, MondayUserConnection, User
from rag_agent.db.session import get_engine, get_session_factory

logger = logging.getLogger(__name__)


def _patch_anthropic_tool_converter() -> None:
    """Patch langchain-anthropic's tool converter to strip unsupported constraints.

    Anthropic's API rejects some JSON Schema constraints (e.g. number and array
    bound constraints) in tool definitions.
    `convert_to_anthropic_tool` builds the exact dict sent over the wire — cleaning
    its output is the most direct fix and bypasses Pydantic schema caching issues.
    """
    import copy as _copy

    _STRIP_KEYS_BY_TYPE = {
        "number": ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"),
        "integer": ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"),
        "array": ("minItems", "maxItems", "uniqueItems"),
    }

    def _strip(obj):
        if isinstance(obj, dict):
            obj_type = str(obj.get("type") or "").strip().lower()
            for k in _STRIP_KEYS_BY_TYPE.get(obj_type, ()):
                    obj.pop(k, None)
            for v in list(obj.values()):
                _strip(v)
        elif isinstance(obj, list):
            for item in obj:
                _strip(item)
        return obj

    def _wrap(module, attr_name: str) -> bool:
        original = getattr(module, attr_name, None)
        if original is None or getattr(original, "_anthropic_strip_patched", False):
            return False

        def _patched(*args, **kwargs):
            result = original(*args, **kwargs)
            if isinstance(result, dict):
                cleaned = _copy.deepcopy(result)
                _strip(cleaned)
                return cleaned
            return result

        _patched._anthropic_strip_patched = True  # type: ignore[attr-defined]
        setattr(module, attr_name, _patched)
        return True

    patched_any = False

    # 1. langchain_anthropic.chat_models.convert_to_anthropic_tool — final dict to API
    try:
        import langchain_anthropic.chat_models as _ac_chat
        if _wrap(_ac_chat, "convert_to_anthropic_tool"):
            patched_any = True
    except Exception:
        pass

    # 2. Same symbol on the package root, in case it's imported from there
    try:
        import langchain_anthropic as _ac_root
        if _wrap(_ac_root, "convert_to_anthropic_tool"):
            patched_any = True
    except Exception:
        pass

    # 3. Sibling module location used by some versions
    try:
        import langchain_anthropic._client as _ac_client  # type: ignore[import-not-found]
        if _wrap(_ac_client, "convert_to_anthropic_tool"):
            patched_any = True
    except Exception:
        pass

    # 4. Belt-and-suspenders: clean langchain_core's openai-tool converter too
    try:
        import langchain_core.utils.function_calling as _fc

        original = getattr(_fc, "convert_to_openai_tool", None)
        if original is not None and not getattr(original, "_anthropic_strip_patched", False):
            def _patched_oai(*args, **kwargs):
                result = original(*args, **kwargs)
                if isinstance(result, dict):
                    cleaned = _copy.deepcopy(result)
                    _strip(cleaned)
                    return cleaned
                return result

            _patched_oai._anthropic_strip_patched = True  # type: ignore[attr-defined]
            _fc.convert_to_openai_tool = _patched_oai
            patched_any = True
    except Exception:
        pass

    if patched_any:
        logger.info("Patched langchain tool converters to strip Anthropic-incompatible schema constraints")
    else:
        logger.warning("Could not locate langchain tool converter to patch — Anthropic tool calls may 400")


_patch_anthropic_tool_converter()


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
        connected = bool(token and not revoked)
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
        if conn.expires_at and conn.expires_at <= _utcnow():
            return ""
        return _decrypt_token(conn.access_token_encrypted)
    finally:
        db.close()


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


def _run_async_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    logger.warning("Cannot load MCP tools: running event loop detected in sync path.")
    return None


def _sanitize_tools_for_anthropic(tools: list) -> list:
    """Strip number-range constraints (Ge/Le/Gt/Lt/MultipleOf) from MCP tool args_schema.

    Anthropic's API rejects `minimum`/`maximum` on number/integer JSON schema types.
    LangChain's `tool_call_schema` builds a fresh Pydantic model via `_create_subset_model`,
    so overriding `model_json_schema` via subclassing doesn't survive — we must strip
    the constraints at the FieldInfo metadata level (and from `Annotated[...]` annotations)
    so the regenerated schema contains no minimum/maximum to begin with.
    """
    import copy
    import typing
    from pydantic import create_model
    from langchain_core.tools import StructuredTool as _ST

    _STRIP_CONSTRAINTS = frozenset({
        "Ge", "Le", "Gt", "Lt", "MultipleOf", "Interval",
        "_PydanticGeneralMetadata",
    })
    _STRIP_JSON_KEYS = frozenset({
        "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    })

    def _is_constraint(meta_obj) -> bool:
        return type(meta_obj).__name__ in _STRIP_CONSTRAINTS

    def _strip_annotation(ann):
        """If `ann` is Annotated[T, Ge(0), Le(10)], return Annotated[T, ...] with constraints removed."""
        meta = getattr(ann, "__metadata__", None)
        if not meta:
            return ann
        base = getattr(ann, "__origin__", None)
        if base is None:
            return ann
        kept = tuple(m for m in meta if not _is_constraint(m))
        if not kept:
            return base
        return typing.Annotated[(base, *kept)]

    def _strip_field(field_info):
        new_field = copy.copy(field_info)
        meta = list(getattr(new_field, "metadata", []) or [])
        new_meta = [m for m in meta if not _is_constraint(m)]
        try:
            new_field.metadata = new_meta
        except Exception:
            pass
        # Some Pydantic versions also store constraints directly on FieldInfo attrs.
        for attr in ("ge", "le", "gt", "lt", "multiple_of"):
            try:
                if getattr(new_field, attr, None) is not None:
                    setattr(new_field, attr, None)
            except Exception:
                pass
        return new_field

    def _build_clean_schema(schema_cls):
        new_fields = {}
        for fname, finfo in schema_cls.model_fields.items():
            new_ann = _strip_annotation(finfo.annotation)
            new_fields[fname] = (new_ann, _strip_field(finfo))
        cleaned = create_model(f"Clean_{schema_cls.__name__}", **new_fields)

        # Belt-and-suspenders: also override model_json_schema to strip leftover
        # numeric constraints from the generated dict.
        def _clean_dict(obj):
            if isinstance(obj, dict):
                if obj.get("type") in ("number", "integer"):
                    for k in _STRIP_JSON_KEYS:
                        obj.pop(k, None)
                for v in list(obj.values()):
                    _clean_dict(v)
            elif isinstance(obj, list):
                for item in obj:
                    _clean_dict(item)
            return obj

        original_mjs = cleaned.model_json_schema

        def _patched_mjs(*args, **kw):
            return _clean_dict(copy.deepcopy(original_mjs(*args, **kw)))

        try:
            cleaned.model_json_schema = _patched_mjs  # type: ignore[assignment]
        except Exception:
            pass
        return cleaned

    result = []
    for tool in tools:
        schema_cls = getattr(tool, "args_schema", None)
        func = getattr(tool, "func", None)
        coroutine = getattr(tool, "coroutine", None)
        if schema_cls is None or not hasattr(schema_cls, "model_fields"):
            result.append(tool)
            continue
        if func is None and coroutine is None:
            result.append(tool)
            continue
        try:
            cleaned_cls = _build_clean_schema(schema_cls)
            new_tool = _ST.from_function(
                name=tool.name,
                description=tool.description or "",
                args_schema=cleaned_cls,
                func=func,
                coroutine=coroutine,
            )
            result.append(new_tool)
        except Exception as e:
            logger.warning(
                "Failed to sanitize monday tool %s for anthropic: %s",
                getattr(tool, "name", "?"),
                e,
            )
            result.append(tool)

    return result


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
                    txt = str(e or "")
                    if "MCP error -32602" in txt or "Input validation error" in txt:
                        return _as_mcp_guard_result(
                            name or "monday_tool",
                            f"Monday tool validation failed for `{name or 'monday_tool'}`: {txt}",
                        )
                    raise

            async def _safe_existing_coroutine(**kwargs):
                missing_res = _guard_and_format_missing(kwargs)
                if missing_res:
                    return missing_res
                if not callable(coroutine):
                    return _safe_sync_func(**kwargs)
                try:
                    _bump_monday_call(name or "monday_tool")
                    return await coroutine(**kwargs)
                except Exception as e:
                    txt = str(e or "")
                    if "MCP error -32602" in txt or "Input validation error" in txt:
                        return _as_mcp_guard_result(
                            name or "monday_tool",
                            f"Monday tool validation failed for `{name or 'monday_tool'}`: {txt}",
                        )
                    raise

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
                    return asyncio.run(async_tool.ainvoke(kwargs))
                except Exception as e:
                    txt = str(e or "")
                    if "MCP error -32602" in txt or "Input validation error" in txt:
                        return _as_mcp_guard_result(
                            tool_name,
                            f"Monday tool validation failed for `{tool_name}`: {txt}",
                        )
                    raise

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
                return await coro_tool(**kwargs)
            except Exception as e:
                txt = str(e or "")
                if "MCP error -32602" in txt or "Input validation error" in txt:
                    return _as_mcp_guard_result(
                        tool_name,
                        f"Monday tool validation failed for `{tool_name}`: {txt}",
                    )
                raise

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


def _cap_monday_tools(tools: list, max_tools: int) -> list:
    """Cap tool count to satisfy provider limits on strict tools."""
    cap = max(1, int(max_tools or 1))
    if len(tools) <= cap:
        return tools

    exact_priority = [
        "get_user_context",
        "search",
        "get_board_items_page",
        "create_update",
        "change_simple_column_value",
        "change_multiple_column_values",
        "create_item",
        "create_subitem",
        "delete_item",
        "move_item_to_group",
    ]
    priority_index = {name: i for i, name in enumerate(exact_priority)}
    write_markers = ("create_", "update", "delete", "change_", "move_", "set_", "edit", "write", "add_")

    def _rank(tool_obj):
        name = str(getattr(tool_obj, "name", "") or "").strip().lower()
        score = 0
        if name in priority_index:
            score += 1000 - priority_index[name]
        if any(marker in name for marker in write_markers):
            score += 200
        if name.startswith("get_"):
            score += 100
        if "search" in name:
            score += 80
        return (-score, name)

    ranked = sorted(list(tools), key=_rank)
    kept = ranked[:cap]
    logger.warning(
        "Capped monday MCP tools from %d to %d to satisfy provider strict-tool limits",
        len(tools),
        len(kept),
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
    """Keep tool schemas under provider optional-parameter complexity budget."""
    budget = max(1, int(max_optional_params or 1))
    if not tools:
        return []

    kept: list = []
    used = 0
    # Tools are already priority-sorted/capped before this step.
    for tool_obj in tools:
        opt = _optional_param_count(tool_obj)
        # Always allow a zero/low optional tool; for very first tool, keep at least one.
        if kept and (used + opt) > budget:
            continue
        kept.append(tool_obj)
        used += opt
        if used >= budget:
            break

    if len(kept) < len(tools):
        logger.warning(
            "Capped monday tool optional params from total %d to %d (budget=%d), tools kept=%d/%d",
            sum(_optional_param_count(t) for t in tools),
            sum(_optional_param_count(t) for t in kept),
            budget,
            len(kept),
            len(tools),
        )
    return kept


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
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError:
        logger.warning(
            "langchain-mcp-adapters is not installed; monday tools unavailable"
        )
        return []
    client = MultiServerMCPClient(
        {
            "monday": {
                "transport": RAG_MONDAY_MCP_TRANSPORT,
                "url": RAG_MONDAY_MCP_URL,
                "headers": {"Authorization": f"Bearer {token}"},
                "timeout": RAG_MONDAY_MCP_TIMEOUT_SECONDS,
            }
        }
    )
    tools = _run_async_sync(client.get_tools()) or []
    tools = _ensure_sync_callable_tools(tools)
    tools = _sanitize_tools_for_anthropic(tools)
    if RAG_MONDAY_MCP_TOOL_ALLOWLIST:
        tools = [t for t in tools if str(getattr(t, "name", "")).strip() in RAG_MONDAY_MCP_TOOL_ALLOWLIST]
    tools = _cap_monday_tools(tools, RAG_MONDAY_MCP_MAX_TOOLS)
    tools = _cap_optional_params(tools, RAG_MONDAY_MCP_MAX_OPTIONAL_PARAMS)
    _tools_cache[key_digest] = (now, list(tools))
    return tools


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
