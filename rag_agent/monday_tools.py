"""Load monday.com MCP tools for a specific user.

Each user has their own monday OAuth token (see ``rag_agent.monday_oauth``). We connect to
monday's remote MCP over streamable HTTP, passing *that user's* token as the ``Authorization``
header, so every tool call runs under the user's own monday permissions.

Tools are loaded per user (the schema is identical for everyone, but each tool carries the
user's auth header) and memoized by token hash for a short TTL to avoid a ``tools/list``
round-trip on every chat turn. If monday is disabled, the user is not connected, or loading
fails, callers get an empty list and the agent simply has no monday tools.

The read/write classification here is consumed by the write-confirmation layer (next phase);
it does not gate anything on its own yet.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import logging
import time

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from rag_agent.config import (
    RAG_MONDAY_API_VERSION,
    RAG_MONDAY_MCP_URL,
    monday_enabled,
)
from rag_agent.monday_oauth import get_access_token

log = logging.getLogger(__name__)

# Cache loaded tools by sha256(token) to avoid re-listing on every request.
_TOOLS_CACHE_TTL_SECONDS = 300
_tools_cache: dict[str, tuple[float, list[BaseTool]]] = {}

# Tool-name prefixes that mutate monday data. Used by the write-confirmation layer.
_WRITE_PREFIXES = (
    "create_",
    "add_",
    "update_",
    "change_",
    "set_",
    "move_",
    "duplicate_",
    "archive_",
    "delete_",
    "remove_",
    "clear_",
)


def is_write_tool(tool_name: str) -> bool:
    """True if a monday tool name looks like a mutation (vs a read/search)."""
    return (tool_name or "").lower().startswith(_WRITE_PREFIXES)


# ── Write confirmation gate ──────────────────────────────────────────────────
# Every write tool is wrapped so it only mutates monday when the model passes
# user_confirmed=true — which it is instructed (system prompt) to do ONLY after the user
# explicitly confirms the change in chat. Without it, the tool returns a CONFIRMATION_REQUIRED
# message instead of calling monday, so the agent previews the change and asks first.
_CONFIRM_ARG = "user_confirmed"


def _is_confirmed(value) -> bool:
    if value is True:
        return True
    return str(value).strip().lower() in {"true", "1", "yes", "да"}


def _confirmation_required_message(tool_name: str, args: dict) -> str:
    return (
        f"ПОДТВЕРЖДЕНИЕ ТРЕБУЕТСЯ: операция '{tool_name}' НЕ выполнена. Покажи пользователю, "
        f"что именно изменится (аргументы: {args}), и получи явное подтверждение. Только после "
        f"подтверждения вызови '{tool_name}' снова с теми же аргументами и {_CONFIRM_ARG}=true."
    )


def _args_schema_dict(tool: BaseTool) -> dict:
    """Best-effort JSON-schema dict for a tool's args (MCP tools carry a dict directly)."""
    raw = getattr(tool, "args_schema", None)
    if isinstance(raw, dict):
        return raw
    if raw is not None and hasattr(raw, "model_json_schema"):
        return raw.model_json_schema()
    return {"type": "object", "properties": {}}


def wrap_write_tool_with_confirmation(tool: BaseTool) -> BaseTool:
    """Return a confirmation-gated version of a write tool.

    Adds a synthetic boolean ``user_confirmed`` arg. The underlying monday write runs only when
    it is truthy; otherwise the wrapper returns a CONFIRMATION_REQUIRED message without touching
    monday. The original tool keeps the user's auth (it is captured in the closure), so the
    eventual write still runs under the user's own permissions.
    """
    schema = copy.deepcopy(_args_schema_dict(tool))
    schema.setdefault("properties", {})[_CONFIRM_ARG] = {
        "type": "boolean",
        "default": False,
        "description": (
            "Set to true ONLY after the user has explicitly confirmed this exact change in chat. "
            "If false or omitted, the change is NOT executed."
        ),
    }

    async def _gated(**kwargs):
        if not _is_confirmed(kwargs.pop(_CONFIRM_ARG, False)):
            return _confirmation_required_message(tool.name, kwargs)
        return await tool.ainvoke(kwargs)

    description = (
        f"{tool.description or ''}\n\n[Требует подтверждения: вызывай с {_CONFIRM_ARG}=true "
        "только после явного согласия пользователя.]"
    ).strip()
    return StructuredTool(name=tool.name, description=description, args_schema=schema, coroutine=_gated)


def build_mcp_connection(access_token: str) -> dict:
    """Return the MultiServerMCPClient connection config for a given user token.

    Only pin an API-Version header when explicitly configured; otherwise let the MCP server use
    its own current version (its tool queries target the latest schema).
    """
    headers = {"Authorization": f"Bearer {access_token}"}
    if RAG_MONDAY_API_VERSION:
        headers["API-Version"] = RAG_MONDAY_API_VERSION
    return {
        "monday": {
            "transport": "streamable_http",
            "url": RAG_MONDAY_MCP_URL,
            "headers": headers,
        }
    }


async def load_tools_for_token(access_token: str) -> list[BaseTool]:
    """Connect to the monday MCP with this token and return its tools (no caching)."""
    client = MultiServerMCPClient(build_mcp_connection(access_token))
    return await client.get_tools()


def _cache_get(token_hash: str) -> list[BaseTool] | None:
    entry = _tools_cache.get(token_hash)
    if entry and entry[0] > time.time():
        return entry[1]
    if entry:
        _tools_cache.pop(token_hash, None)
    return None


async def aget_monday_tools_for_user(username: str) -> list[BaseTool]:
    """Return the monday MCP tools for ``username`` (async path).

    Empty list when monday is disabled, the user is not connected, or loading fails — the
    chat must keep working without monday in all of those cases.
    """
    if not monday_enabled():
        return []
    access_token = await asyncio.to_thread(get_access_token, username)
    if not access_token:
        return []
    token_hash = hashlib.sha256(access_token.encode("utf-8")).hexdigest()
    cached = _cache_get(token_hash)
    if cached is not None:
        return cached
    try:
        tools = await load_tools_for_token(access_token)
    except Exception as exc:  # noqa: BLE001 - never break chat if monday is unreachable
        log.warning("Failed to load monday MCP tools for %r: %s", username, exc)
        return []
    # Reads pass through; writes are gated behind in-chat confirmation.
    tools = [wrap_write_tool_with_confirmation(t) if is_write_tool(t.name) else t for t in tools]
    _tools_cache[token_hash] = (time.time() + _TOOLS_CACHE_TTL_SECONDS, tools)
    return tools


def get_monday_tools_for_user(username: str) -> list[BaseTool]:
    """Synchronous wrapper around :func:`aget_monday_tools_for_user` for the sync /chat path.

    Safe to call from a worker thread (FastAPI runs sync endpoints off the event loop), where
    no event loop is running, so ``asyncio.run`` can drive the async MCP client.
    """
    try:
        return asyncio.run(aget_monday_tools_for_user(username))
    except RuntimeError as exc:
        # Defensive: only happens if called while an event loop is already running in this
        # thread (it should not be for the sync path). Async callers must use the async fn.
        log.error("get_monday_tools_for_user called inside a running loop: %s", exc)
        return []


def invalidate_tools_cache() -> None:
    """Drop all cached tools (e.g. after a disconnect/reconnect)."""
    _tools_cache.clear()
