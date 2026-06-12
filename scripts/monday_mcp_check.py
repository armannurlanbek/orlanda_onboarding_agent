"""Connectivity spike / ops diagnostic for the monday remote MCP.

Answers the gating question for the integration: does ``https://mcp.monday.com/mcp`` accept a
pre-obtained monday token via an HTTP header (so the backend can inject each user's OAuth
token), or does it require its own interactive OAuth handshake?

Usage (token is read from env, never printed):
    python -m scripts.monday_mcp_check

Token source: RAG_MONDAY_SPIKE_TOKEN, else RAG_MONDAY_MCP_TOKEN. Provide any valid monday
token (a user's OAuth access token or a personal API token) — we only test transport/auth.
"""
from __future__ import annotations

import asyncio
import os
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient

from rag_agent.config import RAG_MONDAY_API_VERSION, RAG_MONDAY_MCP_URL


def _get_token() -> str | None:
    for key in ("RAG_MONDAY_SPIKE_TOKEN", "RAG_MONDAY_MCP_TOKEN"):
        val = (os.environ.get(key) or "").strip()
        if val:
            print(f"[token] using {key} (len={len(val)})")
            return val
    return None


def _connection(token: str, scheme: str) -> dict:
    auth = f"Bearer {token}" if scheme == "bearer" else token
    return {
        "monday": {
            "transport": "streamable_http",
            "url": RAG_MONDAY_MCP_URL,
            "headers": {"Authorization": auth, "API-Version": RAG_MONDAY_API_VERSION},
        }
    }


async def _try_scheme(token: str, scheme: str) -> bool:
    print(f"\n=== trying Authorization scheme: {scheme!r} against {RAG_MONDAY_MCP_URL} ===")
    client = MultiServerMCPClient(_connection(token, scheme))
    try:
        tools = await client.get_tools()
    except Exception as exc:  # noqa: BLE001 - we want to see the failure mode
        print(f"[{scheme}] get_tools() FAILED: {type(exc).__name__}: {str(exc)[:300]}")
        return False

    names = [t.name for t in tools]
    print(f"[{scheme}] get_tools() OK -> {len(tools)} tools")
    print(f"[{scheme}] sample: {', '.join(names[:12])}")

    # Try one no-arg read tool as a bonus end-to-end check (best-effort).
    read_candidates = [
        n for n in names
        if n.lower().startswith(("get_user", "list_workspace", "get_workspace", "me", "get_account"))
    ]
    for cand in read_candidates:
        tool = next(t for t in tools if t.name == cand)
        required = (getattr(tool, "args_schema", None) or {})
        try:
            print(f"[{scheme}] invoking read tool {cand!r} ...")
            result = await tool.ainvoke({})
            snippet = str(result)[:200].replace("\n", " ")
            print(f"[{scheme}] {cand} -> {snippet} ...")
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[{scheme}] {cand} call failed (non-fatal): {type(exc).__name__}: {str(exc)[:160]}")
    return True


async def main() -> int:
    token = _get_token()
    if not token:
        print("No token found. Set RAG_MONDAY_SPIKE_TOKEN or RAG_MONDAY_MCP_TOKEN in .env.")
        return 2
    for scheme in ("bearer", "raw"):
        if await _try_scheme(token, scheme):
            print(f"\nRESULT: SUCCESS with Authorization scheme = {scheme!r}.")
            print("=> Remote MCP accepts a pre-obtained token via header. Proceed with streamable_http.")
            return 0
    print("\nRESULT: FAILURE — remote MCP did not accept the token via header (any scheme).")
    print("=> Fall back to self-hosted stdio MCP, or revisit the auth approach.")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
