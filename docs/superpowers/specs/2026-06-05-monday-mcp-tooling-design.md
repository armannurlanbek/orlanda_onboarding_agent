# Monday MCP tooling fixes — design spec

**Date:** 2026-06-05
**Branch:** `monday-mcp-tooling-fixes`
**Goal:** Goal 1 from the /council audit — "the AI agent works poorly with Monday.com." Fix the tool surface and the OAuth/session correctness bugs behind it. No architectural rewrite.

## Context

The agent loads Monday tools per-user from the hosted Monday MCP (`https://mcp.monday.com/mcp`) via `langchain-mcp-adapters`. Two load paths share `_prepare_monday_tools`:

- **Sync `/chat`** → `get_monday_mcp_tools_for_user` (connection-bound tools, one short-lived HTTP session per call, cached 600s).
- **Async `/chat/stream`** → `monday_session_tools` (one MCP session reused for the whole turn).

The council found the integration is defensively well-built (tested tool-capping, per-turn session, bounded retry) but **over-built around a bloated ~36-40 tool surface** and carrying **two real OAuth correctness gaps**. The live Orlanda Engineering workspace (id **49959**) was confirmed: 3-tier boards Clients `1479400337` → Projects `1542627875` → TO DO LIST `1542673123` (~2300 items).

## Decisions

- **Capability level:** Read + safe writes (owner-selected). 10-tool allowlist. No `all_monday_api` escape hatch.
- **Mid-turn expiry:** widen proactive refresh leeway + a reconnect guard (not full reactive re-auth). Owner-approved.
- **Cache key:** keep token-scoped — deliberately *not* taking the council's "key on username" suggestion (see Fix 2).

## The 10-tool allowlist

```
reads (7):
  get_user_context        resolve "me"/assignee -> numeric user id
  get_board_info          resolve board-specific column ids (anti-hallucination)
  get_board_items_page    the actual task/project/client reads
  search                  find a board/item/doc by fuzzy name
  get_column_type_info    build valid items_page filters
  get_updates             read task discussion/communication
  list_users_and_teams    map names <-> ids for owner/assignee questions
writes (3):
  create_update           the sanctioned communication primitive
  change_item_column_values  status/owner updates
  create_item             new tasks
dropped: all_monday_api, list_workspaces, workspace_info, + ~25 niche/create tools
```

Rationale: this is overwhelmingly a read/reporting assistant with occasional updates. `all_monday_api` (arbitrary GraphQL) is the biggest blast-radius/complexity risk and call-stats can't classify its safety. There is exactly one relevant workspace, so `list_workspaces`/`workspace_info` are unnecessary — its id is baked into the prompt.

## Fixes

### Fix 1 — Curated allowlist + prompt alignment
- `config.py:82-93`: when `RAG_MONDAY_MCP_TOOL_ALLOWLIST` env is **unset**, default to the 10-tool set. An explicit env value still overrides. To restore the old "allow everything, rely on the cap" behavior, set the env to the sentinel `*` (or `all`) → empties the allowlist so `_cap_monday_tools` does the filtering again. Rewrite the stale comments at `:85-99` that describe the old 25-cap rationale.
- Effect: `_prepare_monday_tools` (`monday_auth.py:1053-1054`) filters to 10; `_cap_monday_tools` (`:891`) becomes a no-op but stays as a safety net.
- `system_prompt.yaml` Monday section: verified it already carries the board map (lines 80, 104) and does **not** reference any dropped tool. Light change: confirm no dropped-tool references remain; add an explicit note that the single workspace is Orlanda Engineering / 49959 so the agent never reaches for workspace discovery.

### Fix 2 — Token-refresh correctness (`monday_auth.py`)
- `_token_needs_refresh` (`:438-444`): stop treating `expires_at is None` as "never expires." When expiry is unknown, fall back to a max-age check on `conn.updated_at` (new config `RAG_MONDAY_TOKEN_MAX_AGE_SECONDS`, default 1800s) so the token still rotates instead of being used forever.
- `_refresh_access_token_for_user` (`:447-536`): when Monday *rejects* the refresh (`invalid_grant`, or an HTTP 400-class with an error payload), set `conn.revoked_at` so `get_monday_status` flips to `connected:false` and the UI prompts reconnect. **Transient/network failures (URLError/TimeoutError/5xx) must NOT revoke** — they just return `""` for this turn as today.
- **Cache key stays token-scoped** (`:1066`). The per-call tools embed the token in their client `Authorization` header; keying on username alone would serve a stale token after a refresh and 401 later. The "waste" the council flagged is one catalog reload per token lifetime — acceptable and correct. Documented in-code so it isn't "fixed" later by mistake.
- Also wrap the OAuth *callback* token exchange (`complete_monday_oauth_callback` → `_post_form`, `:356`) so a Monday 4xx becomes a clean error rather than a raw 500. (`_post_form` `urlopen` raises `HTTPError` on non-2xx; the refresh path catches it, the callback path does not.)

### Fix 3 — Coroutine-only session tools (`monday_auth.py:739-846`)
- Thread a `for_session: bool` flag: `monday_session_tools` → `_prepare_monday_tools` → `_ensure_sync_callable_tools`.
- When `for_session=True`, build session-bound tools **coroutine-only** (`StructuredTool` with `coroutine=`, no `_make_sync_runner`). A `ClientSession` is loop-bound; the current sync runner (`:837-844`) routes through `_run_coro_on_background_loop` (`:579-608`) — a different thread+loop — which, if ever invoked, breaks the session's loop affinity and produces "session terminated" 500s.
- The connection-bound per-call path (sync `/chat`, no running loop) **keeps** its sync runner — it genuinely needs `_run_async_sync` → `asyncio.run`.
- Net: nothing can drive the loop-bound session from a foreign loop; a stray sync `.invoke()` on a session tool fails loudly (NotImplementedError) instead of silently corrupting the session.

### Fix 4 — Mid-turn token expiry
- **Primary:** make `_TOKEN_REFRESH_LEEWAY_SECONDS` (`:420`) a config (`RAG_MONDAY_TOKEN_REFRESH_LEEWAY_SECONDS`, default 300). The token is fetched at `monday_session_tools` `:1118` *before* the client/session is built, so any turn starting with <5 min of token life refreshes first. Combined with the allowlist shrinking turn length, mid-turn expiry becomes negligible.
- **Secondary:** add an auth-expired branch to `_format_tool_error_for_model` (`:96-137`) that detects 401 / "unauthorized" / "token expired" / "invalid token" and returns a clear *"your Monday session expired — please reconnect Monday in settings"* guard (non-retryable, distinct from the existing permission hint).
- **Rejected (revisit later):** full reactive 401 → refresh → rebuild session → retry mid-stream. More correct but tears down/rebuilds the open session mid-turn, destabilizing the just-stabilized streaming path, for a rare case. The baked-in token means an open session can't self-heal anyway; the next turn's proactive refresh covers it.

## Tests (extend `rag_agent/test_monday_tool_capping.py`)

1. With the default allowlist, `_prepare_monday_tools` on a simulated full catalog yields exactly the 10 allowlisted tools; `all_monday_api`/`list_workspaces`/`workspace_info` are absent; `_cap_monday_tools` is a no-op (no "Capped" warning).
2. `_token_needs_refresh`: `expires_at=None` + `updated_at` older than max-age → True; fresh token within validity → False; missing token → True.
3. `_refresh_access_token_for_user`: an `invalid_grant` response sets `revoked_at`; a simulated network error (`URLError`) does **not** set `revoked_at` and returns `""`.
4. `_ensure_sync_callable_tools(..., for_session=True)` produces tools whose `.func` is None (coroutine-only); `for_session=False` keeps a callable `.func`.

## Out of scope (other council items, not this spec)

- P0 secret rotation, failover data-loss, migration reconciliation (separate workstream).
- The brittle Russian-keyword write-intent gating in `api.py` (the allowlist defines which tools *can* bind; existing gating decides *when*). Untouched here.
- Frontend (Goal 2).

## Risk / rollback

All changes are behind env defaults and a feature branch. Kill-switches preserved:
- `RAG_MONDAY_MCP_TOOL_ALLOWLIST=*` (or `all`) → disables the curated allowlist, reverting to the old cap-based filtering.
- `RAG_MONDAY_MCP_TOOL_ALLOWLIST=<explicit,list>` → any custom set.
- `RAG_MONDAY_MCP_PERSISTENT_SESSION=false` → still falls back to per-call tools.

No DB migrations; `revoked_at` is an existing column. New config keys (`RAG_MONDAY_TOKEN_MAX_AGE_SECONDS`, `RAG_MONDAY_TOKEN_REFRESH_LEEWAY_SECONDS`) have safe defaults, so no env change is required to deploy.
