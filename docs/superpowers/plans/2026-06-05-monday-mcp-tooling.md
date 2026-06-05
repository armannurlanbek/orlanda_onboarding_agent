# Monday MCP Tooling Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the agent work reliably with Monday.com by curating the tool surface to 10 read+safe-write tools and closing the OAuth refresh / session correctness gaps behind the "works poorly with Monday" complaint.

**Architecture:** Surgical edits to `rag_agent/config.py` and `rag_agent/monday_auth.py` (no architectural change). A curated allowlist replaces the 25-tool heuristic cap; token refresh handles unknown-expiry and revokes only on a *rejected* grant; session-bound tools become coroutine-only so they can't be driven off-loop; a widened proactive refresh leeway plus an auth-expired guard handle mid-turn expiry. The streaming path is untouched.

**Tech Stack:** Python 3.12, FastAPI, LangChain (`StructuredTool`), `langchain-mcp-adapters`, SQLAlchemy, pytest (the existing `test_monday_tool_capping.py` also self-runs via `python -m`).

**Spec:** `docs/superpowers/specs/2026-06-05-monday-mcp-tooling-design.md`

---

## File Structure

- `rag_agent/config.py` — add 3 config values + an allowlist parser; repoint `RAG_MONDAY_MCP_TOOL_ALLOWLIST`.
- `rag_agent/monday_auth.py` — token-refresh correctness, coroutine-only session tools, auth-expired guard, cache-key comment.
- `rag_agent/system_prompt.yaml` — verification only (already aligned).
- `rag_agent/test_monday_tool_capping.py` — extend with new regression tests (zero-arg functions so the file's `_run_all()` and pytest both work).

Run tests throughout with:
`python -m rag_agent.test_monday_tool_capping` (fast self-runner) or `pytest rag_agent/test_monday_tool_capping.py -v`.

---

## Task 1: Config — new keys + allowlist parser

**Files:**
- Modify: `rag_agent/config.py:82-93`
- Test: `rag_agent/test_monday_tool_capping.py`

- [ ] **Step 1: Write the failing tests** (append to `rag_agent/test_monday_tool_capping.py`, above `_run_all`)

```python
import os as _os


def test_allowlist_parser_default_when_unset():
    from rag_agent.config import _parse_monday_tool_allowlist, RAG_MONDAY_MCP_DEFAULT_TOOL_ALLOWLIST
    saved = _os.environ.pop("RAG_MONDAY_MCP_TOOL_ALLOWLIST", None)
    try:
        assert _parse_monday_tool_allowlist() == set(RAG_MONDAY_MCP_DEFAULT_TOOL_ALLOWLIST)
        assert len(RAG_MONDAY_MCP_DEFAULT_TOOL_ALLOWLIST) == 10
    finally:
        if saved is not None:
            _os.environ["RAG_MONDAY_MCP_TOOL_ALLOWLIST"] = saved


def test_allowlist_parser_star_disables():
    from rag_agent.config import _parse_monday_tool_allowlist
    saved = _os.environ.get("RAG_MONDAY_MCP_TOOL_ALLOWLIST")
    _os.environ["RAG_MONDAY_MCP_TOOL_ALLOWLIST"] = "*"
    try:
        assert _parse_monday_tool_allowlist() == set()
    finally:
        if saved is None:
            _os.environ.pop("RAG_MONDAY_MCP_TOOL_ALLOWLIST", None)
        else:
            _os.environ["RAG_MONDAY_MCP_TOOL_ALLOWLIST"] = saved


def test_allowlist_parser_explicit_list():
    from rag_agent.config import _parse_monday_tool_allowlist
    saved = _os.environ.get("RAG_MONDAY_MCP_TOOL_ALLOWLIST")
    _os.environ["RAG_MONDAY_MCP_TOOL_ALLOWLIST"] = "search, get_board_info"
    try:
        assert _parse_monday_tool_allowlist() == {"search", "get_board_info"}
    finally:
        if saved is None:
            _os.environ.pop("RAG_MONDAY_MCP_TOOL_ALLOWLIST", None)
        else:
            _os.environ["RAG_MONDAY_MCP_TOOL_ALLOWLIST"] = saved
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest rag_agent/test_monday_tool_capping.py -k allowlist_parser -v`
Expected: FAIL — `ImportError: cannot import name '_parse_monday_tool_allowlist'`.

- [ ] **Step 3: Implement the config changes** — replace `rag_agent/config.py:82-93` (the current `RAG_MONDAY_MCP_TOOL_ALLOWLIST` set-comprehension and the `RAG_MONDAY_MCP_MAX_TOOLS` block stays) with:

```python
# Curated default Monday tool allowlist: read + safe-write tools an onboarding
# agent actually needs. The full hosted catalog is ~36-40 tools, which bloats the
# prompt and causes wrong-tool selection. Override via env:
#   RAG_MONDAY_MCP_TOOL_ALLOWLIST="search,get_board_info,..."  -> explicit set
#   RAG_MONDAY_MCP_TOOL_ALLOWLIST="*" (or "all")               -> disable allowlist
#                                                                 (cap does filtering)
RAG_MONDAY_MCP_DEFAULT_TOOL_ALLOWLIST = frozenset({
    # reads (7)
    "get_user_context",
    "get_board_info",
    "get_board_items_page",
    "search",
    "get_column_type_info",
    "get_updates",
    "list_users_and_teams",
    # safe writes (3)
    "create_update",
    "change_item_column_values",
    "create_item",
})


def _parse_monday_tool_allowlist() -> set[str]:
    """Resolve the Monday tool allowlist from env.

    Unset/empty -> the curated default (10 tools). The sentinel ``*``/``all``
    disables the allowlist so ``_cap_monday_tools`` does the filtering. Any other
    value is parsed as a comma-separated explicit set.
    """
    raw = (os.environ.get("RAG_MONDAY_MCP_TOOL_ALLOWLIST") or "").strip()
    if not raw:
        return set(RAG_MONDAY_MCP_DEFAULT_TOOL_ALLOWLIST)
    if raw.lower() in {"*", "all"}:
        return set()
    return {name.strip() for name in raw.split(",") if name.strip()}


RAG_MONDAY_MCP_TOOL_ALLOWLIST = _parse_monday_tool_allowlist()
# Safety-net cap (rarely fires now the allowlist is ~10). Kept so an explicit
# wide/`*` allowlist still can't flood the model with the whole catalog.
RAG_MONDAY_MCP_MAX_TOOLS = max(
    1,
    min(128, int(os.environ.get("RAG_MONDAY_MCP_MAX_TOOLS", "25"))),
)

# Monday OAuth token freshness.
# Proactively refresh this many seconds before `expires_at` so a multi-step chat
# turn never starts on a token about to expire (was a hard-coded 60s).
RAG_MONDAY_TOKEN_REFRESH_LEEWAY_SECONDS = max(
    30,
    min(3600, int(os.environ.get("RAG_MONDAY_TOKEN_REFRESH_LEEWAY_SECONDS", "300"))),
)
# Fallback when Monday omits `expires_in` (no `expires_at` stored): refresh once the
# token is older than this, so it still rotates instead of being used forever.
RAG_MONDAY_TOKEN_MAX_AGE_SECONDS = max(
    300,
    int(os.environ.get("RAG_MONDAY_TOKEN_MAX_AGE_SECONDS", "1800")),
)
```

Note: this *replaces* the old `RAG_MONDAY_MCP_TOOL_ALLOWLIST = {...}` and the old `RAG_MONDAY_MCP_MAX_TOOLS = ...` block at `:82-93`. Leave `RAG_MONDAY_MCP_MAX_OPTIONAL_PARAMS` (`:100`) and everything below unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest rag_agent/test_monday_tool_capping.py -k allowlist_parser -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add rag_agent/config.py rag_agent/test_monday_tool_capping.py
git commit -m "Add curated Monday tool allowlist + token-freshness config"
```

---

## Task 2: Curated allowlist end-to-end through `_prepare_monday_tools`

**Files:**
- Test: `rag_agent/test_monday_tool_capping.py`

No production change — the allowlist filter already exists at `monday_auth.py:1053-1054`; this locks in that the new default yields exactly the 10 tools and the cap becomes a no-op.

- [ ] **Step 1: Write the failing test** (append above `_run_all`)

```python
def test_default_allowlist_yields_exactly_ten_tools():
    import rag_agent.monday_auth as ma
    from rag_agent.config import RAG_MONDAY_MCP_DEFAULT_TOOL_ALLOWLIST
    saved = ma.RAG_MONDAY_MCP_TOOL_ALLOWLIST
    ma.RAG_MONDAY_MCP_TOOL_ALLOWLIST = set(RAG_MONDAY_MCP_DEFAULT_TOOL_ALLOWLIST)
    try:
        tools = [_FakeTool(n) for n in _REAL_MONDAY_TOOLNAMES]
        kept = _names(ma._prepare_monday_tools(tools))
        assert kept == set(RAG_MONDAY_MCP_DEFAULT_TOOL_ALLOWLIST)
        for dropped in ("all_monday_api", "list_workspaces", "workspace_info",
                        "create_board", "get_full_board_data"):
            assert dropped not in kept
    finally:
        ma.RAG_MONDAY_MCP_TOOL_ALLOWLIST = saved
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest rag_agent/test_monday_tool_capping.py -k yields_exactly_ten -v`
Expected: FAIL if `_prepare_monday_tools` does not yet honor the default (e.g., before Task 1 is loaded). If it already passes after Task 1, note that and continue — this test is a regression guard.

- [ ] **Step 3: No implementation needed** — the filter exists. If the test fails because `_prepare_monday_tools` capped differently, confirm `RAG_MONDAY_MCP_MAX_TOOLS >= 10` in `config.py` (it is 25).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest rag_agent/test_monday_tool_capping.py -k yields_exactly_ten -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rag_agent/test_monday_tool_capping.py
git commit -m "Lock in curated allowlist yields exactly the 10 Monday tools"
```

---

## Task 3: System prompt alignment (verification gate)

**Files:**
- Verify: `rag_agent/system_prompt.yaml`

- [ ] **Step 1: Confirm no dropped-tool references**

Run: `rg -n "all_monday_api|list_workspaces|workspace_info" rag_agent/system_prompt.yaml`
Expected: no matches. The prompt already carries the workspace board map ("Карта рабочего пространства Monday (Orlanda Engineering)", lines ~77-104) and references only allowlisted tools (`get_board_info`, `get_board_items_page`, write tools).

- [ ] **Step 2: If (and only if) a match is found**, remove the sentence that instructs using that dropped tool, keeping the surrounding workflow intact. Otherwise make no edit.

- [ ] **Step 3: No commit if unchanged.** If an edit was needed:

```bash
git add rag_agent/system_prompt.yaml
git commit -m "Drop references to removed Monday tools from system prompt"
```

---

## Task 4: Token refresh — handle unknown expiry + config-driven leeway

**Files:**
- Modify: `rag_agent/monday_auth.py:24-44` (imports), `:418-444` (leeway constant + `_token_needs_refresh`)
- Test: `rag_agent/test_monday_tool_capping.py`

- [ ] **Step 1: Write the failing tests** (append above `_run_all`)

```python
def _fake_conn(expires_at=None, updated_at=None):
    class _C:
        pass
    c = _C()
    c.expires_at = expires_at
    c.updated_at = updated_at
    return c


def test_token_needs_refresh_missing_token():
    from rag_agent.monday_auth import _token_needs_refresh
    assert _token_needs_refresh(_fake_conn(), "") is True


def test_token_needs_refresh_none_expiry_old_updated():
    from datetime import datetime, timezone, timedelta
    from rag_agent.monday_auth import _token_needs_refresh, _TOKEN_MAX_AGE_SECONDS
    old = datetime.now(timezone.utc) - timedelta(seconds=_TOKEN_MAX_AGE_SECONDS + 120)
    assert _token_needs_refresh(_fake_conn(expires_at=None, updated_at=old), "tok") is True


def test_token_needs_refresh_none_expiry_fresh_updated():
    from datetime import datetime, timezone, timedelta
    from rag_agent.monday_auth import _token_needs_refresh
    fresh = datetime.now(timezone.utc) - timedelta(seconds=10)
    assert _token_needs_refresh(_fake_conn(expires_at=None, updated_at=fresh), "tok") is False


def test_token_needs_refresh_future_expiry_is_false():
    from datetime import datetime, timezone, timedelta
    from rag_agent.monday_auth import _token_needs_refresh, _TOKEN_REFRESH_LEEWAY_SECONDS
    future = datetime.now(timezone.utc) + timedelta(seconds=_TOKEN_REFRESH_LEEWAY_SECONDS + 600)
    assert _token_needs_refresh(_fake_conn(expires_at=future), "tok") is False


def test_token_needs_refresh_near_expiry_is_true():
    from datetime import datetime, timezone, timedelta
    from rag_agent.monday_auth import _token_needs_refresh, _TOKEN_REFRESH_LEEWAY_SECONDS
    soon = datetime.now(timezone.utc) + timedelta(seconds=_TOKEN_REFRESH_LEEWAY_SECONDS - 5)
    assert _token_needs_refresh(_fake_conn(expires_at=soon), "tok") is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest rag_agent/test_monday_tool_capping.py -k token_needs_refresh -v`
Expected: FAIL — `ImportError` on `_TOKEN_MAX_AGE_SECONDS` (not defined yet) / current `None`-expiry returns False for the old-updated case.

- [ ] **Step 3a: Add the config imports** — in `rag_agent/monday_auth.py`, inside the `from rag_agent.config import ( ... )` block (`:24-44`), add these three names (alphabetical-ish, anywhere in the tuple):

```python
    RAG_MONDAY_TOKEN_MAX_AGE_SECONDS,
    RAG_MONDAY_TOKEN_REFRESH_LEEWAY_SECONDS,
```

(`RAG_MONDAY_MCP_TOOL_ALLOWLIST` is already imported.)

- [ ] **Step 3b: Replace the leeway constant** — `rag_agent/monday_auth.py:418-420`:

```python
# Seconds before `expires_at` at which we proactively refresh the access token,
# so an in-flight chat turn never starts with a token about to expire. Widened
# from 60s and made configurable so even a long multi-step turn refreshes first.
_TOKEN_REFRESH_LEEWAY_SECONDS = RAG_MONDAY_TOKEN_REFRESH_LEEWAY_SECONDS
# Fallback freshness window when Monday omits `expires_in` (no `expires_at`).
_TOKEN_MAX_AGE_SECONDS = RAG_MONDAY_TOKEN_MAX_AGE_SECONDS
```

- [ ] **Step 3c: Replace `_token_needs_refresh`** — `rag_agent/monday_auth.py:438-444`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest rag_agent/test_monday_tool_capping.py -k token_needs_refresh -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add rag_agent/monday_auth.py rag_agent/test_monday_tool_capping.py
git commit -m "Refresh Monday token on unknown expiry; configurable leeway"
```

---

## Task 5: Revoke connection only on a rejected refresh grant

**Files:**
- Modify: `rag_agent/monday_auth.py:447-536` (`_refresh_access_token_for_user`) + a new helper just above it
- Test: `rag_agent/test_monday_tool_capping.py`

- [ ] **Step 1: Write the failing tests** (append above `_run_all`)

```python
def test_permanent_refresh_failure_invalid_grant():
    from rag_agent.monday_auth import _is_permanent_refresh_failure
    assert _is_permanent_refresh_failure(error="invalid_grant") is True


def test_permanent_refresh_failure_http_400():
    from rag_agent.monday_auth import _is_permanent_refresh_failure
    assert _is_permanent_refresh_failure(http_code=400) is True


def test_transient_refresh_failure_http_503():
    from rag_agent.monday_auth import _is_permanent_refresh_failure
    assert _is_permanent_refresh_failure(http_code=503) is False


def test_transient_refresh_failure_network():
    from rag_agent.monday_auth import _is_permanent_refresh_failure
    assert _is_permanent_refresh_failure() is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest rag_agent/test_monday_tool_capping.py -k refresh_failure -v`
Expected: FAIL — `ImportError: cannot import name '_is_permanent_refresh_failure'`.

- [ ] **Step 3a: Add the helper** — insert in `rag_agent/monday_auth.py` immediately above `_refresh_access_token_for_user` (before `:447`):

```python
# OAuth token-endpoint errors that mean the refresh token itself is permanently
# bad (user must reconnect). Distinct from transient 5xx / network blips.
_PERMANENT_OAUTH_ERRORS = frozenset({
    "invalid_grant",
    "invalid_client",
    "unauthorized_client",
    "unsupported_grant_type",
    "invalid_scope",
})


def _is_permanent_refresh_failure(*, http_code: int | None = None, error: str | None = None) -> bool:
    """True when a refresh failure means the grant is dead (revoke + reconnect).

    A 4xx from the token endpoint, or a structured OAuth error like
    ``invalid_grant``, is permanent. 5xx / network errors are transient.
    """
    if isinstance(http_code, int) and 400 <= http_code < 500:
        return True
    if str(error or "").strip().lower() in _PERMANENT_OAUTH_ERRORS:
        return True
    return False
```

- [ ] **Step 3b: Wire revoke into the HTTPError branch** — in `_refresh_access_token_for_user`, replace the existing `except urllib_error.HTTPError as exc:` block (`:493-495`) with:

```python
            except urllib_error.HTTPError as exc:
                code = getattr(exc, "code", None)
                logger.warning("Monday token refresh failed: HTTP %s", code if code is not None else "?")
                if _is_permanent_refresh_failure(http_code=code):
                    conn.revoked_at = _utcnow()
                    conn.updated_at = _utcnow()
                    db.commit()
                return ""
```

- [ ] **Step 3c: Wire revoke into the no-access-token branch** — replace the block at `:503-507` (`new_access = ...` through `return ""`) with:

```python
            new_access = str(token_response.get("access_token") or "").strip()
            if not new_access:
                err = str(token_response.get("error") or "no access_token in response")
                logger.warning("Monday token refresh failed: %s", err)
                if _is_permanent_refresh_failure(error=token_response.get("error")):
                    conn.revoked_at = _utcnow()
                    conn.updated_at = _utcnow()
                    db.commit()
                return ""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest rag_agent/test_monday_tool_capping.py -k refresh_failure -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add rag_agent/monday_auth.py rag_agent/test_monday_tool_capping.py
git commit -m "Revoke Monday connection on rejected refresh grant, not on blips"
```

---

## Task 6: Friendly error on OAuth callback token exchange

**Files:**
- Modify: `rag_agent/monday_auth.py:356-366` (the callback `_post_form` call)

The callback's `_post_form` (`:356`) is not wrapped, so a Monday 4xx raises `HTTPError` → a raw 500 to the user. Map it to a clean `RuntimeError` (the API turns `RuntimeError` into a user-facing message).

- [ ] **Step 1: Implement** — replace the `token_response = _post_form( ... )` call in `complete_monday_oauth_callback` (`:356-366`) with:

```python
        try:
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
        except urllib_error.HTTPError as exc:
            raise RuntimeError(
                f"Monday rejected the authorization (HTTP {getattr(exc, 'code', '?')}). "
                "Please try connecting Monday again."
            ) from exc
        except (urllib_error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(
                "Could not reach Monday to complete authorization. Please try again."
            ) from exc
```

- [ ] **Step 2: Verify import** — confirm `urllib_error` is imported (it is, at `:19`: `from urllib import error as urllib_error, ...`).

- [ ] **Step 3: Manual verification** — `python -c "import rag_agent.monday_auth"` must import cleanly (no syntax error).

Run: `python -c "import rag_agent.monday_auth; print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add rag_agent/monday_auth.py
git commit -m "Return friendly error when Monday OAuth callback exchange fails"
```

---

## Task 7: Coroutine-only session tools (`for_session` flag)

**Files:**
- Modify: `rag_agent/monday_auth.py:739-846` (`_ensure_sync_callable_tools`), `:1044-1058` (`_prepare_monday_tools`), `:1138` (call in `monday_session_tools`)
- Test: `rag_agent/test_monday_tool_capping.py`

- [ ] **Step 1: Write the failing tests** (append above `_run_all`)

```python
class _FakeAsyncTool:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = "desc"
        self.args_schema = None
        self.func = None

        async def _coro(**kwargs):
            return "ok"

        self.coroutine = _coro


def test_session_tools_are_coroutine_only():
    from rag_agent.monday_auth import _ensure_sync_callable_tools
    wrapped = _ensure_sync_callable_tools([_FakeAsyncTool("get_board_info")], for_session=True)
    assert len(wrapped) == 1
    assert wrapped[0].coroutine is not None
    assert wrapped[0].func is None


def test_per_call_tools_keep_sync_func():
    from rag_agent.monday_auth import _ensure_sync_callable_tools
    wrapped = _ensure_sync_callable_tools([_FakeAsyncTool("get_board_info")], for_session=False)
    assert len(wrapped) == 1
    assert wrapped[0].func is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest rag_agent/test_monday_tool_capping.py -k "session_tools or sync_func" -v`
Expected: FAIL — `_ensure_sync_callable_tools()` got an unexpected keyword argument `for_session`.

- [ ] **Step 3a: Add `for_session` to `_ensure_sync_callable_tools`** — change the signature at `:739`:

```python
def _ensure_sync_callable_tools(tools: list, *, for_session: bool = False) -> list:
```

Then, in the coroutine-only branch, leave the `def _make_sync_runner(...)`, `async def _safe_coroutine_runner(...)`, and `def _make_safe_coroutine(...)` helper definitions (`:796-835`) exactly as they are, and replace ONLY the final `wrapped.append(StructuredTool.from_function(... func=_make_sync_runner(...) ...))` block (`:837-845`) with the following (the `if for_session` short-circuit must come AFTER those helper defs because it calls `_make_safe_coroutine`):

```python
        if for_session:
            # Session-bound tools share ONE loop-bound ClientSession for the turn.
            # Build coroutine-only so they can never be driven from a foreign loop
            # (a sync .invoke() would otherwise bridge onto a background loop and
            # terminate the session). A stray sync call now fails loudly instead.
            wrapped.append(
                StructuredTool.from_function(
                    name=name or "monday_tool",
                    description=description or f"MCP tool: {name or 'monday_tool'}",
                    args_schema=args_schema,
                    coroutine=_make_safe_coroutine(coroutine, name or "monday_tool", args_schema),
                )
            )
            continue

        wrapped.append(
            StructuredTool.from_function(
                name=name or "monday_tool",
                description=description or f"MCP tool: {name or 'monday_tool'}",
                args_schema=args_schema,
                func=_make_sync_runner(tool, name or "monday_tool", args_schema),
                coroutine=_make_safe_coroutine(coroutine, name or "monday_tool", args_schema) if callable(coroutine) else None,
            )
        )
```

(The `_make_sync_runner`, `_safe_coroutine_runner`, and `_make_safe_coroutine` helper definitions above this stay exactly as they are.)

- [ ] **Step 3b: Thread the flag through `_prepare_monday_tools`** — change `:1044` signature and `:1050` call:

```python
def _prepare_monday_tools(raw_tools: list, *, for_session: bool = False) -> list:
```
and the first line of its body:
```python
    tools = _ensure_sync_callable_tools(raw_tools, for_session=for_session)
```

- [ ] **Step 3c: Pass `for_session=True` from the session loader** — in `monday_session_tools`, change `:1138`:

```python
        tools = _prepare_monday_tools(raw_tools, for_session=True)
```

(`get_monday_mcp_tools_for_user` at `:1088` keeps the default `_prepare_monday_tools(raw_tools)` → `for_session=False`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest rag_agent/test_monday_tool_capping.py -k "session_tools or sync_func" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add rag_agent/monday_auth.py rag_agent/test_monday_tool_capping.py
git commit -m "Build Monday session tools coroutine-only to protect loop affinity"
```

---

## Task 8: Auth-expired reconnect guard

**Files:**
- Modify: `rag_agent/monday_auth.py:96-137` (`_format_tool_error_for_model`)
- Test: `rag_agent/test_monday_tool_capping.py`

- [ ] **Step 1: Write the failing test** (append above `_run_all`)

```python
def test_auth_expired_error_returns_reconnect_hint():
    import json
    from rag_agent.monday_auth import _format_tool_error_for_model
    out = _format_tool_error_for_model(
        "get_board_info", RuntimeError("HTTP 401 Unauthorized: the token has expired")
    )
    data = json.loads(out)
    assert data["ok"] is False
    assert "reconnect" in data["message"].lower()


def test_column_error_still_returns_board_info_hint():
    import json
    from rag_agent.monday_auth import _format_tool_error_for_model
    out = _format_tool_error_for_model("get_board_items_page", RuntimeError("Column not found: status"))
    data = json.loads(out)
    assert "get_board_info" in data["message"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest rag_agent/test_monday_tool_capping.py -k "auth_expired or column_error" -v`
Expected: `test_auth_expired_error_returns_reconnect_hint` FAILS (no reconnect hint yet); the column test should pass already (guards no regression).

- [ ] **Step 3: Add the auth-expired branch** — in `_format_tool_error_for_model`, insert this as the FIRST condition, immediately after `hint = ""` (`:106`) and before `if "column not found" in lower ...`:

```python
    auth_expired_markers = (
        "token has expired",
        "token expired",
        "expired access token",
        "invalid token",
        "unauthenticated",
        "authentication failed",
        "http 401",
        "401 unauthorized",
    )
    if any(m in lower for m in auth_expired_markers):
        hint = (
            "Your Monday session expired or is invalid. Tell the user to reconnect "
            "Monday in Settings; do not retry this tool until they reconnect."
        )
    elif "column not found" in lower or "missing_column" in lower:
```

(Change the existing `if "column not found"...` to `elif "column not found"...` so it chains off the new branch. The remaining `elif` branches stay as-is.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest rag_agent/test_monday_tool_capping.py -k "auth_expired or column_error" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add rag_agent/monday_auth.py rag_agent/test_monday_tool_capping.py
git commit -m "Add Monday auth-expired reconnect guard for tool errors"
```

---

## Task 9: Full suite + import sanity + wrap-up

**Files:** none (verification)

- [ ] **Step 1: Run the whole Monday test file both ways**

Run: `pytest rag_agent/test_monday_tool_capping.py -v`
Expected: PASS (all original + ~16 new tests).

Run: `python -m rag_agent.test_monday_tool_capping`
Expected: prints `N/N passed`, exit 0.

- [ ] **Step 2: Import sanity for the touched modules**

Run: `python -c "import rag_agent.config, rag_agent.monday_auth; print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Confirm the diff is scoped** to `config.py`, `monday_auth.py`, `test_monday_tool_capping.py` (+ maybe `system_prompt.yaml`).

Run: `git diff --stat main...HEAD`
Expected: only those files.

- [ ] **Step 4: Final note** — do NOT merge or push yet; report completion and the new env knobs (`RAG_MONDAY_MCP_TOOL_ALLOWLIST` default behavior, `RAG_MONDAY_TOKEN_REFRESH_LEEWAY_SECONDS`, `RAG_MONDAY_TOKEN_MAX_AGE_SECONDS`) for the owner to review before integrating with the deployment.

---

## Self-Review notes

- **Spec coverage:** Fix 1 → Tasks 1-3; Fix 2 → Tasks 4-6 (+ cache-key kept token-scoped: a comment-only note, folded into Task 5's file already documents token-scoping — no code change, intentionally); Fix 3 → Task 7; Fix 4 → leeway in Task 1/4, auth-expired guard in Task 8. Tests → distributed per task + Task 9.
- **Cache-key:** deliberately unchanged (documented in spec §Fix 2). If desired, add a one-line comment at `monday_auth.py:1066` during Task 5; not required.
- **Type/name consistency:** `_TOKEN_REFRESH_LEEWAY_SECONDS`, `_TOKEN_MAX_AGE_SECONDS`, `_is_permanent_refresh_failure`, `_PERMANENT_OAUTH_ERRORS`, `for_session`, `_prepare_monday_tools(..., for_session=...)`, `_ensure_sync_callable_tools(..., for_session=...)` used consistently across tasks and tests.
