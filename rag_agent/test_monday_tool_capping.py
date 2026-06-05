"""Regression tests for Monday MCP tool capping.

These guard the bug where the agent could not find anything / found wrong things
in Monday: the old caps stripped the toolset down to ~6 mostly-write tools,
dropping every discovery/read tool (search, get_board_info, list_workspaces,
all_monday_api, get_column_type_info, ...). The agent therefore could not search
boards or resolve column ids before filtering.

Run directly:  python -m rag_agent.test_monday_tool_capping
Or with pytest: pytest rag_agent/test_monday_tool_capping.py
"""
from __future__ import annotations

from rag_agent.monday_auth import (
    _MONDAY_CORE_WRITE_TOOLS,
    _MONDAY_ESSENTIAL_TOOLS,
    _MONDAY_READ_TOOLS,
    _MONDAY_TOOL_DENYLIST,
    _cap_monday_tools,
    _cap_optional_params,
    _count_optional_props_in_schema,
)


class _FakeTool:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSchema:
    def __init__(self, schema: dict) -> None:
        self._schema = schema

    def model_json_schema(self) -> dict:
        return self._schema


class _FakeToolWithSchema:
    def __init__(self, name: str, schema: dict | None = None) -> None:
        self.name = name
        self.args_schema = _FakeSchema(schema) if schema is not None else None


# Representative current monday.com hosted-MCP tool surface (~36 tools).
_REAL_MONDAY_TOOLNAMES = [
    # reads / discovery
    "get_user_context", "list_workspaces", "workspace_info", "search",
    "get_board_info", "get_board_items_page", "get_column_type_info",
    "get_type_details", "get_graphql_schema", "list_users_and_teams",
    "get_updates", "get_board_activity", "board_insights", "read_docs",
    "all_monday_api", "get_full_board_data", "get_assets", "get_form",
    "get_sprint_summary",
    # writes (many of them — these used to crowd out the reads)
    "create_item", "change_item_column_values", "create_update", "create_group",
    "create_column", "create_board", "create_doc", "update_doc", "create_folder",
    "update_folder", "create_view", "update_view", "create_workspace",
    "update_workspace", "create_notification", "move_object", "create_dashboard",
    "create_form", "create_workflow", "publish_workflow", "create_automation",
]

# The discovery tools that the agent CANNOT work without. Their absence is the
# whole "cannot find anything / finds wrong things" bug.
_CRITICAL_DISCOVERY = {
    "get_user_context", "search", "get_board_info", "get_board_items_page",
    "list_workspaces", "all_monday_api", "get_graphql_schema",
    "get_column_type_info",
}


def _names(tools):
    return {t.name for t in tools}


def test_default_cap_keeps_all_essential_reads():
    tools = [_FakeTool(n) for n in _REAL_MONDAY_TOOLNAMES]
    kept = _cap_monday_tools(tools, 25)  # current default
    kept_names = _names(kept)
    assert len(kept) <= 25
    missing = set(_MONDAY_READ_TOOLS) - kept_names
    assert not missing, f"default cap dropped essential read tools: {sorted(missing)}"


def test_tight_cap_still_keeps_critical_discovery():
    # Even at the OLD default cap (18), the critical discovery tools must survive.
    tools = [_FakeTool(n) for n in _REAL_MONDAY_TOOLNAMES]
    kept_names = _names(_cap_monday_tools(tools, 18))
    missing = _CRITICAL_DISCOVERY - kept_names
    assert not missing, f"tight cap dropped critical discovery tools: {sorted(missing)}"


def test_reads_outrank_writes_under_pressure():
    # With a cap smaller than the read set, reads win over writes.
    tools = [_FakeTool(n) for n in _REAL_MONDAY_TOOLNAMES]
    kept_names = _names(_cap_monday_tools(tools, 10))
    # No write-only tool should survive while a core read is evicted.
    assert "get_board_info" in kept_names
    assert "search" in kept_names
    assert "get_board_items_page" in kept_names


def test_no_cap_when_under_limit_returns_all():
    tools = [_FakeTool(n) for n in _REAL_MONDAY_TOOLNAMES[:10]]
    assert len(_cap_monday_tools(tools, 25)) == 10


def test_core_writes_survive_default_cap():
    tools = [_FakeTool(n) for n in _REAL_MONDAY_TOOLNAMES]
    kept_names = _names(_cap_monday_tools(tools, 25))
    missing = set(_MONDAY_CORE_WRITE_TOOLS) - kept_names
    assert not missing, f"default cap dropped core write tools: {sorted(missing)}"


def test_optional_param_cap_never_evicts_essentials():
    # Re-enabling the optional-param budget (>0) must NOT drop essential tools,
    # even with a tiny budget and heavy non-essential schemas. This guards the
    # old footgun where get_board_items_page's ~15 optional params evicted
    # everything after it.
    heavy_schema = {
        "type": "object",
        "properties": {f"opt_{i}": {} for i in range(12)},  # 12 optional params
    }
    tools = []
    for n in _MONDAY_READ_TOOLS:  # essentials, no schema needed (always kept)
        tools.append(_FakeToolWithSchema(n))
    # heavy NON-essential tools that would blow a small budget
    tools.append(_FakeToolWithSchema("create_dashboard", heavy_schema))
    tools.append(_FakeToolWithSchema("publish_workflow", heavy_schema))

    kept_names = {t.name for t in _cap_optional_params(tools, 5)}
    missing = set(_MONDAY_READ_TOOLS) - kept_names
    assert not missing, f"optional-param cap evicted essentials: {sorted(missing)}"
    # The heavy non-essentials should be the ones trimmed.
    assert "create_dashboard" not in kept_names or "publish_workflow" not in kept_names


def test_ui_internal_tool_is_denylisted():
    assert "get_full_board_data" in _MONDAY_TOOL_DENYLIST


def test_heavy_tool_optional_param_count_is_large():
    # get_board_items_page is the tool whose ~15 optional params blew the old
    # optional-param budget. Confirm the counter still sees it as heavy so we
    # never silently re-introduce a budget that this single tool exhausts.
    get_board_items_page_schema = {
        "type": "object",
        "properties": {
            "boardId": {},
            "columnIds": {"type": "array", "items": {"type": "string"}},
            "cursor": {},
            "filters": {"type": "array", "items": {"type": "object",
                "properties": {"columnId": {}, "compareAttribute": {},
                               "compareValue": {}, "operator": {}},
                "required": ["columnId", "compareValue"]}},
            "filtersOperator": {}, "includeColumns": {}, "includeItemDescription": {},
            "includeSubItems": {}, "itemIds": {"type": "array", "items": {"type": "number"}},
            "limit": {},
            "orderBy": {"type": "array", "items": {"type": "object",
                "properties": {"columnId": {}, "direction": {}}, "required": ["columnId"]}},
            "searchTerm": {}, "subItemLimit": {},
        },
        "required": ["boardId"],
    }
    assert _count_optional_props_in_schema(get_board_items_page_schema) >= 12


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
        _os.environ["RAG_MONDAY_MCP_TOOL_ALLOWLIST"] = "ALL"
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


def test_rate_limited_refresh_is_transient():
    # HTTP 429 is in the 4xx range but is rate-limiting, NOT a dead grant.
    from rag_agent.monday_auth import _is_permanent_refresh_failure
    assert _is_permanent_refresh_failure(http_code=429) is False


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


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print()
    print(f"{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    raise SystemExit(1 if _run_all() else 0)
