"""
Direct Monday API tools (no separate MCP server required).
"""
from __future__ import annotations

import json
import time
import urllib.request
from typing import Any, Callable

from langchain.tools import tool

ToolEventCallback = Callable[[dict[str, Any]], None]

_MONDAY_GRAPHQL_URL = "https://api.monday.com/v2"


def _emit(on_event: ToolEventCallback | None, tool_name: str, status: str, message: str) -> None:
    if not on_event:
        return
    on_event(
        {
            "source": "monday",
            "tool_name": tool_name,
            "status": status,
            "message": message,
            "ts": int(time.time() * 1000),
        }
    )


def _monday_graphql(access_token: str, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {"query": query, "variables": variables or {}}
    req = urllib.request.Request(
        _MONDAY_GRAPHQL_URL,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": access_token,
            "Content-Type": "application/json",
            "API-Version": "2023-10",
        },
    )
    with urllib.request.urlopen(req, timeout=45) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("Invalid Monday API response.")
    if data.get("errors"):
        raise RuntimeError(str(data["errors"]))
    return data.get("data") or {}


def build_monday_tools(access_token: str, on_event: ToolEventCallback | None = None) -> list[Any]:
    if not access_token:
        return []

    @tool
    def monday_me() -> str:
        """Get current Monday user info for the connected account."""
        name = "monday_me"
        _emit(on_event, name, "start", "Reading Monday user profile")
        try:
            data = _monday_graphql(access_token, "query { me { id name email } }")
            _emit(on_event, name, "success", "Monday user profile loaded")
            return json.dumps(data, ensure_ascii=False)
        except Exception as e:
            _emit(on_event, name, "error", f"Monday profile request failed: {e!s}")
            return f"Ошибка Monday API: {e!s}"

    @tool
    def monday_list_boards(limit: int = 10) -> str:
        """List Monday boards (limit 1..50)."""
        name = "monday_list_boards"
        safe_limit = max(1, min(int(limit or 10), 50))
        _emit(on_event, name, "start", f"Loading Monday boards (limit={safe_limit})")
        try:
            data = _monday_graphql(
                access_token,
                "query ($limit: Int!) { boards(limit: $limit) { id name state board_kind } }",
                {"limit": safe_limit},
            )
            _emit(on_event, name, "success", "Monday boards loaded")
            return json.dumps(data, ensure_ascii=False)
        except Exception as e:
            _emit(on_event, name, "error", f"Monday boards request failed: {e!s}")
            return f"Ошибка Monday API: {e!s}"

    @tool
    def monday_list_board_items(board_id: int, limit: int = 25) -> str:
        """List items from a Monday board by board_id."""
        name = "monday_list_board_items"
        safe_limit = max(1, min(int(limit or 25), 100))
        _emit(on_event, name, "start", f"Loading items for board {board_id}")
        try:
            data = _monday_graphql(
                access_token,
                (
                    "query ($boardId: [ID!], $limit: Int!) { "
                    "boards(ids: $boardId) { id name items_page(limit: $limit) { cursor items { id name updated_at } } } "
                    "}"
                ),
                {"boardId": [str(board_id)], "limit": safe_limit},
            )
            _emit(on_event, name, "success", f"Loaded board items for {board_id}")
            return json.dumps(data, ensure_ascii=False)
        except Exception as e:
            _emit(on_event, name, "error", f"Monday items request failed: {e!s}")
            return f"Ошибка Monday API: {e!s}"

    @tool
    def monday_get_board_items_by_name(board_id: int, term: str, limit: int = 25) -> str:
        """Search board items by name term (case-insensitive on server side)."""
        name = "monday_get_board_items_by_name"
        safe_limit = max(1, min(int(limit or 25), 100))
        _emit(on_event, name, "start", f"Searching items in board {board_id} by '{term}'")
        try:
            data = _monday_graphql(
                access_token,
                (
                    "query ($boardId: ID!, $term: String!, $limit: Int!) { "
                    "items_page_by_column_values("
                    "board_id: $boardId, "
                    "columns: [{column_id: \"name\", column_values: [$term]}], "
                    "limit: $limit"
                    ") { cursor items { id name updated_at } } "
                    "}"
                ),
                {"boardId": int(board_id), "term": term, "limit": safe_limit},
            )
            _emit(on_event, name, "success", "Item search completed")
            return json.dumps(data, ensure_ascii=False)
        except Exception as e:
            _emit(on_event, name, "error", f"Item search failed: {e!s}")
            return f"Ошибка Monday API: {e!s}"

    @tool
    def monday_create_item(board_id: int, group_id: str, item_name: str) -> str:
        """Create a Monday item in a board/group."""
        name = "monday_create_item"
        _emit(on_event, name, "start", f"Creating Monday item in board {board_id}")
        try:
            data = _monday_graphql(
                access_token,
                (
                    "mutation ($boardId: ID!, $groupId: String!, $itemName: String!) { "
                    "create_item(board_id: $boardId, group_id: $groupId, item_name: $itemName) { id name } "
                    "}"
                ),
                {"boardId": str(board_id), "groupId": group_id, "itemName": item_name},
            )
            _emit(on_event, name, "success", "Monday item created")
            return json.dumps(data, ensure_ascii=False)
        except Exception as e:
            _emit(on_event, name, "error", f"Monday create item failed: {e!s}")
            return f"Ошибка Monday API: {e!s}"

    @tool
    def monday_get_board_schema(board_id: int) -> str:
        """Get board columns/groups metadata by board_id."""
        name = "monday_get_board_schema"
        _emit(on_event, name, "start", f"Loading board schema for {board_id}")
        try:
            data = _monday_graphql(
                access_token,
                (
                    "query ($boardId: [ID!]) { "
                    "boards(ids: $boardId) { "
                    "id name state board_kind "
                    "groups { id title } "
                    "columns { id title type settings_str } "
                    "} "
                    "}"
                ),
                {"boardId": [str(board_id)]},
            )
            _emit(on_event, name, "success", "Board schema loaded")
            return json.dumps(data, ensure_ascii=False)
        except Exception as e:
            _emit(on_event, name, "error", f"Board schema request failed: {e!s}")
            return f"Ошибка Monday API: {e!s}"

    @tool
    def monday_get_item(item_id: int) -> str:
        """Get one Monday item with column values by item_id."""
        name = "monday_get_item"
        _emit(on_event, name, "start", f"Loading item {item_id}")
        try:
            data = _monday_graphql(
                access_token,
                (
                    "query ($itemIds: [ID!]) { "
                    "items(ids: $itemIds) { "
                    "id name state "
                    "board { id name } "
                    "group { id title } "
                    "column_values { id text value type } "
                    "} "
                    "}"
                ),
                {"itemIds": [str(item_id)]},
            )
            _emit(on_event, name, "success", f"Item {item_id} loaded")
            return json.dumps(data, ensure_ascii=False)
        except Exception as e:
            _emit(on_event, name, "error", f"Item request failed: {e!s}")
            return f"Ошибка Monday API: {e!s}"

    @tool
    def monday_change_column_value(board_id: int, item_id: int, column_id: str, value_json: str) -> str:
        """Update one column value on an item. value_json must be a JSON object string."""
        name = "monday_change_column_value"
        _emit(on_event, name, "start", f"Updating column {column_id} for item {item_id}")
        try:
            parsed = json.loads(value_json or "{}")
            if not isinstance(parsed, dict):
                raise ValueError("value_json must be a JSON object")
            data = _monday_graphql(
                access_token,
                (
                    "mutation ($boardId: ID!, $itemId: ID!, $columnId: String!, $value: JSON!) { "
                    "change_column_value(board_id: $boardId, item_id: $itemId, column_id: $columnId, value: $value) { id } "
                    "}"
                ),
                {
                    "boardId": str(board_id),
                    "itemId": str(item_id),
                    "columnId": column_id,
                    "value": json.dumps(parsed, ensure_ascii=False),
                },
            )
            _emit(on_event, name, "success", "Column value updated")
            return json.dumps(data, ensure_ascii=False)
        except Exception as e:
            _emit(on_event, name, "error", f"Update column failed: {e!s}")
            return f"Ошибка Monday API: {e!s}"

    @tool
    def monday_change_item_column_values(board_id: int, item_id: int, column_values_json: str) -> str:
        """Update multiple item columns. column_values_json must be JSON object string."""
        name = "monday_change_item_column_values"
        _emit(on_event, name, "start", f"Updating multiple columns for item {item_id}")
        try:
            parsed = json.loads(column_values_json or "{}")
            if not isinstance(parsed, dict):
                raise ValueError("column_values_json must be a JSON object")
            data = _monday_graphql(
                access_token,
                (
                    "mutation ($boardId: ID!, $itemId: ID!, $columnValues: JSON!) { "
                    "change_multiple_column_values("
                    "board_id: $boardId, item_id: $itemId, column_values: $columnValues"
                    ") { id } "
                    "}"
                ),
                {
                    "boardId": str(board_id),
                    "itemId": str(item_id),
                    "columnValues": json.dumps(parsed, ensure_ascii=False),
                },
            )
            _emit(on_event, name, "success", "Multiple columns updated")
            return json.dumps(data, ensure_ascii=False)
        except Exception as e:
            _emit(on_event, name, "error", f"Batch update failed: {e!s}")
            return f"Ошибка Monday API: {e!s}"

    @tool
    def monday_move_item_to_group(item_id: int, group_id: str) -> str:
        """Move item to another group on the same board."""
        name = "monday_move_item_to_group"
        _emit(on_event, name, "start", f"Moving item {item_id} to group {group_id}")
        try:
            data = _monday_graphql(
                access_token,
                (
                    "mutation ($itemId: ID!, $groupId: String!) { "
                    "move_item_to_group(item_id: $itemId, group_id: $groupId) { id } "
                    "}"
                ),
                {"itemId": str(item_id), "groupId": group_id},
            )
            _emit(on_event, name, "success", "Item moved to group")
            return json.dumps(data, ensure_ascii=False)
        except Exception as e:
            _emit(on_event, name, "error", f"Move item failed: {e!s}")
            return f"Ошибка Monday API: {e!s}"

    @tool
    def monday_create_update(item_id: int, body: str) -> str:
        """Create update/comment for an item."""
        name = "monday_create_update"
        _emit(on_event, name, "start", f"Creating update for item {item_id}")
        try:
            data = _monday_graphql(
                access_token,
                "mutation ($itemId: ID!, $body: String!) { create_update(item_id: $itemId, body: $body) { id } }",
                {"itemId": str(item_id), "body": body},
            )
            _emit(on_event, name, "success", "Update created")
            return json.dumps(data, ensure_ascii=False)
        except Exception as e:
            _emit(on_event, name, "error", f"Create update failed: {e!s}")
            return f"Ошибка Monday API: {e!s}"

    @tool
    def monday_get_board_items_for_assignee(board_id: int, assignee_name: str, limit: int = 100) -> str:
        """Find board items assigned to a person by matching assignee name in people column values."""
        name = "monday_get_board_items_for_assignee"
        safe_limit = max(1, min(int(limit or 100), 500))
        _emit(on_event, name, "start", f"Finding tasks for '{assignee_name}' in board {board_id}")
        try:
            data = _monday_graphql(
                access_token,
                (
                    "query ($boardId: [ID!], $limit: Int!) { "
                    "boards(ids: $boardId) { "
                    "id name "
                    "items_page(limit: $limit) { "
                    "items { id name updated_at column_values { id type text value } } "
                    "} "
                    "} "
                    "}"
                ),
                {"boardId": [str(board_id)], "limit": safe_limit},
            )
            boards = (data or {}).get("boards") or []
            if not boards:
                _emit(on_event, name, "success", "Board not found")
                return json.dumps({"board_found": False, "items": []}, ensure_ascii=False)

            needle = (assignee_name or "").strip().lower()
            result_items: list[dict[str, Any]] = []
            for item in ((boards[0].get("items_page") or {}).get("items") or []):
                cols = item.get("column_values") or []
                matched = False
                for c in cols:
                    ctype = str(c.get("type") or "").lower()
                    text = str(c.get("text") or "").lower()
                    if ctype in ("people", "multiple-person", "personsandteams", "person") and needle and needle in text:
                        matched = True
                        break
                if matched:
                    result_items.append(
                        {
                            "id": item.get("id"),
                            "name": item.get("name"),
                            "updated_at": item.get("updated_at"),
                        }
                    )

            _emit(on_event, name, "success", f"Found {len(result_items)} tasks for {assignee_name}")
            return json.dumps(
                {
                    "board_id": boards[0].get("id"),
                    "board_name": boards[0].get("name"),
                    "assignee_name": assignee_name,
                    "count": len(result_items),
                    "items": result_items,
                },
                ensure_ascii=False,
            )
        except Exception as e:
            _emit(on_event, name, "error", f"Assignee task search failed: {e!s}")
            return f"Ошибка Monday API: {e!s}"

    @tool
    def monday_delete_item(item_id: int) -> str:
        """Delete item by item_id."""
        name = "monday_delete_item"
        _emit(on_event, name, "start", f"Deleting item {item_id}")
        try:
            data = _monday_graphql(
                access_token,
                "mutation ($itemId: ID!) { delete_item(item_id: $itemId) { id } }",
                {"itemId": str(item_id)},
            )
            _emit(on_event, name, "success", f"Item {item_id} deleted")
            return json.dumps(data, ensure_ascii=False)
        except Exception as e:
            _emit(on_event, name, "error", f"Delete item failed: {e!s}")
            return f"Ошибка Monday API: {e!s}"

    @tool
    def monday_graphql(query: str, variables_json: str = "{}") -> str:
        """Universal Monday GraphQL executor for any query/mutation. variables_json must be JSON object string."""
        name = "monday_graphql"
        _emit(on_event, name, "start", "Executing custom Monday GraphQL")
        try:
            variables = json.loads(variables_json or "{}")
            if not isinstance(variables, dict):
                raise ValueError("variables_json must be a JSON object")
            data = _monday_graphql(access_token, query, variables)
            _emit(on_event, name, "success", "Custom Monday GraphQL executed")
            return json.dumps(data, ensure_ascii=False)
        except Exception as e:
            _emit(on_event, name, "error", f"Custom GraphQL failed: {e!s}")
            return f"Ошибка Monday API: {e!s}"

    @tool
    def monday_get_graphql_schema() -> str:
        """Fetch Monday GraphQL schema summary (types + root operation names)."""
        name = "monday_get_graphql_schema"
        _emit(on_event, name, "start", "Loading GraphQL schema summary")
        try:
            data = _monday_graphql(
                access_token,
                (
                    "query { "
                    "__schema { "
                    "queryType { name } "
                    "mutationType { name } "
                    "types { name kind description } "
                    "} "
                    "}"
                ),
            )
            _emit(on_event, name, "success", "GraphQL schema summary loaded")
            return json.dumps(data, ensure_ascii=False)
        except Exception as e:
            _emit(on_event, name, "error", f"Schema request failed: {e!s}")
            return f"Ошибка Monday API: {e!s}"

    @tool
    def monday_get_type_details(type_name: str) -> str:
        """Fetch GraphQL type details by name."""
        name = "monday_get_type_details"
        _emit(on_event, name, "start", f"Loading GraphQL type details for {type_name}")
        try:
            data = _monday_graphql(
                access_token,
                (
                    "query ($typeName: String!) { "
                    "__type(name: $typeName) { "
                    "name kind description "
                    "fields { name description args { name description } type { name kind ofType { name kind } } } "
                    "inputFields { name description type { name kind ofType { name kind } } } "
                    "enumValues { name description } "
                    "} "
                    "}"
                ),
                {"typeName": type_name},
            )
            _emit(on_event, name, "success", f"Type details loaded for {type_name}")
            return json.dumps(data, ensure_ascii=False)
        except Exception as e:
            _emit(on_event, name, "error", f"Type details request failed: {e!s}")
            return f"Ошибка Monday API: {e!s}"

    @tool
    def monday_list_users_and_teams(limit: int = 100, search: str = "") -> str:
        """List users and teams with optional search term."""
        name = "monday_list_users_and_teams"
        safe_limit = max(1, min(int(limit or 100), 500))
        _emit(on_event, name, "start", "Loading users and teams")
        try:
            data = _monday_graphql(
                access_token,
                (
                    "query ($limit: Int!, $search: String!) { "
                    "users(limit: $limit, kind: all, name: $search) { id name email enabled is_guest is_admin } "
                    "teams { id name picture_url users { id name } } "
                    "}"
                ),
                {"limit": safe_limit, "search": search},
            )
            _emit(on_event, name, "success", "Users and teams loaded")
            return json.dumps(data, ensure_ascii=False)
        except Exception as e:
            _emit(on_event, name, "error", f"Users/teams request failed: {e!s}")
            return f"Ошибка Monday API: {e!s}"

    return [
        monday_me,
        monday_list_boards,
        monday_get_board_schema,
        monday_list_board_items,
        monday_get_board_items_by_name,
        monday_get_item,
        monday_create_item,
        monday_change_column_value,
        monday_change_item_column_values,
        monday_move_item_to_group,
        monday_create_update,
        monday_delete_item,
        monday_get_board_items_for_assignee,
        monday_get_graphql_schema,
        monday_get_type_details,
        monday_graphql,
    ]
