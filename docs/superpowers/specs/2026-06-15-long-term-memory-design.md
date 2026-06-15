# Long-Term Cross-Conversation Memory — Design

- **Date:** 2026-06-15
- **Status:** Approved (brainstorming complete; ready for implementation plan)
- **Author:** armann@orlanda.info (with Claude Code)
- **Component:** `platform` app (Orlanda onboarding/knowledge bot)

## 1. Summary

Today the agent has only per-thread conversation memory (LangGraph Postgres
checkpointer, keyed `username:conversation_id`). Threads are isolated: a "new chat"
is a clean slate, and nothing learned in one conversation carries to another.

This feature adds **long-term, per-user memory that persists across all of a user's
conversations** — in the style of ChatGPT/Claude memory. The agent itself decides,
mid-conversation, what is worth remembering and writes it via a tool; the user (and
admins) can view, add, edit, and delete those memories.

## 2. Goals

- The agent remembers durable, useful information about a user across conversations.
- The agent decides in real time when to save/update/delete a memory (visible in chat).
- Memory covers two kinds of content:
  1. **Facts & preferences** — role, team, responsibilities, stated preferences
     (e.g. answer language, format, detail level).
  2. **Learned task recipes** — when a request took several turns to get right, the
     agent saves the working approach **and the result the user actually wanted**, so
     the next equivalent request succeeds on the first try. (Primary motivating case:
     the monday.com tool needs very specific `items_page` filter formats; users
     currently iterate to find them. See `memory/monday-agent-filter-format`.)
- Users can manage their own memories and toggle the feature off for themselves.
- Admins have full control (view/edit/delete any user's memories); all admin writes
  are audited.
- Ops can disable the whole feature platform-wide without a redeploy.

## 3. Non-goals (YAGNI)

- **No semantic/embedding retrieval in v1.** Memories are injected wholesale (see §7).
  The schema is built upgrade-ready (a nullable `embedding` column) so this can be
  added later without a migration.
- **No background/async extraction pass in v1.** Saving is in-chat only. A background
  "catch what was missed" pass can be added later without changing storage or UI.
- **No cross-user or team-shared memory.** Memory is strictly per-user. The shared,
  global knowledge stays the RAG knowledge base (unchanged).
- **No storage of sensitive HR/compensation/health data or secrets** — forbidden by
  prompt guidance (§6).

## 4. Locked decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Write trigger | In-chat tool the model calls (faithful to ChatGPT/Claude; reuses `extra_tools`). |
| Storage/retrieval | Approach A: plain-text list, injected wholesale; built upgrade-ready for semantic retrieval. |
| Privacy/governance | Per-user memories; **admin full control**, audited. |
| Memory scope | Work/user facts + preferences + learned task recipes. |
| Keying | `users.id` (UUID FK, `ON DELETE CASCADE`). |
| Save visibility | Agent **announces** saves/updates/deletes in chat (not silent). |
| Kill-switches | Per-user toggle **and** a global env kill-switch. |

## 5. Data model

### 5.1 New table `user_memories`

SQLAlchemy model in `rag_agent/db/models.py`, following existing style.

| Column | Type | Notes |
|---|---|---|
| `id` | `UUID` PK, default `uuid4` | |
| `user_id` | `UUID` FK → `users.id` `ON DELETE CASCADE`, indexed | memories die with the user |
| `content` | `Text`, not null | the memory, plain language |
| `category` | `String(16)`, not null | `fact` \| `preference` \| `task_recipe` |
| `source` | `String(16)`, not null, default `agent` | `agent` \| `user` \| `admin` |
| `source_thread_id` | `String(512)`, nullable | conversation it originated from |
| `embedding` | `Vector(1536)`, **nullable** | unused in v1; upgrade hook for semantic retrieval |
| `created_at` | `timestamptz`, server_default `now()` | |
| `updated_at` | `timestamptz`, server_default `now()`, `onupdate now()` | |

Index on `user_id` (the only hot query is "all memories for a user").

### 5.2 New column on `users`

- `memory_enabled` `Boolean`, not null, default `true` — per-user on/off toggle.

### 5.3 Migration

One new Alembic revision in `alembic/versions/` creating the table + column. Set
`down_revision` from `alembic history` (filename numbers are not apply order — per
CLAUDE.md). The `users.memory_enabled` column is added with a server default so the
migration is safe on existing rows.

## 6. Write path (memory creation)

### 6.1 Tool factory

`rag_agent/memory_tools.py` — `get_memory_tools_for_user(user_id)` returns three
**sync** LangChain tools bound to that user (mirrors `aget_monday_tools_for_user` in
`monday_tools.py`, but sync so it runs on **both** chat paths). DB access uses the
existing sync SQLAlchemy session.

- **`save_memory(content: str, category: str) -> str`**
  - Enforces cap `RAG_MAX_USER_MEMORIES` (default 100). If full, returns a "memory
    full" message so the model can ask the user to prune (does not silently evict).
  - Duplicate guard: if near-identical `content` already exists for the user, update
    that row instead of inserting (v1 = case-insensitive exact/normalized match; not
    semantic).
  - Returns a short confirmation the model relays to the user.
- **`update_memory(memory_id: str, new_content: str) -> str`** — edits an existing
  row owned by the user. Rejects ids not belonging to the user.
- **`delete_memory(memory_id: str) -> str`** — deletes a row owned by the user.

All three resolve `memory_id` against the `[mem_xxxx]` ids shown in the injected block
(§7). Ownership is always enforced (`WHERE user_id = :uid`).

### 6.2 Prompt guidance (`rag_agent/system_prompt.yaml`)

Add a "Long-term memory" section instructing the model:

- **Save when:** the user states a durable fact or preference, **or** a multi-turn
  effort finally produces a result the user confirms is correct — then save the
  working approach + the desired result (task recipe).
- **Do not save:** sensitive HR/compensation/health data, secrets/tokens/credentials,
  or one-off trivia.
- **Avoid duplicates:** check the already-injected memory list first; prefer
  `update_memory` over creating a near-duplicate.
- **Be concise** and write each memory as a self-contained statement.
- **Be transparent:** briefly tell the user when you save/update/delete a memory.

## 7. Read path (injection)

`build_memory_suffix(user_id) -> str | None` renders the user's memories into a block:

```
## What you remember about this user
[mem_3f9a] (preference) Answer in Russian, concise.
[mem_7c21] (task_recipe) To get their assigned monday items: items_page filter
            person-<id> on board "Sales Pipeline" -> returns the "My Deals" view.
```

- The `[mem_xxxx]` handle is a short, stable identifier **derived from the row `id`**
  (e.g. the first 8 hex chars of the UUID) and is unique within a single user's memory
  set. `update_memory`/`delete_memory` accept this handle (or the full UUID) and
  resolve it back to the row, always scoped `WHERE user_id = :uid`. Because all
  memories are already in context, no separate "list memories" tool is needed.
- **Guards:** count cap `RAG_MAX_USER_MEMORIES` and token budget
  `RAG_MEMORY_INJECT_TOKEN_BUDGET` (default ~1500). If exceeded, most-recently-updated
  memories win.
- Returns `None` when: global kill-switch off, user's `memory_enabled` is false, or the
  user has no memories.

### 7.1 Integration with `build_agent` (the only call-site changes)

`build_agent` already accepts `extra_tools` and `system_prompt_suffix`. Two call sites
in `rag_agent/api.py`:

- **Async `/chat/stream`** (~`api.py:1878`): currently passes
  `extra_tools=monday_tools` and `system_prompt_suffix=monday_system_prompt`. Add the
  memory tools to `extra_tools` and **compose** the suffix: memory block first, then
  the existing monday orientation (don't replace it).
- **Sync `/chat`** (~`api.py:1647`): currently `extra_tools=[]`, no suffix. Add memory
  tools + memory suffix. (Memory tools are sync, so unlike the async-only monday tools
  they work here.)

A small `_compose_system_prompt_suffix(*blocks)` helper joins non-empty blocks.

### 7.2 Upgrade seam (Approach B/C, future)

To graduate to semantic retrieval: start writing `embedding` on save and swap the
"inject all" query for an embedding top-k query. No schema migration required.

## 8. API surface (`rag_agent/api.py`)

### 8.1 User endpoints (own memories, existing bearer auth)

- `GET /memories` — list own memories.
- `POST /memories` — add (`source=user`).
- `PATCH /memories/{id}` — edit own memory.
- `DELETE /memories/{id}` — delete own memory.
- `DELETE /memories` — clear all own memories.
- `GET /me/memory-settings` / `PUT /me/memory-settings` — read/set `memory_enabled`.

### 8.2 Admin endpoints (full control, audited)

- `GET /admin/users/{username}/memories` — view any user's memories.
- `POST` / `PATCH /{id}` / `DELETE /{id}` equivalents under
  `/admin/users/{username}/memories`.
- Every admin write records an `AdminAuditLog` entry (action e.g.
  `memory.admin_delete`, target = username, details = memory id/content), matching the
  existing admin pattern.

Endpoints that are also SPA routes must respect the existing vite `bypass` convention
(§ frontend) — these are API-only, so no SPA collision.

## 9. Frontend (`rag_agent/frontend/`, React/Vite + shadcn)

- **Memory settings page** (reached from the user menu): list with category badges,
  inline edit/delete, "add memory," "clear all," and the enable/disable toggle. Calls
  the §8.1 endpoints. Matches the current Swiss/enterprise-neutral theme.
- **Admin per-user memory panel** in the existing admin area: view/edit/delete a
  selected user's memories (§8.2).
- **Live feedback:** `/chat/stream` already emits SSE step events; add a lightweight
  `memory` event emitted when a memory tool runs, so the chat UI shows a "Memory
  updated" toast and refreshes the list. (Core deliverable = the settings page; the
  toast reuses the same data and is a nice-to-have.)
- New routes added to `vite.config.ts` proxy/bypass lists as needed.

## 10. Configuration (`rag_agent/config.py`)

All clamped to sane ranges, per the config conventions.

| Env var | Default | Meaning |
|---|---|---|
| `RAG_MEMORY_ENABLED` | `true` | Global kill-switch. When false, no injection, tools no-op/absent, settings reflect disabled. |
| `RAG_MAX_USER_MEMORIES` | `100` | Per-user cap; `save_memory` refuses beyond this. |
| `RAG_MEMORY_INJECT_TOKEN_BUDGET` | `1500` | Token budget for the injected block. |

## 11. Ops / cluster considerations

- `user_memories` lives in **Postgres**, so it **replicates** across the
  `pg_auto_failover` cluster automatically — no rsync needed (unlike `knowledge_base/`
  and `data/`). This is a deliberate reason to store memory in PG, not on disk.
- No new background process, scheduler, or cron — nothing that needs primary-only
  gating. Memory writes happen inline in the request that the agent is already serving.
- The global kill-switch is an env var, so disabling requires an env change; document
  that flipping it to `false` is the fast mitigation if memory misbehaves. (A redeploy
  re-reads env; if a no-redeploy flip is later wanted, move the flag to
  `runtime_settings.json` like the active-model setting — out of scope for v1.)

## 12. Error handling

- Memory tool failures are **non-fatal** to the chat (consistent with the existing
  "monday tool errors are non-fatal" decision): a DB error in `save_memory` returns an
  error string to the model, the chat continues, nothing is persisted.
- `build_memory_suffix` failures degrade gracefully to `None` (chat proceeds without
  memory) and are logged.
- Ownership checks on every user/agent memory mutation prevent cross-user access via a
  guessed id.
- Cap and token-budget guards prevent prompt bloat / unbounded growth.

## 13. Testing

- **Backend** (no Python suite today — at minimum the import smoke test
  `python -c "import rag_agent.api"`, plus targeted checks):
  - `save_memory` enforces cap and dedup; `update`/`delete` enforce ownership.
  - `build_memory_suffix` respects global flag, per-user toggle, count cap, token
    budget, and empty state (returns `None`).
  - Migration applies cleanly (`alembic upgrade head`) and `users.memory_enabled`
    backfills on existing rows.
  - Admin endpoints write `AdminAuditLog` rows.
- **Frontend** (`npm run test`, vitest): memory settings page renders list, add/edit/
  delete call the right endpoints, toggle flips `memory_enabled`.

## 14. Future work (explicitly deferred)

- Semantic retrieval (Approach B/C) once per-user memory volume justifies it.
- Background extraction pass for memories the model didn't save in-band.
- Team-shared memory, if ever desired (significant privacy redesign).
- Moving the global kill-switch to `runtime_settings.json` for no-redeploy toggling.
