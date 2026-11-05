# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Orlanda's internal onboarding/knowledge bot: a FastAPI backend wrapping a LangGraph agent
(`langchain.agents.create_agent`) whose core tool is RAG over company PDFs/text (pgvector),
plus optional per-user long-term memory and monday.com tools. It also hosts an external
**client portal** ("cabinet") for invited customers. A React/Vite SPA (shadcn/ui) is served by
the same FastAPI process. Employee answers default to Russian; the client cabinet is EN/HE.

Everything backend lives under `rag_agent/`. The frontend is `rag_agent/frontend/`.

## Commands

Backend (run from project root; copy `.env.example` → `.env` for the required keys —
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, a non-default `RAG_AGENT_SECRET_KEY`). Note
`.env.example` deliberately omits `DATABASE_URL` because `docker-compose.yml` assembles it from
`DB_PASSWORD`; a **local, non-Docker** run must set `DATABASE_URL` yourself):

```bash
python -m alembic upgrade head        # apply DB migrations (also runs on container start)
python -m rag_agent.indexing          # (re)index knowledge_base/ PDFs into pgvector
python -m rag_agent.api               # start API + serve built frontend (default :8000)
python -m rag_agent.run               # one-off agent smoke test from the CLI
python -m rag_agent.eval_retrieval --dataset rag_agent/data/retrieval_eval.jsonl --k 8 --show-failures
```

Backend tests: there is currently no Python test suite. Sanity-check edits with an import smoke
test instead: `python -c "import rag_agent.api"`.

Frontend (run from `rag_agent/frontend/`):

```bash
npm install
npm run dev      # Vite dev server on :8080, proxies API calls to 127.0.0.1:8000
npm run build    # production build → dist/ (Dockerfile bakes this in)
npm run lint     # eslint
npm run test     # vitest (single run);  npm run test:watch to watch
npx vitest run src/test/MemorySettingsCard.test.tsx   # run a single test file
```

In dev you run **both**: `python -m rag_agent.api` (:8000) and `npm run dev` (:8080), and
use :8080. `vite.config.ts` proxies API prefixes (`/auth`, `/chat`, `/knowledge`, `/client`,
`/memories`, `/integrations`, `/admin/*`, ...) to the backend, with a `bypass` so HTML
navigations to dual SPA/API routes (`/chat`, `/auth`, `/client`, `/admin/logs`,
`/admin/invites`) still boot the React app instead of returning FastAPI's prod index.html.
**A new backend route prefix must also be added to this proxy list**, or it 404s in dev only.

## Architecture

**Agent construction (`rag_agent/agent.py`).** `build_agent(...)` assembles the tool list
(`retrieve_context` plus `extra_tools` the API passes per request — memory and monday tools),
applies the system prompt plus optional suffix blocks (memory, monday guidance), and wires the
checkpointer. The active chat model is switchable at runtime (`set_active_model`) and
**persists across restarts** via `rag_agent/data/runtime_settings.json`; env `RAG_AGENT_MODEL`
(default `anthropic:claude-sonnet-4-6`) is only the bootstrap default. Models are
provider-prefixed (`anthropic:`, `openai:`, ...).

**System prompt (`rag_agent/system_prompt.yaml`).** `system_prompt` (always) and
`monday_system_prompt` (appended only when the user has monday tools), YAML literal blocks
loaded once at import by `agent.py`. The main prompt enforces a strict anti-hallucination policy:
answer only from `retrieve_context` results, otherwise state that the knowledge base has nothing.

**Checkpointer (conversation memory).** `CHECKPOINT_BACKEND=postgres` (default) uses a pooled
`PostgresSaver` over a `psycopg_pool.ConnectionPool` with `check_connection` + TCP keepalives.
This is intentional: a single long-lived connection dies on any db-router/failover socket drop
and is never revalidated, breaking every later chat until restart. Don't revert to
`from_conn_string`. Sync savers are made async-compatible by `_attach_async_compat` (wraps sync
methods in `asyncio.to_thread`) so one instance serves both the sync `/chat` and async
`/chat/stream` paths.

**Two chat paths in `rag_agent/api.py`.** `POST /chat` (sync) and `POST /chat/stream` (async
SSE, used by the UI). Both build a per-request agent, run it, then persist/repair history. History is compacted when a thread exceeds
`RAG_MAX_HISTORY_MESSAGES` (summarize old turns, keep last N). There is logic to sanitize/repair
persisted messages so provider strict-mode (orphaned tool_use/tool_result blocks) doesn't 400.
The paths differ in tooling: monday MCP tools are async-only, so they are added **only** on
`/chat/stream`; memory tools are sync and are added on both.

**Long-term user memory (`user_memory.py`, `memory_tools.py`).** Separate from the per-thread
checkpointer: per-user facts in the `user_memories` table that persist across conversations.
The agent writes them via `save/update/delete_memory` tools; users and admins manage them via
`/memories` and `/admin/users/{u}/memories`. They're rendered wholesale into the system prompt
each turn, gated by both the global `RAG_MEMORY_ENABLED` kill-switch and per-user
`users.memory_enabled`. Tool errors are returned to the model as strings, never raised.

**monday.com (`monday_oauth.py`, `monday_tools.py`).** Each user connects their own monday
account via OAuth (`/integrations/monday/*`, callback `/auth/monday/callback`). The token is
stored encrypted (`crypto.py`), and the OAuth `state` is HMAC-signed with `SECRET_KEY` (no
state table). Tools come from monday's remote MCP, called with that user's token, and are
memoized briefly per token. Enabled only when the monday OAuth client id/secret are set.
Failures yield no tools / a non-fatal error string.

**Client portal (`client_portal.py`, frontend `pages/client/`, `ClientShell`).** External
customers register via invite links (`/admin/invites` → `/invites/{token}` →
`/auth/register-client`) and get `role='client'` (any e-mail domain). Their cabinet (tasks
table, assistant chat, progress, files, feedback) is **not** served by this app's agent: it
proxies over HTTP to a separate service, **orlanda-api** (`ORLANDA_API_BASE_URL`, header
`X-Internal-Key` = `ORLANDA_API_KEY`, which must equal orlanda-api's `CLIENT_PORTAL_API_KEY`).
Project access lives only in orlanda-api's `ProjectMember` table, never here. The portal is
off (503) when `ORLANDA_API_KEY` is unset.

**RAG retrieval (`rag_agent/rag_tool.py`, `indexing.py`).** Incremental indexing into Postgres
`pgvector` (per-PDF / per-item add/update/delete, not full reindex). Query pipeline is hybrid
(dense + BM25) with RRF fusion, optional MMR diversity, optional cross-encoder rerank, and
neighbor-page expansion — all tuned by `RAG_*` env vars (see `config.py`). Embeddings use OpenAI
(`text-embedding-3-small`, 1536-dim) regardless of the chat provider, so `OPENAI_API_KEY` is
always required.

**Auth (`rag_agent/auth.py`).** PostgreSQL-only (`users` + `auth_sessions`); no `users.json` at
runtime. Passwords are Argon2 (legacy SHA-256 rehashed on login). Session tokens are stored only
as `sha256(SECRET_KEY:token)` — so `RAG_AGENT_SECRET_KEY` must be stable, and must be **identical
across both cluster servers** or everyone is logged out after failover. Roles are
`admin` / `user` / `client`. Employee logins are restricted to `RAG_AGENT_ADMIN_USERNAMES` or
`RAG_ALLOWED_EMAIL_DOMAIN` (default `@orlanda.info`) emails, and clients come only from invites. New employees are provisioned via
`POST /admin/users/provision` (returns a one-time temp password, `must_change_password=true`).
When `SSO_COOKIE_DOMAIN` is set (e.g. `.n8norlanda.com`), login/password-change and `/auth/me`
also set a cross-subdomain cookie (`SSO_COOKIE_NAME`, default `orlanda_sso`) carrying the
session token, so sibling Orlanda apps can auto-login.

**Config (`rag_agent/config.py`) is the single source of env truth.** All numeric settings are
clamped to sane ranges there; read it before adding a new env var. `require_runtime_keys()`
validates provider key + `OPENAI_API_KEY` + `DATABASE_URL` + non-default `SECRET_KEY` at startup.
`warn_oauth_redirect_misconfig()` only logs (never raises) on a LAN/http `RAG_FRONTEND_BASE_URL`.

## Deployment (read before touching anything ops-related)

This repo deploys as the **`platform`** app on the Orlanda two-server `pg_auto_failover` cluster
(see `DEPLOY_ORLANDA.md`, also `rag_agent/DEPLOY.md` / `rag_agent/PRODUCTION.md`). Critical, non-obvious rules:

- **Pushing to `main` does NOT deploy.** The running app only updates after an SSH step on the
  server. There is no staging — "testing on the server" is production.
- Deploy/redeploy must happen on the **current primary** (`sudo -u postgres
  /usr/bin/pg_autoctl show state --pgdata /var/lib/postgresql/ha`); the standby keeps the
  container stopped via the role agent. App dir is `/opt/platform-agent`. Redeploy:
  `git fetch && git checkout <branch> && git pull && docker compose build && docker compose up -d`,
  then `curl -i 127.0.0.1:8001/healthz`. A stale-image symptom ("I deployed but nothing changed")
  usually means `up -d` reused the old image — use `docker compose build --no-cache`.
- App listens on **:8001** (roman owns :8000). DB is reached via the db-router at
  **127.0.0.1:6432** (HAProxy → current primary), never `:5432` or a server IP. `DATABASE_URL`
  is assembled in `docker-compose.yml` from `${DB_PASSWORD}`.
- **Disk does not replicate.** `knowledge_base/` (raw PDFs) and `data/` (chat_conversations.json
  titles) are bind-mounted and must be rsync'd to the standby after changes. PDF *embeddings*
  live in Postgres and do replicate.
- The Dockerfile is two-stage: Node builds the frontend, Python serves it. `CMD` runs `alembic
  upgrade head` then `python -m rag_agent.api`.

## Conventions

- Dependency versions in `requirements.txt` are pinned exactly on purpose (LangChain/LangGraph/
  Anthropic/Pydantic drive tool-schema + Anthropic wire format; unpinned prod was diverging
  from local). Don't loosen them casually.
- Migrations: Alembic in `alembic/versions/`. The chain is linear but **filename numbers are not
  always apply order** (a few early revisions are numbered out of sequence) — trust
  `down_revision` / `alembic history`, not the filename, before adding one.
- Health endpoints: `/healthz` is DB-free (Cloudflare LB health check); `/health` touches the DB.
