# Production readiness: RAG agent

This checklist helps you deploy the RAG agent on a **website** or **Slack**.

---

## 1. Environment and config

- [ ] **OPENAI_API_KEY** set in the server environment (or `.env`), never in code (used by current RAG embeddings).
- [ ] **OpenAI-only runtime:** set `RAG_AGENT_MODEL` to an OpenAI model (recommended: `openai:gpt-4o-mini`).
- [ ] **CHECKPOINT_DB** set to a file path for persistent chat history (e.g. `./data/checkpoints.db`).  
  Without it, history is in-memory and lost on restart.
- [ ] Optional: **RAG_AGENT_MODEL**, **RAG_AGENT_MAX_TOKENS**, **RAG_AGENT_API_PORT** (see `config.py`).
- [ ] **PostgreSQL (required):** **`DATABASE_URL`** is mandatory for the API — accounts and sessions live only in **`users`** and **`auth_sessions`**. Run **`python -m alembic upgrade head`** (includes `004_user_password_lifecycle`). One-time migration from old `data/users.json`: **`python -m rag_agent.import_json_users`**, then remove or archive that file. Registration passwords: **`RAG_MIN_PASSWORD_LENGTH`** (default 12), letter + digit; **`RAG_SESSION_EXPIRY_DAYS`** (default 7).
- [ ] **Employee onboarding flow:** create users via **`POST /admin/users/provision`** (returns random temporary password once). First login is blocked from app routes until user completes **`POST /auth/password/change`**.
- [ ] **Backups (operational):** schedule **`pg_dump`** (or your host’s automated backups) for the auth database; include **`RAG_AGENT_SECRET_KEY`** and **`DATABASE_URL`** credentials in your secrets backup — not in git.

---

## 2. Running as a web service (website or backend for Slack)

- [ ] Install: `pip install fastapi uvicorn`
- [ ] Create checkpoint dir: `mkdir -p data` and set `CHECKPOINT_DB=./data/checkpoints.db` (or absolute path).
- [ ] Run API:  
  `CHECKPOINT_DB=./data/checkpoints.db python -m rag_agent.api`  
  Or: `uvicorn rag_agent.api:app --host 0.0.0.0 --port 8000`
- [ ] Put behind a **reverse proxy** (e.g. Nginx, Caddy) with HTTPS.
- [ ] Add **rate limiting** and **auth** (API key or OAuth) in the proxy or in FastAPI middleware.

**Endpoints:**

- `GET /health` — use for load balancer / health checks.
- `POST /chat` — body: `{"message": "user text", "thread_id": "unique-id"}`; response: `{"response": "...", "sources": [{"file": "...", "page": ...}]}`.  
  Use `thread_id` = user id or Slack channel id so each conversation keeps its history.

---

## 3. Slack

- [ ] Create a Slack app and bot token.
- [ ] Your backend (website or a small Slack bot service) receives Slack events or slash commands.
- [ ] For each user message, call your API:  
  `POST /chat` with `message=<user text>` and `thread_id=<channel_id or user_id>`.
- [ ] Post the API `response` (and optionally `sources`) back to Slack.

So: **Slack → your backend → RAG API (`/chat`) → response back to Slack.** The RAG agent stays a stateless API; Slack is just a client.

---

## 4. Reliability

- [ ] **Index on deploy**: run `python -m rag_agent.indexing` after adding/updating PDFs (e.g. in CI or a deploy script).
- [ ] **Error handling**: RAG tool and API already return errors without crashing; add **logging** (e.g. to a file or log aggregator) and optional **retries** for OpenAI.
- [ ] **Timeouts**: model timeout is set in config; ensure your reverse proxy timeouts are higher than the agent’s (e.g. 60s).

---

## 5. Security

- [ ] Do not expose the API publicly without **authentication** (API key, JWT, or OAuth).
- [ ] **Validate input**: message length is limited in the API (max 10000 chars); add any extra rules you need.
- [ ] Keep **knowledge_base** and **index_store** out of web-accessible directories.

---

## 6. Optional improvements

- **Structured logging**: log each request (thread_id, message length, success/error) for debugging and analytics.
- **Metrics**: count requests, latency, errors (e.g. Prometheus + Grafana).
- **Docker**: image that runs `uvicorn rag_agent.api:app` and sets `CHECKPOINT_DB` and `OPENAI_API_KEY` via env.

---

## Quick start (production-like)

```bash
export OPENAI_API_KEY=sk-...
export CHECKPOINT_DB=./data/checkpoints.db
mkdir -p data
python -m rag_agent.indexing   # if you added/updated PDFs
python -m rag_agent.api        # or: uvicorn rag_agent.api:app --host 0.0.0.0 --port 8000
# GET http://localhost:8000/health
# POST http://localhost:8000/chat  Body: {"message": "Что в базе знаний?", "thread_id": "user-1"}
```
