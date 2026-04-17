# Deploying the RAG Agent for Employee Testing

This app is a **FastAPI server** that uses **persistent local files** (users, chat log, knowledge base PDFs, FAISS index, optional SQLite checkpoints). To deploy so employees can test it and you can update it later, use a host that provides a **persistent filesystem** and can run a long-lived process.

---

## Why not Vercel?

**Vercel** is built for serverless: each request runs in a short-lived function. Limits that affect this app:

- **No persistent disk** — only `/tmp` is writable, and it is cleared between runs. Your users, chat log, knowledge items, uploaded PDFs, and FAISS index would be lost or never persist.
- **Short timeouts** on the free tier (e.g. 10s) — RAG + OpenAI often takes longer.
- **Stateless** — you’d need external services (database, blob storage, vector DB) and code changes to move off file-based storage.

So **Vercel is not a good fit** for this app as it is today. The options below keep your current design and work with minimal changes.

---

## Recommended: Railway (free credit, persistent disk)

[Railway](https://railway.app) gives you a small monthly free credit and **persistent disk**, so users, history, and knowledge base survive restarts and redeploys.

### 1. Prepare the repo

- Ensure the app runs from the **repository root** (so `rag_agent` and `knowledge_base` are at the root level).
- A **Procfile** at the repo root is optional; if present it should be:

  ```
  web: python -m rag_agent.api
  ```

  The app already reads the `PORT` environment variable, which Railway sets.

### 2. Create a Railway project

1. Sign up at [railway.app](https://railway.app) (GitHub login is fine).
2. **New Project** → **Deploy from GitHub repo**.
3. Select your `onboarding_bot_from_scratch` repository (or the one that contains `rag_agent`).
4. Railway will detect Python and use `requirements.txt` from the repo root. If your `requirements.txt` is in the repo root, you’re set. If it’s only under `rag_agent`, either move/copy it to the root or set the **Root Directory** in Railway to the folder that contains both `rag_agent` and `requirements.txt`.
5. **Start command** (if not using Procfile):  
   `python -m rag_agent.api`  
   Ensure the **working directory** is the repo root so `rag_agent` and `knowledge_base` resolve correctly.

### 3. Set environment variables

In the Railway service → **Variables**, add:

| Variable | Required | Example / note |
|----------|----------|------------------|
| `OPENAI_API_KEY` | Yes | Your OpenAI API key. |
| `RAG_AGENT_SECRET_KEY` | Yes | Random string for auth (e.g. `openssl rand -hex 32`). |
| `RAG_AGENT_ADMIN_USERNAMES` | Optional | Comma-separated logins that can open the admin panel (e.g. `admin,manager`). |
| `CHECKPOINT_DB` | Recommended | e.g. `./data/checkpoints.db` so chat history persists. |
| `KNOWLEDGE_BASE_DIR` | Optional | Default `knowledge_base` (relative to project root). |

Do **not** set `PORT`; Railway sets it automatically.

### 4. Persistent data (optional)

- By default, the app writes `data/` and `knowledge_base/` under the project root. On Railway, the filesystem is persistent for the life of the deployment, so restarts and redeploys keep data **if** you don’t clear the deployment environment.
- For stronger guarantees, you can attach a [Railway Volume](https://docs.railway.app/reference/volumes) and point `CHECKPOINT_DB` and/or `KNOWLEDGE_BASE_DIR` to paths on that volume (requires adjusting paths in config/env).

### 5. Deploy and share

- Push to the connected branch; Railway will redeploy.
- Open the **Generated Domain** (e.g. `https://your-app.up.railway.app`) and share that link with employees for testing.
- For future updates: push changes to the same branch (or the one Railway watches); Railway will redeploy automatically.

---

## Alternative: Render (free tier, ephemeral disk)

[Render](https://render.com) has a **free** web service tier.

- **Caveat:** the filesystem is **ephemeral**. Anything written to disk (users, chat log, knowledge base, index) is **lost on deploy or restart**. Fine for a quick, disposable demo; not for ongoing testing with real data.
- **Steps:** New → Web Service → Connect repo → Build: `pip install -r requirements.txt` (or use Render’s Python detection). Start: `python -m rag_agent.api`. Set `OPENAI_API_KEY`, `RAG_AGENT_SECRET_KEY`, and optionally `CHECKPOINT_DB`. Render sets `PORT` automatically (the app already uses it).

---

## Alternative: Fly.io (free allowance, persistent volume)

[Fly.io](https://fly.io) has a free allowance and supports **persistent volumes**.

- Create a `Dockerfile` that runs `uvicorn rag_agent.api:app --host 0.0.0.0 --port 8080` (or run `python -m rag_agent.api` and use `PORT`).
- Create an app, attach a [volume](https://fly.io/docs/reference/volumes/), and mount it e.g. at `/data`. Set `CHECKPOINT_DB=/data/checkpoints.db` and ensure `knowledge_base` and other writable paths use `/data` so they persist across deploys.

---

## Summary

| Goal | Suggested host |
|------|-----------------|
| Free, minimal setup, data persists between restarts | **Railway** (use free credit; set `CHECKPOINT_DB` and env vars). |
| Truly free, “throwaway” demo only | **Render** (accept that data is lost on restart/deploy). |
| More control, free allowance, persistent disk | **Fly.io** with a volume. |
| Must use Vercel later | Refactor to use a database, blob storage, and a vector store; then you can run the API as serverless functions. |

After deployment, send employees the app URL. They can register, log in, and use the chat. You can update the app by pushing to the connected branch; the host will redeploy.
