# Deploying the `platform` app to the Orlanda failover cluster

This is the concrete, values-filled deployment guide for **this** app (the RAG agent /
onboarding bot) onto the two-server pg_auto_failover cluster described in `server-info/`
(Docs 1–4). It is the generic Doc 4 playbook with every `<APP>` / `<PORT>` already resolved.

> Read `server-info/04-DEPLOY-NEW-APP-PLAYBOOK.md` once for the *why*. This doc is the *do*.

---

## What this app is (the deploy-relevant facts)

| Item | Value |
|---|---|
| App name | `platform` (already reserved in Doc 2 §8 as `db_platform` / user `platform`) |
| Public URL | `https://platform.n8norlanda.com` |
| Folder on both servers | `/opt/platform-agent/` |
| Local port | **8001** (roman owns 8000) |
| Stack | FastAPI + SQLAlchemy 2.0 (psycopg3) + LangGraph; React frontend built into the image |
| DB | `db_platform` / user `platform`, via the router at `127.0.0.1:6432` |
| Postgres extension | **pgvector** (`vector`) — already installed on both servers (Doc 2 §3) |
| Migrations | Alembic, run automatically on container start (`alembic upgrade head`) |
| Checkpoints/history | Postgres (`CHECKPOINT_BACKEND=postgres`) → replicates |
| Health endpoints | `/healthz` (DB-free, for the LB) and `/health` (runs `SELECT 1`) |
| Network | `network_mode: host` (no `ports:` mapping) |

### Two app-specific gotchas (do not skip)

1. **`RAG_AGENT_SECRET_KEY` MUST be identical on both servers.** It hashes both
   passwords and session tokens (`rag_agent/auth.py`). If the two servers differ, the
   moment the app fails over **nobody can log in** and all sessions drop. This is this
   app's equivalent of n8n's `N8N_ENCRYPTION_KEY`. Generate it once, paste the same value
   in both `.env` files.

2. **Disk state does NOT replicate** (Doc 1 §5). This app writes two things to disk that
   Postgres does *not* carry across:
   - `knowledge_base/` — the raw uploaded PDFs (their *embeddings* are in Postgres and do
     replicate, but the source PDFs are files).
   - `data/chat_conversations.json` — per-user conversation titles.

   Both are bind-mounted to `/opt/platform-agent/{knowledge_base,data}`. After any admin
   upload or change, **rsync them to the standby** (see §8) or they are lost on failover.
   (Hardening option for later: move both into Postgres so they replicate for free.)

---

## 0. Find the current primary

On the **witness** (`188.166.162.156`):
```
sudo -u postgres /usr/bin/pg_autoctl show state --pgdata /var/lib/postgresql/ha
```
The `read-write` / `primary` row is the Boss. Do all "on the primary" steps there.
(At time of writing that is **Hetzner**, `46.62.186.134`.)

---

## 1. Extension — already done, just verify

pgvector is already installed on both servers (it is required by roman too). Confirm on
**each** server:
```
apt -qq list --installed 2>/dev/null | grep pgvector
```
If for some reason it is missing on one: `sudo apt-get install -y postgresql-16-pgvector`.

---

## 2. Create the database, user, grants, extension — ONCE on the PRIMARY

**2a — make a password and keep it** (goes into `.env` as `DB_PASSWORD`):
```
openssl rand -base64 18
```

**2b — create db + user + grant** (interactive prompt avoids heredoc paste-scramble):
```
sudo -u postgres psql -p 5432
```
Then type, one line at a time (paste your password in place of `PASTE_PASSWORD`):
```
CREATE DATABASE db_platform;
CREATE USER platform WITH PASSWORD 'PASTE_PASSWORD';
GRANT ALL PRIVILEGES ON DATABASE db_platform TO platform;
\q
```

**2c — schema rights** (PostgreSQL 16 needs this or Alembic can't create tables):
```
sudo -u postgres psql -p 5432 -d db_platform -c "GRANT ALL ON SCHEMA public TO platform;"
```

**2d — create the pgvector extension as superuser** (the `platform` user is not a
superuser; migration 006 tries `CREATE EXTENSION` and falls back to verifying it exists,
so creating it here makes startup clean):
```
sudo -u postgres psql -p 5432 -d db_platform -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

**2e — verify it replicated** to the **standby**:
```
sudo -u postgres psql -p 5432 -c "\l" | grep db_platform
sudo -u postgres psql -p 5432 -d db_platform -c "\dx" | grep vector
```

---

## 3. `pg_hba.conf` app lines — on BOTH servers (does not replicate)

On **each** of Hetzner and OVH, edit with `nano` (never a heredoc):
```
sudo nano /var/lib/postgresql/ha/pg_hba.conf
```
Add at the **bottom** (these are scram app rules; they don't conflict with the replicator
`trust` rules above):
```
host    db_platform  platform  127.0.0.1/32       scram-sha-256
host    db_platform  platform  172.16.0.0/12      scram-sha-256
hostssl db_platform  platform  46.62.186.134/32   scram-sha-256
hostssl db_platform  platform  193.70.47.219/32   scram-sha-256
```
Reload on **each** server:
```
sudo systemctl reload pgautofailover
```
Verify the parse on each:
```
sudo -u postgres psql -p 5432 -c "SELECT line_number, database, address, auth_method FROM pg_hba_file_rules WHERE user_name='{platform}' ORDER BY line_number;"
```

---

## 4. Get the code onto BOTH servers → `/opt/platform-agent/`

If the repo is pushed to git:
```
cd /opt
sudo git clone <PLATFORM_REPO_URL> platform-agent
```
If it only exists locally, copy the folder to each server instead (from your machine):
```
scp -r ./onboarding_bot_from_scratch arman@193.70.47.219:/tmp/platform-agent
# then on the server:  sudo mv /tmp/platform-agent /opt/platform-agent
```
The two servers' `/opt/platform-agent/` folders must end up identical.

> The repo's `docker-compose.yml` is already cluster-ready: **no bundled Postgres**,
> `network_mode: host`, DB at `127.0.0.1:6432`, port 8001. Nothing to edit there.

---

## 5. Create `.env` on BOTH servers (identical)

Copy `.env.example` → `.env` and fill it in on the **primary**:
```
cd /opt/platform-agent
sudo cp .env.example .env
sudo nano .env
```
Set:
- `DB_PASSWORD=` the password from step 2a
- `OPENAI_API_KEY=` , `ANTHROPIC_API_KEY=`
- `RAG_AGENT_SECRET_KEY=` generate once: `python3 -c "import secrets; print(secrets.token_hex(32))"`
- leave `RAG_AGENT_ADMIN_USERNAMES`, `RAG_ALLOWED_EMAIL_DOMAIN`, `RAG_AGENT_MODEL`,
  `RAG_CORS_ALLOWED_ORIGINS=https://platform.n8norlanda.com` as provided.

Then copy the **exact same** `.env` to the standby (don't retype secrets):
```
scp /opt/platform-agent/.env arman@193.70.47.219:/tmp/platform.env
# on the standby:
sudo mv /tmp/platform.env /opt/platform-agent/.env
```
> ⚠ Confirm `RAG_AGENT_SECRET_KEY` and `DB_PASSWORD` are byte-identical on both.

---

## 6. Start on the PRIMARY; build-only on the STANDBY

**On the PRIMARY:**
```
cd /opt/platform-agent
sudo docker compose up -d --build
sudo docker compose logs --tail 60 -f      # watch Alembic migrate then uvicorn boot; Ctrl+C when stable
```
Verify locally (host network mode → port is on the host):
```
curl -i http://127.0.0.1:8001/healthz      # 200, DB-free
curl -i http://127.0.0.1:8001/health       # 200 {"db":"ok"} proves app→router→primary DB
```

**On the STANDBY** — build but do **not** start:
```
cd /opt/platform-agent
sudo docker compose build
sudo docker compose ps        # nothing running — correct for the standby
```

---

## 7. Add `platform` to the role agent — on BOTH servers

The watchdog `/usr/local/bin/orlanda-role-agent.sh` starts/stops apps on failover. Add
`/opt/platform-agent` to its app list on **each** server:
```
sudo nano /usr/local/bin/orlanda-role-agent.sh
```
Make the app-dir list include both apps (keep identical on both servers):
```bash
APP_DIRS=("/opt/roman-agent" "/opt/platform-agent")
apps_up()   { for d in "${APP_DIRS[@]}"; do (cd "$d" && docker compose up -d); done; }
apps_down() { for d in "${APP_DIRS[@]}"; do (cd "$d" && docker compose stop); done; }
```
Restart it on **each** server, then check:
```
sudo systemctl restart orlanda-role-agent
sudo journalctl -t orlanda-role -n 10
```
Primary: `platform` now running. Standby: still stopped.

---

## 8. Tunnel ingress — on BOTH servers

Add the platform hostname to each server's existing tunnel (do **not** create a new
tunnel). On **each** server:
```
sudo nano /etc/cloudflared/config.yml
```
Add the `platform` block above the final `http_status:404` (keep the roman line; leave
`tunnel:` and `credentials-file:` untouched — they differ per server):
```yaml
ingress:
  - hostname: roman.n8norlanda.com
    service: http://localhost:8000
  - hostname: platform.n8norlanda.com
    service: http://localhost:8001
  - service: http_status:404
```
Restart and check on **each** server:
```
sudo systemctl restart cloudflared
sudo journalctl -u cloudflared -n 15
```

### Keeping the disk state in sync (because it doesn't replicate)

After any PDF upload or conversation change on the primary, push the bind-mounted files to
the standby (adjust direction/user to match the current primary):
```
sudo rsync -a /opt/platform-agent/knowledge_base/ arman@193.70.47.219:/opt/platform-agent/knowledge_base/
sudo rsync -a /opt/platform-agent/data/           arman@193.70.47.219:/opt/platform-agent/data/
```
(Consider a cron rsync, or the longer-term fix of moving these into Postgres.)

---

## 9. Cloudflare Load Balancer for `platform.n8norlanda.com` (dashboard)

In the `n8norlanda.com` zone → **Traffic → Load Balancing**, create a public LB for
hostname `platform.n8norlanda.com`:

**Two origin pools** (endpoint = `<tunnel-id>.cfargotunnel.com`, app hostname in the
**Host header**):
- `hetzner-platform` → `eef8c8ec-751d-4015-a43e-97f658e44a6d.cfargotunnel.com`, Host header `platform.n8norlanda.com`
- `ovh-platform` → `a63d51ed-2749-446d-8a73-9f40a4f53f85.cfargotunnel.com`, Host header `platform.n8norlanda.com`

**Health monitor:** Type **HTTPS**, path **`/healthz`**, expected `200`, interval ~15 s
(set the monitor's Host header to `platform.n8norlanda.com` too if it has that field).

**Steering / fallback:** pool order `hetzner-platform` then `ovh-platform`; method
**Off / failover** (first healthy pool).

Expected after deploy: `hetzner-platform` healthy, `ovh-platform` unhealthy (correct — OVH
is the standby). On failover, OVH's role agent starts the app and the LB shifts traffic.

> Gotchas (Doc 2 §11): the endpoint must be the `cfargotunnel.com` form **with the Host
> header**, and the monitor must hit **`/healthz`** (DB-free), not `/health`.

---

## 10. Verify end-to-end

1. Browser: `https://platform.n8norlanda.com` loads (served by the primary).
2. `curl https://platform.n8norlanda.com/health` → 200 (`app → router → primary DB` works).
3. Single-writer: `docker compose ps` shows **running on primary, stopped on standby**.
4. (Calm window) run the failover drill (Doc 3 §5): kill `pgautofailover` on the primary,
   confirm platform comes up on the survivor and the page stays reachable; recover per
   Doc 3 §6. Re-check that the `knowledge_base`/`data` files were in sync first.
5. Monitoring: add an UptimeRobot **keyword** monitor on `platform.n8norlanda.com`
   (Doc 3 §9) — a word only present when the real app loads.

---

## Quick checklist

- [ ] Confirm the current primary (§0)
- [ ] Verify pgvector on both servers (§1)
- [ ] `db_platform` + user + grants + extension on the **primary**; verify it replicated (§2)
- [ ] `pg_hba.conf` platform lines on **both** servers; reload each (§3)
- [ ] Code in `/opt/platform-agent/` on **both** servers (§4)
- [ ] Identical `.env` on both — **same `RAG_AGENT_SECRET_KEY` & `DB_PASSWORD`** (§5)
- [ ] `up -d --build` on the **primary**; `build` only on the **standby** (§6)
- [ ] `platform` added to the role agent on **both**; restart (§7)
- [ ] Tunnel ingress line on **both**; restart cloudflared; set up `knowledge_base`/`data` rsync (§8)
- [ ] Cloudflare LB: 2 pools + Host header + `/healthz` monitor (§9)
- [ ] End-to-end verify + monitoring (§10)
