"""
REST API for the RAG agent. Serves a local chat UI with login; /chat uses your account as thread_id so history is restored.
"""
import json
import logging
import logging.config
import time
from contextlib import asynccontextmanager
from pathlib import Path

import asyncio
from urllib.parse import urlencode

from fastapi import FastAPI, HTTPException, Header, Request, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from rag_agent.agent import (
    Context,
    build_agent,
    close_checkpointer,
    compose_system_prompt_suffix,
    delete_conversation_state,
    get_active_model_name,
    get_base_agent,
    monday_system_prompt,
    set_active_model,
)
from rag_agent.auth import (
    change_client_email as auth_change_client_email,
    change_password as auth_change_password,
    get_user_id,
    get_user_role,
    get_user_auth_flags,
    invalidate_token,
    is_password_change_required,
    login as auth_login,
    provision_user_with_temp_password,
    register as auth_register,
    resolve_token,
)
from rag_agent import user_memory
from rag_agent.memory_tools import get_memory_tools_for_user
from rag_agent.config import (
    API_HOST,
    API_PORT,
    RAG_CORS_ALLOWED_ORIGINS,
    RAG_FRONTEND_BASE_URL,
    RAG_LOG_FORMAT,
    RAG_MAX_AGENT_RECURSION_LIMIT,
    RAG_MONDAY_AGENT_RECURSION_LIMIT,
    RAG_ENABLE_RATE_LIMIT_FALLBACK,
    RAG_FALLBACK_MODEL,
    RAG_HISTORY_KEEP_LAST_MESSAGES,
    RAG_HISTORY_SUMMARY_MAX_TOKEN_LIMIT,
    RAG_AGENT_DIR,
    RAG_MAX_HISTORY_MESSAGES,
    RAG_MAX_PASSWORD_LENGTH,
    RAG_MAX_USER_MEMORIES,
    RAG_MAX_USER_MESSAGE_CHARS,
    RAG_MIN_PASSWORD_LENGTH,
    RAG_RATE_LIMIT_CHAT,
    CLIENT_MIN_PASSWORD_LENGTH,
    PROGRESS_BASE_URL,
    RAG_RATE_LIMIT_CLIENT_REGISTER,
    RAG_RATE_LIMIT_LOGIN,
    RAG_RATE_LIMIT_REGISTER,
    RAG_USERNAME_MAX_LEN,
    client_portal_enabled,
    memory_enabled,
    monday_enabled,
    require_runtime_keys,
    warn_oauth_redirect_misconfig,
)
from rag_agent.audit_log import count_audit, list_audit, write_audit
from rag_agent import client_portal
from rag_agent.monday_oauth import (
    build_authorize_url as monday_build_authorize_url,
    delete_token as monday_delete_token,
    exchange_code_for_token as monday_exchange_code_for_token,
    fetch_monday_identity as monday_fetch_identity,
    get_connection_status as monday_get_connection_status,
    store_token as monday_store_token,
    verify_state as monday_verify_state,
)
from rag_agent.monday_tools import aget_monday_tools_for_user
from rag_agent.indexing import (
    KNOWLEDGE_BASE_DIR,
    reconcile_all_documents,
    upsert_pdf_document,
    delete_pdf_document,
    upsert_knowledge_item,
    delete_knowledge_item_document,
    extract_pdf_plain_text,
    list_knowledge_files,
    rag_sidecar_path,
)
from rag_agent.knowledge_items import (
    add_item as ki_add,
    delete_item as ki_delete,
    get_item as ki_get,
    list_items as ki_list,
    update_item as ki_update,
    UNSET as KI_UNSET,
)
from rag_agent.doc_metadata import (
    compute_expiry,
    delete_pdf_metadata,
    get_pdf_metadata,
    record_pdf_upload,
    set_pdf_update_period,
)
from rag_agent.rag_tool import get_last_sources, invalidate_vector_store, retrieval_debug
from rag_agent.chat_log import (
    append as log_append,
    list_entries as log_list_entries,
    count as log_count,
    update_review as log_update_review,
)
from rag_agent.chat_conversations import (
    list_for_user as list_conversation_meta_for_user,
    upsert_for_user as upsert_conversation_meta_for_user,
    delete_for_user as delete_conversation_meta_for_user,
)

STATIC_DIR = RAG_AGENT_DIR / "static"
PROJECT_DIR = RAG_AGENT_DIR.parent
FRONTEND_DIR = RAG_AGENT_DIR / "frontend"
FRONTEND_DIST_DIR = FRONTEND_DIR / "dist"
FRONTEND_INDEX_PATH = FRONTEND_DIST_DIR / "index.html"


class _JsonFormatter(logging.Formatter):
    """Single-line JSON log records for structured log aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


if RAG_LOG_FORMAT == "text":
    logging.basicConfig(level=logging.INFO)
else:
    _json_handler = logging.StreamHandler()
    _json_handler.setFormatter(_JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[_json_handler], force=True)

logger = logging.getLogger(__name__)


def _is_rate_limit_error(err: Exception) -> bool:
    txt = str(err).lower()
    return "rate_limit" in txt or "rate limit" in txt or "error code: 429" in txt


def _is_provider_overloaded_error(err: Exception) -> bool:
    txt = str(err).lower()
    return "overloaded" in txt or "error code: 529" in txt


def _is_structured_output_validation_error(err: Exception) -> bool:
    txt = str(err).lower()
    return "structuredoutputvalidationerror" in txt or "failed to parse structured output" in txt


def _extract_agent_response_text(response: dict) -> str:
    """
    Extract assistant text from both structured and non-structured agent responses.
    """
    structured = response.get("structured_response")
    if structured is not None:
        val = getattr(structured, "response_content", None)
        if isinstance(val, str) and val.strip():
            return val.strip()

    messages = response.get("messages")
    if isinstance(messages, list) and messages:
        for msg in reversed(messages):
            role = str(getattr(msg, "type", None) or getattr(msg, "role", None) or "").lower()
            if role not in {"assistant", "ai"}:
                continue
            content = getattr(msg, "content", "")
            if isinstance(content, list):
                text_parts = []
                for c in content:
                    if isinstance(c, dict):
                        t = c.get("text")
                        if t:
                            text_parts.append(str(t))
                    elif c:
                        text_parts.append(str(c))
                merged = " ".join(p.strip() for p in text_parts if p and p.strip()).strip()
                if merged:
                    return merged
            elif isinstance(content, str) and content.strip():
                return content.strip()
            elif content:
                s = str(content).strip()
                if s:
                    return s

    output = response.get("output")
    if isinstance(output, str) and output.strip():
        return output.strip()
    return ""


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)


class ConversationCreateRequest(BaseModel):
    id: str = Field(..., min_length=1, max_length=128)
    title: str = Field(..., min_length=1, max_length=256)


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=RAG_USERNAME_MAX_LEN)
    password: str = Field(..., min_length=1, max_length=RAG_MAX_PASSWORD_LENGTH)


class RegisterRequest(BaseModel):
    """Registration: password length matches server policy (letter + digit checked in auth)."""

    username: str = Field(..., min_length=2, max_length=RAG_USERNAME_MAX_LEN)
    password: str = Field(..., min_length=RAG_MIN_PASSWORD_LENGTH, max_length=RAG_MAX_PASSWORD_LENGTH)


class ChatResponse(BaseModel):
    response: str
    sources: list[dict] = Field(description="List of {file, page} used for the answer; empty if RAG was not used.")
    tool_events: list[dict] = Field(default_factory=list, description="Operational tool activity events shown in UI.")


class AuthResponse(BaseModel):
    token: str
    username: str
    role: str = "user"
    must_change_password: bool = False


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(default="", max_length=RAG_MAX_PASSWORD_LENGTH)
    new_password: str = Field(..., min_length=RAG_MIN_PASSWORD_LENGTH, max_length=RAG_MAX_PASSWORD_LENGTH)
    repeat_password: str = Field(..., min_length=RAG_MIN_PASSWORD_LENGTH, max_length=RAG_MAX_PASSWORD_LENGTH)


class AdminProvisionUserRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=RAG_USERNAME_MAX_LEN)
    role: str = Field(default="user", max_length=16)


class PdfMetadataUpdate(BaseModel):
    # Relative path under `knowledge_base` (must be a PDF).
    path: str = Field(..., min_length=1, max_length=1024)
    # How often it should be reviewed/replaced. Use null to disable expiry.
    update_period_days: int | None = Field(..., ge=1, le=3650)


class AdminLogReviewUpdate(BaseModel):
    # 1..10 score of answer quality; null means "not set".
    score: int | None = Field(default=None, ge=1, le=10)
    # Correct answer text entered by admin; empty string allowed.
    correct_answer: str | None = Field(default=None, max_length=50_000)


class AdminModelUpdate(BaseModel):
    model: str = Field(..., min_length=1, max_length=256)


class MemoryCreate(BaseModel):
    content: str = Field(..., min_length=1, max_length=4000)
    category: str = Field(default="fact", max_length=16)


class MemoryUpdate(BaseModel):
    content: str = Field(..., min_length=1, max_length=4000)


class MemorySettingsUpdate(BaseModel):
    enabled: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    require_runtime_keys()
    warn_oauth_redirect_misconfig()
    logger.info("Active chat model: %s", get_active_model_name())
    # Warm the checkpointer/base agent once at startup. Doing this inside a try
    # means a DB/router problem is logged and surfaced cleanly here instead of
    # crashing at import time; endpoints can still return a clean 5xx afterwards.
    try:
        get_base_agent()
    except Exception:
        logger.exception("Failed to warm base agent/checkpointer on startup")
    try:
        yield
    finally:
        close_checkpointer()


app = FastAPI(title="RAG Agent API", lifespan=lifespan)

# Rate limiting — keyed by client IP.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# CORS — allow browser requests from configured origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=RAG_CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
)


@app.middleware("http")
async def _security_headers(request: Request, call_next):
    """Inject HTTP security headers on every response."""
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    # frame-src whitelists the progress-tracking app: the client cabinet embeds
    # its /p/{token} pages in an iframe (default-src 'self' would block them).
    progress_origin = PROGRESS_BASE_URL or ""
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        f"frame-src 'self' {progress_origin}".rstrip() + "; "
        "connect-src 'self'"
    )
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    return response


def _serve_frontend_or_legacy(legacy_path: Path) -> str:
    """Serve built React app if present, otherwise fallback to legacy static HTML."""
    if FRONTEND_INDEX_PATH.is_file():
        return FRONTEND_INDEX_PATH.read_text(encoding="utf-8")
    if legacy_path.is_file():
        return legacy_path.read_text(encoding="utf-8")
    raise HTTPException(status_code=404, detail="UI bundle not found")


def _dist_asset_path(relative_path: str) -> Path | None:
    """Resolve one dist asset and ensure it stays inside frontend/dist directory."""
    clean = (relative_path or "").strip().replace("\\", "/")
    if not clean or ".." in clean or clean.startswith("/"):
        return None
    target = (FRONTEND_DIST_DIR / clean).resolve()
    try:
        target.relative_to(FRONTEND_DIST_DIR.resolve())
    except ValueError:
        return None
    return target


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve application shell (React if built, legacy otherwise)."""
    return _serve_frontend_or_legacy(STATIC_DIR / "index.html")


@app.get("/healthz")
def healthz():
    """Liveness check for the Cloudflare load balancer — intentionally DB-free.

    The LB health monitor points here (not at /health) so a transient DB blip on
    the primary cannot flap the pool. See server-info Doc 2 §11 / Doc 4 §9.
    """
    return {"status": "ok"}


@app.get("/health")
def health():
    """Production health check — verifies database connectivity."""
    from sqlalchemy import text as _sa_text
    from rag_agent.db.session import get_session_factory
    try:
        with get_session_factory()() as db:
            db.execute(_sa_text("SELECT 1"))
        return {"status": "ok", "db": "ok"}
    except Exception as exc:
        logger.error("Health check DB error: %s", exc)
        return JSONResponse(status_code=503, content={"status": "degraded", "db": "error"})


@app.get("/admin", response_class=HTMLResponse)
def admin_index():
    """Serve admin logs page by default."""
    return _serve_frontend_or_legacy(STATIC_DIR / "admin.html")


@app.get("/admin/logs")
def admin_logs(
    authorization: str | None = Header(default=None),
    limit: int = 100,
    offset: int = 0,
    accept: str | None = Header(default=None),
):
    """
    Serve admin logs SPA shell for browser navigation (text/html),
    and return JSON log entries for API fetches.
    """
    accept_l = (accept or "").lower()
    if "text/html" in accept_l:
        return _serve_frontend_or_legacy(STATIC_DIR / "admin.html")

    _require_admin(authorization)
    if limit < 1:
        limit = 100
    if limit > 500:
        limit = 500
    if offset < 0:
        offset = 0
    entries = log_list_entries(limit=limit, offset=offset)
    total = log_count()
    return {"entries": entries, "total": total}


@app.get("/admin/documents", response_class=HTMLResponse)
def admin_documents_page():
    """Serve dedicated admin document metadata page."""
    return _serve_frontend_or_legacy(STATIC_DIR / "admin_documents.html")


@app.get("/auth", response_class=HTMLResponse)
def auth_index():
    """Serve auth route shell for SPA frontend."""
    return _serve_frontend_or_legacy(STATIC_DIR / "index.html")


@app.get("/chat", response_class=HTMLResponse)
def chat_index():
    """Serve chat route shell for SPA frontend."""
    return _serve_frontend_or_legacy(STATIC_DIR / "index.html")


@app.get("/components", response_class=HTMLResponse)
def components_index():
    """Serve components route shell for SPA frontend."""
    return _serve_frontend_or_legacy(STATIC_DIR / "index.html")


@app.get("/assets/{asset_path:path}")
def frontend_asset(asset_path: str):
    """Serve built frontend assets from frontend/dist/assets."""
    target = _dist_asset_path(f"assets/{asset_path}")
    if not target or not target.is_file():
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(target)


@app.get("/favicon.ico")
def frontend_favicon():
    """Serve frontend favicon if available, fallback to legacy static favicon."""
    dist_favicon = _dist_asset_path("favicon.ico")
    if dist_favicon and dist_favicon.is_file():
        return FileResponse(dist_favicon)
    legacy_favicon = STATIC_DIR / "favicon.ico"
    if legacy_favicon.is_file():
        return FileResponse(legacy_favicon)
    raise HTTPException(status_code=404, detail="favicon not found")


@app.get("/branding/logo")
def branding_logo():
    """Serve Orlanda logo image if present in project folders."""
    candidates: list[Path] = []
    search_dirs = [PROJECT_DIR, STATIC_DIR]
    allowed_suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    for base in search_dirs:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in allowed_suffixes:
                continue
            name_l = p.name.lower()
            if ("orlanda" in name_l and "logo" in name_l) or name_l in {"logo.png", "orlanda.png"}:
                candidates.append(p)
    if not candidates:
        raise HTTPException(status_code=404, detail="Logo not found")
    # Prefer files from static directory if available.
    candidates.sort(key=lambda p: (0 if STATIC_DIR in p.parents else 1, len(str(p))))
    return FileResponse(candidates[0])


@app.patch("/admin/logs/{entry_id}/review")
def admin_log_review_update(
    entry_id: str,
    body: AdminLogReviewUpdate,
    request: Request,
    authorization: str | None = Header(default=None),
):
    """Update score/correct_answer for one log entry. Requires admin."""
    admin_username = _require_admin(authorization)
    fields_set = getattr(body, "model_fields_set", set()) or set()
    if not fields_set:
        raise HTTPException(status_code=400, detail="Нет полей для обновления")
    updated = log_update_review(
        entry_id=entry_id,
        score=body.score if "score" in fields_set else None,
        correct_answer=body.correct_answer if "correct_answer" in fields_set else None,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Запись лога не найдена")
    write_audit(
        "review_log",
        admin_username,
        target=entry_id,
        details={k: getattr(body, k) for k in fields_set},
        ip_address=request.client.host if request.client else "",
    )
    return {"ok": True, "entry": updated}


@app.get("/admin/model")
def admin_model_get(authorization: str | None = Header(default=None)):
    """Return active chat model. Requires admin."""
    _require_admin(authorization)
    return {"model": get_active_model_name()}


@app.get("/admin/retrieval/debug")
def admin_retrieval_debug(
    q: str = Query(..., min_length=1, max_length=2000),
    limit: int = Query(default=12, ge=1, le=50),
    authorization: str | None = Header(default=None),
):
    """
    Retrieval diagnostics for one query:
    returns query variants, ranked candidates, and selected sources. Requires admin.
    """
    _require_admin(authorization)
    result = retrieval_debug(q.strip(), limit=limit)
    return result


@app.put("/admin/model")
def admin_model_put(
    body: AdminModelUpdate,
    request: Request,
    authorization: str | None = Header(default=None),
):
    """Set active chat model at runtime. Requires admin."""
    admin_username = _require_admin(authorization)
    old_model = get_active_model_name()
    try:
        model = set_active_model(body.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    write_audit(
        "change_model",
        admin_username,
        details={"old": old_model, "new": model},
        ip_address=request.client.host if request.client else "",
    )
    return {"ok": True, "model": model}


@app.get("/admin/audit")
def admin_audit(
    authorization: str | None = Header(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """Return admin action audit log (newest first). Requires admin."""
    _require_admin(authorization)
    entries = list_audit(limit=limit, offset=offset)
    total = count_audit()
    return {"entries": entries, "total": total}


@app.get("/admin/documents/metadata")
def admin_documents_metadata(authorization: str | None = Header(default=None)):
    """Return metadata for PDFs and text knowledge items (for admin table). Requires admin."""
    _require_admin(authorization)

    from datetime import datetime, timezone

    pdfs = []
    for f in list_knowledge_files():
        rel_path = f["path"]
        meta = get_pdf_metadata(rel_path)
        last_updated_at = meta.get("last_updated_at") or ""

        # Backfill last_updated_at from file mtime if metadata is missing.
        if not last_updated_at:
            try:
                target = KNOWLEDGE_BASE_DIR / rel_path
                if target.is_file():
                    last_updated_at = datetime.fromtimestamp(
                        target.stat().st_mtime, tz=timezone.utc
                    ).isoformat()
            except Exception:
                pass

        expiry = compute_expiry(last_updated_at, meta.get("update_period_days"))
        pdfs.append(
            {
                "path": rel_path,
                "name": f.get("name"),
                "size": f.get("size"),
                "last_updated_at": last_updated_at,
                "update_period_days": meta.get("update_period_days"),
                "responsible": meta.get("responsible") or "",
                "expires_at": expiry.get("expires_at") or "",
                "expired": bool(expiry.get("expired")),
            }
        )

    items = []
    for it in ki_list():
        last_updated_at = it.get("last_updated_at") or ""
        update_period_days = it.get("update_period_days")
        expiry = compute_expiry(last_updated_at, update_period_days)
        items.append(
            {
                "id": it.get("id"),
                "name": it.get("name"),
                "last_updated_at": last_updated_at,
                "update_period_days": update_period_days,
                "responsible": it.get("responsible") or "",
                "expires_at": expiry.get("expires_at") or "",
                "expired": bool(expiry.get("expired")),
            }
        )

    return {"pdfs": pdfs, "items": items}


@app.get("/admin/history/threads")
def admin_history_threads(
    authorization: str | None = Header(default=None),
    max_threads: int = Query(default=200, ge=1, le=2000),
    scan_checkpoints: int = Query(default=5000, ge=100, le=50000),
    near_ratio: float = Query(default=0.8, ge=0.1, le=1.0),
):
    """
    Inspect existing chat threads and report history pressure (near/over compaction threshold).
    Read-only diagnostics endpoint for admins.
    """
    _require_admin(authorization)
    base_agent = get_base_agent()
    cp = getattr(base_agent, "checkpointer", None)
    list_fn = getattr(cp, "list", None) if cp is not None else None
    if not callable(list_fn):
        return {
            "threshold": RAG_MAX_HISTORY_MESSAGES,
            "near_ratio": near_ratio,
            "total_threads": 0,
            "near_limit": 0,
            "over_limit": 0,
            "threads": [],
            "warning": "Checkpointer does not support thread listing in this runtime.",
        }

    # We dedupe by thread_id while scanning newest checkpoints first.
    discovered_thread_ids: list[str] = []
    seen = set()
    scanned = 0
    for item in list_fn(None, limit=scan_checkpoints):
        scanned += 1
        conf = getattr(item, "config", None) or {}
        confg = conf.get("configurable", {}) if isinstance(conf, dict) else {}
        thread_id = str(confg.get("thread_id") or "").strip()
        if not thread_id or thread_id in seen:
            continue
        seen.add(thread_id)
        discovered_thread_ids.append(thread_id)
        if len(discovered_thread_ids) >= max_threads:
            break

    threshold = max(0, int(RAG_MAX_HISTORY_MESSAGES))
    near_threshold = max(1, int(threshold * float(near_ratio))) if threshold > 0 else 0
    threads: list[dict] = []
    near_count = 0
    over_count = 0

    for thread_id in discovered_thread_ids:
        cfg = {"configurable": {"thread_id": thread_id}}
        semantic_count = 0
        load_error = ""
        try:
            state = base_agent.get_state(cfg)
            values = getattr(state, "values", None) or {}
            messages = values.get("messages", []) or []
            semantic_count = _semantic_message_count(messages)
        except Exception as e:
            load_error = str(e)
        if threshold > 0 and semantic_count >= near_threshold:
            near_count += 1
        if threshold > 0 and semantic_count > threshold:
            over_count += 1

        username, conversation_id = thread_id, "default"
        if ":" in thread_id:
            username, conversation_id = thread_id.split(":", 1)

        status = "ok"
        if threshold > 0 and semantic_count > threshold:
            status = "over_limit"
        elif threshold > 0 and semantic_count >= near_threshold:
            status = "near_limit"

        threads.append(
            {
                "thread_id": thread_id,
                "username": username,
                "conversation_id": conversation_id,
                "semantic_messages": semantic_count,
                "status": status,
                "error": load_error,
            }
        )

    threads.sort(key=lambda x: x.get("semantic_messages", 0), reverse=True)
    return {
        "threshold": threshold,
        "near_ratio": near_ratio,
        "near_threshold": near_threshold if threshold > 0 else 0,
        "scanned_checkpoints": scanned,
        "total_threads": len(threads),
        "near_limit": near_count,
        "over_limit": over_count,
        "threads": threads,
    }


def _get_username(
    authorization: str | None = Header(default=None),
    *,
    enforce_password_rotation: bool = True,
) -> str:
    """Require Bearer token and return username; optionally block access until password is changed."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Требуется вход в аккаунт")
    token = authorization[7:].strip()
    username = resolve_token(token)
    if not username:
        raise HTTPException(status_code=401, detail="Сессия истекла, войдите снова")
    if enforce_password_rotation and is_password_change_required(username):
        raise HTTPException(
            status_code=403,
            detail="Требуется сменить временный пароль. Откройте настройки аккаунта и задайте новый пароль.",
        )
    return username


def _require_admin(authorization: str | None = Header(default=None)) -> str:
    """Require valid token and admin role; return username. 401 if not logged in, 403 if not admin."""
    username = _get_username(authorization)
    if get_user_role(username) != "admin":
        raise HTTPException(status_code=403, detail="Доступ только для администратора")
    return username


def _make_thread_id(username: str, conversation_id: str | None) -> str:
    """
    Build a stable thread_id so one user can have multiple separate chats.
    If conversation_id is empty, fall back to a default conversation.
    """
    conv = (conversation_id or "default").strip() or "default"
    return f"{username}:{conv}"


def _parse_thread_id(thread_id: str) -> tuple[str, str]:
    """Split thread id into (username, conversation_id) with defaults."""
    raw = str(thread_id or "").strip()
    if not raw:
        return "", "default"
    if ":" in raw:
        username, conversation_id = raw.split(":", 1)
        return username.strip(), (conversation_id.strip() or "default")
    return raw, "default"


def _semantic_message_count(messages) -> int:
    """Count only user/assistant turns (ignore tool/system chatter)."""
    total = 0
    for m in messages or []:
        role_raw = None
        if isinstance(m, dict):
            role_raw = m.get("role") or m.get("type")
        else:
            role_raw = getattr(m, "role", None) or getattr(m, "type", None)
            if not role_raw and hasattr(m, "__class__"):
                name = m.__class__.__name__.lower()
                if "ai" in name or "assistant" in name:
                    role_raw = "assistant"
                elif "human" in name or "user" in name:
                    role_raw = "user"
        role = str(role_raw or "").strip().lower()
        if role in {"assistant", "ai", "user", "human"}:
            total += 1
    return total


def _plain_text_from_message_content(content_raw) -> str:
    """
    Keep only human-readable text from message content.
    Drops tool_use/tool_result blocks so persisted history stays provider-safe.
    """
    if content_raw is None:
        return ""
    if isinstance(content_raw, str):
        return content_raw.strip()
    if isinstance(content_raw, list):
        text_parts: list[str] = []
        for block in content_raw:
            if isinstance(block, dict):
                block_type = str(block.get("type") or "").strip().lower()
                if block_type in {"tool_use", "tool_result"}:
                    continue
                txt = block.get("text")
                if txt:
                    text_parts.append(str(txt).strip())
            elif block:
                text_parts.append(str(block).strip())
        return " ".join(p for p in text_parts if p).strip()
    if isinstance(content_raw, dict):
        txt = content_raw.get("text") or content_raw.get("content") or ""
        return str(txt).strip()
    return str(content_raw).strip()


def _has_tool_protocol_blocks(content_raw) -> bool:
    """Detect provider-specific tool protocol blocks in message content."""
    if isinstance(content_raw, list):
        for block in content_raw:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type") or "").strip().lower()
            if block_type in {"tool_use", "tool_result"}:
                return True
    return False


def _extract_tool_call_names(message_obj) -> list[str]:
    """Return tool names referenced by a persisted message object/dict."""
    try:
        calls = None
        if isinstance(message_obj, dict):
            calls = message_obj.get("tool_calls")
        else:
            calls = getattr(message_obj, "tool_calls", None)
        if not isinstance(calls, list):
            return []
        names: list[str] = []
        for call in calls:
            if isinstance(call, dict):
                nm = str(call.get("name") or "").strip()
                if nm:
                    names.append(nm)
        return names
    except Exception:
        return []


def _sanitize_persisted_messages_for_provider(messages) -> tuple[list[dict[str, str]], bool]:
    """
    Normalize persisted history into provider-safe plain-text turns.
    Removes tool protocol blocks so resumed requests cannot fail on strict pairing rules.
    Returns (sanitized_messages, changed_flag).
    """
    sanitized: list[dict[str, str]] = []
    changed = False
    removed_tool_call_msgs = 0
    for m in messages or []:
        role_raw = None
        content_raw = ""
        if isinstance(m, dict):
            role_raw = m.get("role") or m.get("type")
            content_raw = m.get("content") or ""
        else:
            role_raw = getattr(m, "role", None) or getattr(m, "type", None)
            content_raw = getattr(m, "content", "") or ""
            if not role_raw and hasattr(m, "__class__"):
                name = m.__class__.__name__.lower()
                if "ai" in name or "assistant" in name:
                    role_raw = "assistant"
                elif "human" in name or "user" in name:
                    role_raw = "user"

        role = str(role_raw or "").strip().lower()
        if role in {"tool"}:
            changed = True
            continue
        if role in {"ai"}:
            role = "assistant"
            changed = True
        elif role in {"human"}:
            role = "user"
            changed = True
        elif role not in {"assistant", "user", "system"}:
            changed = True
            continue

        tool_call_names = _extract_tool_call_names(m)
        if tool_call_names:
            # Drop assistant planning/tool-call stubs from persisted state to avoid
            # replay loops and cross-mode tool contamination on next invocation.
            changed = True
            removed_tool_call_msgs += 1
            continue

        if _has_tool_protocol_blocks(content_raw):
            changed = True
        content = _plain_text_from_message_content(content_raw)
        if not content:
            if content_raw:
                changed = True
            continue
        if not isinstance(content_raw, str):
            changed = True
        sanitized.append({"role": role, "content": content})

    return sanitized, changed


def _repair_conversation_history_for_provider(runtime_agent, config: dict) -> bool:
    """
    Best-effort repair for persisted message history.
    Rewrites thread history into plain user/assistant/system turns without tool protocol blocks.
    """
    get_state = getattr(runtime_agent, "get_state", None)
    update_state = getattr(runtime_agent, "update_state", None)
    if not callable(get_state) or not callable(update_state):
        return False
    try:
        state = get_state(config)
        values = getattr(state, "values", None) or {}
        history_messages = values.get("messages", []) or []
        sanitized_messages, changed = _sanitize_persisted_messages_for_provider(history_messages)
        if not changed:
            return False
        thread_id = str(((config or {}).get("configurable") or {}).get("thread_id") or "").strip()
        if thread_id:
            delete_conversation_state(thread_id)
        if not sanitized_messages:
            return True
        try:
            update_state(config, {"messages": sanitized_messages})
        except TypeError:
            update_state({"messages": sanitized_messages}, config=config)
        return True
    except Exception:
        # Never block chat request on repair path.
        return False


def _semantic_messages_only(messages) -> list[dict[str, str]]:
    """Return only user/assistant messages in normalized dict shape."""
    normalized: list[dict[str, str]] = []
    for m in messages or []:
        role_raw = None
        content_raw = ""
        if isinstance(m, dict):
            role_raw = m.get("role") or m.get("type")
            content_raw = m.get("content") or ""
        else:
            role_raw = getattr(m, "role", None) or getattr(m, "type", None)
            content_raw = getattr(m, "content", "") or ""
            if not role_raw and hasattr(m, "__class__"):
                name = m.__class__.__name__.lower()
                if "ai" in name or "assistant" in name:
                    role_raw = "assistant"
                elif "human" in name or "user" in name:
                    role_raw = "user"

        role = str(role_raw or "").strip().lower()
        if role not in {"assistant", "ai", "user", "human"}:
            continue
        content = _plain_text_from_message_content(content_raw)
        if not content:
            continue
        normalized.append(
            {
                "role": "assistant" if role in {"assistant", "ai"} else "user",
                "content": content,
            }
        )
    return normalized


def _summarize_messages(
    messages: list[dict[str, str]],
    model_name: str | None,
) -> str:
    """
    Summarize older dialog turns with a direct LLM summarization prompt.
    Returns empty string on failure (best-effort path).
    """
    if not messages:
        return ""
    chosen_model = (model_name or get_active_model_name() or "").strip()
    if not chosen_model:
        return ""
    try:
        llm = init_chat_model(
            model=chosen_model,
            temperature=0.0,
            max_tokens=700,
            timeout=60,
        )
        max_chars = max(1200, RAG_HISTORY_SUMMARY_MAX_TOKEN_LIMIT * 4)
        transcript_lines: list[str] = []
        used_chars = 0
        for msg in messages:
            role = "Assistant" if str(msg.get("role") or "").lower() == "assistant" else "User"
            content = str(msg.get("content") or "").strip()
            if not content:
                continue
            line = f"{role}: {content}"
            if used_chars + len(line) > max_chars:
                remaining = max_chars - used_chars
                if remaining <= 0:
                    break
                line = line[:remaining]
            transcript_lines.append(line)
            used_chars += len(line)
            if used_chars >= max_chars:
                break
        transcript = "\n".join(transcript_lines).strip()
        if not transcript:
            return ""

        prompt = (
            "Summarize the older part of this conversation for future assistant turns. "
            "Keep it concise and factual. Capture user goals, constraints, decisions, "
            "preferences, and unresolved questions. Do not invent facts.\n\n"
            "Return plain text only."
        )
        resp = llm.invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(content=f"Conversation transcript:\n{transcript}"),
            ]
        )
        content = getattr(resp, "content", "")
        if isinstance(content, list):
            merged_parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if text:
                        merged_parts.append(str(text))
                elif part:
                    merged_parts.append(str(part))
            return "\n".join(merged_parts).strip()
        return str(content or "").strip()
    except Exception:
        logger.exception("History summarization failed")
        return ""


def _compact_conversation_history(runtime_agent, config: dict, model_name: str | None) -> bool:
    """
    Compact long history per conversation instead of dropping everything.
    Keeps latest turns and stores summary of older turns as a system message.
    """
    get_state = getattr(runtime_agent, "get_state", None)
    update_state = getattr(runtime_agent, "update_state", None)
    if not callable(get_state) or not callable(update_state) or RAG_MAX_HISTORY_MESSAGES <= 0:
        return False

    state = get_state(config)
    values = getattr(state, "values", None) or {}
    history_messages = values.get("messages", []) or []
    semantic_messages = _semantic_messages_only(history_messages)
    if len(semantic_messages) <= RAG_MAX_HISTORY_MESSAGES:
        return False

    keep_last = max(2, min(RAG_HISTORY_KEEP_LAST_MESSAGES, RAG_MAX_HISTORY_MESSAGES))
    recent_turns = semantic_messages[-keep_last:]
    older_turns = semantic_messages[:-keep_last]
    summary = _summarize_messages(older_turns, model_name=model_name)

    thread_id = str(((config or {}).get("configurable") or {}).get("thread_id") or "").strip()
    if thread_id:
        delete_conversation_state(thread_id)

    seed_messages: list[dict[str, str]] = []
    if summary:
        seed_messages.append(
            {
                "role": "system",
                "content": (
                    "Conversation summary (older turns):\n"
                    f"{summary}\n\n"
                    "Use this summary as prior context, then rely on the explicit recent turns below."
                ),
            }
        )
    seed_messages.extend(recent_turns)
    if not seed_messages:
        return False

    try:
        update_state(config, {"messages": seed_messages})
    except TypeError:
        update_state({"messages": seed_messages}, config=config)
    return True


def _ensure_assistant_turn_persisted(runtime_agent, config: dict, content: str) -> None:
    """
    Best-effort guard: some tool-heavy runs may not persist final assistant text
    in `messages` history. Append one assistant message if missing.
    """
    text = str(content or "").strip()
    if not text:
        return

    get_state = getattr(runtime_agent, "get_state", None)
    update_state = getattr(runtime_agent, "update_state", None)
    if not callable(update_state):
        return

    def _role_of(m) -> str:
        if isinstance(m, dict):
            return str(m.get("role") or m.get("type") or "").lower()
        return str(getattr(m, "role", None) or getattr(m, "type", None) or "").lower()

    def _content_of(m):
        if isinstance(m, dict):
            return m.get("content") or ""
        return getattr(m, "content", "") or ""

    try:
        if callable(get_state):
            state = get_state(config)
            values = getattr(state, "values", None) or {}
            messages = values.get("messages", []) or []
            # The agent already persists its own assistant message(s) for the turn.
            # `text` here is the streamed `final_content`, i.e. the CONCATENATION of
            # every text delta across a multi-step run (intermediate "let me check…"
            # narration + the final answer), so it equals no single persisted turn —
            # appending it adds a giant message that repeats the whole turn on refresh.
            # Only append when the agent persisted NO assistant text since the last
            # user turn (the genuine "nothing got saved" safety net this guard exists
            # for); otherwise the answer is already in history, so do nothing.
            last_user_idx = -1
            for i, msg in enumerate(messages):
                if _role_of(msg) in {"user", "human"}:
                    last_user_idx = i
            for msg in messages[last_user_idx + 1:]:
                if _role_of(msg) in {"assistant", "ai"} and _unwrap_response_content(_content_of(msg)).strip():
                    return
        # Nothing assistant-side was persisted for this turn — add the answer.
        try:
            update_state(config, {"messages": [{"role": "assistant", "content": text}]})
        except TypeError:
            # Compatibility fallback for other signatures.
            update_state({"messages": [{"role": "assistant", "content": text}]}, config=config)
    except Exception:
        # Best-effort only; never break chat response.
        return


@app.post("/auth/register", response_model=AuthResponse)
@limiter.limit(RAG_RATE_LIMIT_REGISTER)
def register(request: Request, body: RegisterRequest):
    """Create account; returns token and username. thread_id = username so history is per user."""
    ok, result = auth_register(body.username.strip(), body.password)
    if not ok:
        raise HTTPException(status_code=400, detail=result)
    username = resolve_token(result) or body.username.strip()
    flags = get_user_auth_flags(username)
    role = get_user_role(username)
    return AuthResponse(token=result, username=username, role=role, must_change_password=bool(flags.get("must_change_password")))


@app.post("/auth/login", response_model=AuthResponse)
@limiter.limit(RAG_RATE_LIMIT_LOGIN)
def login(request: Request, body: LoginRequest):
    """Log in; returns token and username."""
    ok, result = auth_login(body.username.strip(), body.password)
    if not ok:
        raise HTTPException(status_code=401, detail=result)
    username = resolve_token(result) or body.username.strip()
    flags = get_user_auth_flags(username)
    role = get_user_role(username)
    return AuthResponse(token=result, username=username, role=role, must_change_password=bool(flags.get("must_change_password")))


@app.post("/auth/logout")
def logout(authorization: str | None = Header(default=None)):
    """Invalidate the current bearer token (server-side session row when using PostgreSQL)."""
    if authorization and authorization.startswith("Bearer "):
        invalidate_token(authorization[7:].strip())
    return {"ok": True}


@app.post("/auth/password/change", response_model=AuthResponse)
def password_change(
    body: PasswordChangeRequest,
    authorization: str | None = Header(default=None),
):
    """Change password for current user (forced on first login or optional from settings)."""
    username = _get_username(authorization, enforce_password_rotation=False)
    ok, result = auth_change_password(
        username=username,
        current_password=body.current_password,
        new_password=body.new_password,
        repeat_password=body.repeat_password,
    )
    if not ok:
        raise HTTPException(status_code=400, detail=result)

    # Revoke current bearer token too; return a fresh token from password-change flow.
    if authorization and authorization.startswith("Bearer "):
        invalidate_token(authorization[7:].strip())

    new_token = result
    canonical_username = resolve_token(new_token) or username
    role = get_user_role(canonical_username)
    flags = get_user_auth_flags(canonical_username)
    return AuthResponse(
        token=new_token,
        username=canonical_username,
        role=role,
        must_change_password=bool(flags.get("must_change_password")),
    )


@app.get("/auth/me")
def me(authorization: str | None = Header(default=None)):
    """Return current user and role if token valid."""
    username = _get_username(authorization, enforce_password_rotation=False)
    flags = get_user_auth_flags(username)
    return {
        "username": username,
        "role": get_user_role(username),
        "must_change_password": bool(flags.get("must_change_password")),
    }


# ── Long-term user memory ────────────────────────────────────────────────────
def _resolve_user_id_or_404(username: str):
    """Resolve a username to its user id, or raise 404."""
    uid = get_user_id(username)
    if uid is None:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    return uid


def _add_memory_or_error(uid, content: str, category: str, source: str) -> dict:
    """Shared add path that maps the store status to HTTP errors."""
    res = user_memory.add_memory(uid, content, category, source=source)
    status = res.get("status")
    if status == "full":
        raise HTTPException(
            status_code=409,
            detail=f"Достигнут лимит памяти ({RAG_MAX_USER_MEMORIES}). Удалите ненужные записи.",
        )
    if status == "invalid":
        raise HTTPException(status_code=400, detail="Пустой или некорректный текст памяти")
    return res["memory"]


@app.get("/memories")
def memories_list(authorization: str | None = Header(default=None)):
    """List the current user's long-term memories."""
    username = _get_username(authorization)
    uid = _resolve_user_id_or_404(username)
    return {"memories": user_memory.list_memories(uid)}


@app.post("/memories")
def memories_add(body: MemoryCreate, authorization: str | None = Header(default=None)):
    """Add a memory manually (source=user)."""
    username = _get_username(authorization)
    uid = _resolve_user_id_or_404(username)
    memory = _add_memory_or_error(uid, body.content, body.category, source="user")
    return {"ok": True, "memory": memory}


@app.patch("/memories/{handle}")
def memories_update(handle: str, body: MemoryUpdate, authorization: str | None = Header(default=None)):
    """Edit one of the current user's memories."""
    username = _get_username(authorization)
    uid = _resolve_user_id_or_404(username)
    memory = user_memory.update_memory(uid, handle, body.content)
    if memory is None:
        raise HTTPException(status_code=404, detail="Запись памяти не найдена")
    return {"ok": True, "memory": memory}


@app.delete("/memories/{handle}")
def memories_delete(handle: str, authorization: str | None = Header(default=None)):
    """Delete one of the current user's memories."""
    username = _get_username(authorization)
    uid = _resolve_user_id_or_404(username)
    if not user_memory.delete_memory(uid, handle):
        raise HTTPException(status_code=404, detail="Запись памяти не найдена")
    return {"ok": True}


@app.delete("/memories")
def memories_clear(authorization: str | None = Header(default=None)):
    """Delete all of the current user's memories."""
    username = _get_username(authorization)
    uid = _resolve_user_id_or_404(username)
    return {"ok": True, "deleted": user_memory.clear_memories(uid)}


@app.get("/me/memory-settings")
def memory_settings_get(authorization: str | None = Header(default=None)):
    """Return the current user's memory toggle and the global kill-switch state."""
    username = _get_username(authorization)
    uid = _resolve_user_id_or_404(username)
    return {
        "enabled": user_memory.get_memory_enabled(uid),
        "globally_enabled": memory_enabled(),
    }


@app.put("/me/memory-settings")
def memory_settings_put(body: MemorySettingsUpdate, authorization: str | None = Header(default=None)):
    """Turn the current user's long-term memory on/off."""
    username = _get_username(authorization)
    uid = _resolve_user_id_or_404(username)
    return {
        "enabled": user_memory.set_memory_enabled(uid, body.enabled),
        "globally_enabled": memory_enabled(),
    }


@app.get("/admin/users/{target_username}/memories")
def admin_memories_list(target_username: str, authorization: str | None = Header(default=None)):
    """Admin: view any user's memories."""
    _require_admin(authorization)
    uid = _resolve_user_id_or_404(target_username)
    return {
        "username": target_username,
        "enabled": user_memory.get_memory_enabled(uid),
        "memories": user_memory.list_memories(uid),
    }


@app.post("/admin/users/{target_username}/memories")
def admin_memories_add(
    target_username: str,
    body: MemoryCreate,
    request: Request,
    authorization: str | None = Header(default=None),
):
    """Admin: add a memory to a user (source=admin)."""
    admin_username = _require_admin(authorization)
    uid = _resolve_user_id_or_404(target_username)
    memory = _add_memory_or_error(uid, body.content, body.category, source="admin")
    write_audit(
        "memory.admin_add",
        admin_username,
        target=target_username,
        details={"memory_id": memory["id"], "category": memory["category"]},
        ip_address=request.client.host if request.client else "",
    )
    return {"ok": True, "memory": memory}


@app.patch("/admin/users/{target_username}/memories/{handle}")
def admin_memories_update(
    target_username: str,
    handle: str,
    body: MemoryUpdate,
    request: Request,
    authorization: str | None = Header(default=None),
):
    """Admin: edit a user's memory."""
    admin_username = _require_admin(authorization)
    uid = _resolve_user_id_or_404(target_username)
    memory = user_memory.update_memory(uid, handle, body.content)
    if memory is None:
        raise HTTPException(status_code=404, detail="Запись памяти не найдена")
    write_audit(
        "memory.admin_update",
        admin_username,
        target=target_username,
        details={"memory_id": handle},
        ip_address=request.client.host if request.client else "",
    )
    return {"ok": True, "memory": memory}


@app.delete("/admin/users/{target_username}/memories/{handle}")
def admin_memories_delete(
    target_username: str,
    handle: str,
    request: Request,
    authorization: str | None = Header(default=None),
):
    """Admin: delete a user's memory."""
    admin_username = _require_admin(authorization)
    uid = _resolve_user_id_or_404(target_username)
    if not user_memory.delete_memory(uid, handle):
        raise HTTPException(status_code=404, detail="Запись памяти не найдена")
    write_audit(
        "memory.admin_delete",
        admin_username,
        target=target_username,
        details={"memory_id": handle},
        ip_address=request.client.host if request.client else "",
    )
    return {"ok": True}


@app.post("/admin/users/provision")
def admin_user_provision(
    request: Request,
    body: AdminProvisionUserRequest,
    authorization: str | None = Header(default=None),
):
    """
    Create employee account with a random temporary password.
    User must change password on first login.
    """
    admin_username = _require_admin(authorization)
    new_username = body.username.strip()
    role = body.role.strip().lower()
    ok, result = provision_user_with_temp_password(
        created_by_username=admin_username,
        username=new_username,
        role=role,
    )
    if not ok:
        raise HTTPException(status_code=400, detail=result)
    write_audit(
        "provision_user",
        admin_username,
        target=new_username,
        details={"role": role},
        ip_address=request.client.host if request.client else "",
    )
    return {"ok": True, "user": result}


# ── Client portal (external clients: invites, cabinet data) ──────────────────

class InviteCreateRequest(BaseModel):
    project_ids: list[int] = Field(..., min_length=1)
    project_names: list[str] = Field(default_factory=list)
    company_name: str = Field(default="", max_length=255)
    max_uses: int = Field(default=1, ge=1, le=100)
    expires_days: int | None = Field(default=None, ge=1, le=365)


class ClientRegisterRequest(BaseModel):
    invite_token: str = Field(..., min_length=8, max_length=64)
    email: str = Field(..., min_length=5, max_length=RAG_USERNAME_MAX_LEN)
    password: str = Field(..., min_length=CLIENT_MIN_PASSWORD_LENGTH, max_length=RAG_MAX_PASSWORD_LENGTH)


class ClientChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    conversation_id: str = Field(default="default", max_length=64)


class ClientEmailChangeRequest(BaseModel):
    new_email: str = Field(..., min_length=5, max_length=RAG_USERNAME_MAX_LEN)
    password: str = Field(..., min_length=1, max_length=RAG_MAX_PASSWORD_LENGTH)


def _require_client(authorization: str | None = Header(default=None)) -> str:
    """Require a logged-in client (admins allowed too, for testing the cabinet)."""
    username = _get_username(authorization)
    if get_user_role(username) not in ("client", "admin"):
        raise HTTPException(status_code=403, detail="Доступ только для клиентов")
    return username


def _client_portal_or_503() -> None:
    if not client_portal_enabled():
        raise HTTPException(status_code=503, detail="Клиентский портал не настроен")


@app.get("/admin/orlanda/projects")
async def admin_orlanda_projects(authorization: str | None = Header(default=None)):
    """OrlandaBot project list for the invite-creation picker."""
    _require_admin(authorization)
    _client_portal_or_503()
    try:
        return {"projects": await client_portal.orlanda_all_projects()}
    except client_portal.OrlandaApiError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/admin/orlanda/customers")
async def admin_orlanda_customers(authorization: str | None = Header(default=None)):
    """Customer directory for client-first project selection in the invite UI."""
    _require_admin(authorization)
    _client_portal_or_503()
    try:
        return {"customers": await client_portal.orlanda_customers()}
    except client_portal.OrlandaApiError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/admin/invites")
def admin_invite_create(
    request: Request,
    body: InviteCreateRequest,
    authorization: str | None = Header(default=None),
):
    admin_username = _require_admin(authorization)
    _client_portal_or_503()
    invite = client_portal.create_invite(
        created_by=admin_username,
        project_ids=body.project_ids,
        project_names=body.project_names,
        company_name=body.company_name,
        max_uses=body.max_uses,
        expires_days=body.expires_days,
    )
    base = RAG_FRONTEND_BASE_URL or ""
    invite["url"] = f"{base}/register?invite={invite['token']}"
    write_audit(
        "client_invite_create",
        admin_username,
        target=invite["token"],
        details={"project_ids": body.project_ids, "company": body.company_name},
        ip_address=request.client.host if request.client else "",
    )
    return {"invite": invite}


@app.get("/admin/invites")
def admin_invite_list(authorization: str | None = Header(default=None)):
    _require_admin(authorization)
    base = RAG_FRONTEND_BASE_URL or ""
    invites = client_portal.list_invites()
    for inv in invites:
        inv["url"] = f"{base}/register?invite={inv['token']}"
    return {"invites": invites}


@app.delete("/admin/invites/{token}")
def admin_invite_delete(
    request: Request,
    token: str,
    authorization: str | None = Header(default=None),
):
    admin_username = _require_admin(authorization)
    if not client_portal.delete_invite(token):
        raise HTTPException(status_code=404, detail="Приглашение не найдено")
    write_audit(
        "client_invite_delete",
        admin_username,
        target=token,
        ip_address=request.client.host if request.client else "",
    )
    return {"ok": True}


@app.get("/invites/{token}")
def invite_preview(token: str):
    """Public invite preview for the registration page (no auth)."""
    return client_portal.check_invite(token)


@app.post("/auth/register-client", response_model=AuthResponse)
@limiter.limit(RAG_RATE_LIMIT_CLIENT_REGISTER)
async def register_client_account(request: Request, body: ClientRegisterRequest):
    """Register an external client through an invite link."""
    _client_portal_or_503()
    ok, result = await client_portal.register_client_via_invite(
        body.invite_token, body.email, body.password
    )
    if not ok:
        raise HTTPException(status_code=400, detail=result)
    username = resolve_token(result) or body.email.strip().lower()
    return AuthResponse(token=result, username=username, role="client")


@app.get("/client/portal/tasks-table")
async def client_portal_tasks_table(authorization: str | None = Header(default=None)):
    username = _require_client(authorization)
    _client_portal_or_503()
    try:
        return await client_portal.orlanda_tasks_table(username)
    except client_portal.OrlandaApiError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/client/portal/chat")
@limiter.limit(RAG_RATE_LIMIT_CHAT)
async def client_portal_chat(
    request: Request,
    body: ClientChatRequest,
    authorization: str | None = Header(default=None),
):
    username = _require_client(authorization)
    _client_portal_or_503()
    try:
        return await client_portal.orlanda_chat(username, body.message, body.conversation_id)
    except client_portal.OrlandaApiError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/client/portal/chat/reset")
async def client_portal_chat_reset(
    authorization: str | None = Header(default=None),
    conversation_id: str = "default",
):
    username = _require_client(authorization)
    _client_portal_or_503()
    try:
        await client_portal.orlanda_chat_reset(username, conversation_id)
    except client_portal.OrlandaApiError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return {"ok": True}


@app.get("/client/portal/chats")
async def client_portal_chats(authorization: str | None = Header(default=None)):
    username = _require_client(authorization)
    _client_portal_or_503()
    try:
        return {"conversations": await client_portal.orlanda_conversations(username)}
    except client_portal.OrlandaApiError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/client/portal/chat/history")
async def client_portal_chat_history(
    conversation_id: str,
    authorization: str | None = Header(default=None),
):
    username = _require_client(authorization)
    _client_portal_or_503()
    try:
        return {"messages": await client_portal.orlanda_chat_history(username, conversation_id)}
    except client_portal.OrlandaApiError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.delete("/client/portal/chat")
async def client_portal_chat_delete(
    conversation_id: str,
    authorization: str | None = Header(default=None),
):
    username = _require_client(authorization)
    _client_portal_or_503()
    try:
        await client_portal.orlanda_delete_conversation(username, conversation_id)
    except client_portal.OrlandaApiError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return {"ok": True}


@app.get("/client/portal/progress")
async def client_portal_progress(authorization: str | None = Header(default=None)):
    username = _require_client(authorization)
    _client_portal_or_503()
    try:
        return {"projects": await client_portal.progress_links(username)}
    except client_portal.OrlandaApiError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/client/portal/feedback")
async def client_portal_feedback(authorization: str | None = Header(default=None)):
    username = _require_client(authorization)
    _client_portal_or_503()
    try:
        return {"links": await client_portal.orlanda_feedback_links(username)}
    except client_portal.OrlandaApiError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/client/portal/email", response_model=AuthResponse)
async def client_portal_email_change(
    body: ClientEmailChangeRequest,
    authorization: str | None = Header(default=None),
):
    """Change a client's e-mail (their login identity).

    The client's e-mail is also their identity in orlanda-api (ProjectMember ACL +
    Redis assistant history), so the mirror row is renamed there first. Everything
    that can fail on validity/credentials is checked with a dry run *before* the
    orlanda-api call; if the real rename on this side still fails afterwards
    (e.g. a race), the orlanda-side rename is rolled back so the two systems never
    drift out of sync. Admin accounts (used to test the cabinet) are rejected —
    renaming them would touch their real employee login.
    """
    username = _require_client(authorization)
    if get_user_role(username) == "admin":
        raise HTTPException(status_code=403, detail="Только для клиентских аккаунтов")
    _client_portal_or_503()

    new_email = body.new_email.strip().lower()

    ok, err = auth_change_client_email(username, new_email, body.password, dry_run=True)
    if not ok:
        raise HTTPException(status_code=400, detail=err)

    try:
        await client_portal.orlanda_rename(username, new_email)
    except client_portal.OrlandaApiError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    ok, result = auth_change_client_email(username, new_email, body.password)
    if not ok:
        try:
            await client_portal.orlanda_rename(new_email, username)
        except client_portal.OrlandaApiError:
            logger.error("Rollback of orlanda-api rename %s -> %s failed", new_email, username)
        raise HTTPException(status_code=400, detail=result)

    return AuthResponse(token=result, username=new_email, role="client")


# ── Admin: client accounts & their project access ────────────────────────────

class ClientAccessUpdateRequest(BaseModel):
    project_ids: list[int] = Field(default_factory=list)


@app.get("/admin/clients")
def admin_clients_list(authorization: str | None = Header(default=None)):
    _require_admin(authorization)
    return {"clients": client_portal.list_client_accounts()}


@app.get("/admin/clients/{client_username}/access")
async def admin_client_access_get(
    client_username: str,
    authorization: str | None = Header(default=None),
):
    _require_admin(authorization)
    _client_portal_or_503()
    try:
        projects = await client_portal.orlanda_get_access(client_username)
    except client_portal.OrlandaApiError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return {"username": client_username, "projects": projects}


@app.post("/admin/clients/{client_username}/reset-password")
def admin_client_reset_password(
    request: Request,
    client_username: str,
    authorization: str | None = Header(default=None),
):
    """Set a fresh random password for a client account (admin hands it over)."""
    admin_username = _require_admin(authorization)
    from rag_agent.auth import reset_client_password

    ok, result = reset_client_password(client_username)
    if not ok:
        raise HTTPException(status_code=404, detail=result)
    write_audit(
        "client_password_reset",
        admin_username,
        target=client_username,
        ip_address=request.client.host if request.client else "",
    )
    return {"username": client_username, "temporary_password": result}


@app.put("/admin/clients/{client_username}/access")
async def admin_client_access_put(
    request: Request,
    client_username: str,
    body: ClientAccessUpdateRequest,
    authorization: str | None = Header(default=None),
):
    admin_username = _require_admin(authorization)
    _client_portal_or_503()
    try:
        project_ids = await client_portal.orlanda_sync_access(client_username, body.project_ids)
    except client_portal.OrlandaApiError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    write_audit(
        "client_access_update",
        admin_username,
        target=client_username,
        details={"project_ids": project_ids},
        ip_address=request.client.host if request.client else "",
    )
    return {"username": client_username, "project_ids": project_ids}


# ── monday.com integration (per-user OAuth + remote MCP) ─────────────────────
def _monday_settings_redirect(status: str, reason: str | None = None) -> RedirectResponse:
    """302 back to the frontend settings page carrying a monday status flag.

    Uses RAG_FRONTEND_BASE_URL when set (split dev/prod hosts), else a same-origin
    relative path. This is a browser redirect target, not an API response.
    """
    params = {"monday": status}
    if reason:
        params["reason"] = reason
    base = RAG_FRONTEND_BASE_URL or ""
    return RedirectResponse(url=f"{base}/settings?{urlencode(params)}", status_code=302)


@app.get("/integrations/monday/authorize")
def monday_authorize(authorization: str | None = Header(default=None)):
    """Return the monday consent URL for the logged-in user to open in the browser."""
    username = _get_username(authorization)
    if not monday_enabled():
        raise HTTPException(status_code=503, detail="Интеграция monday.com не настроена")
    return {"authorize_url": monday_build_authorize_url(username)}


@app.get("/auth/monday/callback")
def monday_callback(
    request: Request,
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
):
    """OAuth redirect target: validate state, exchange code, store token, bounce to UI.

    The user's identity comes from the signed `state` (a browser redirect carries no
    Authorization header). All failure paths redirect back with `?monday=error&reason=...`.
    """
    if not monday_enabled():
        return _monday_settings_redirect("error", "not_configured")
    if error:
        return _monday_settings_redirect("error", error)
    username = monday_verify_state(state)
    if not username:
        return _monday_settings_redirect("error", "invalid_state")
    if not code:
        return _monday_settings_redirect("error", "missing_code")
    try:
        token_data = monday_exchange_code_for_token(code)
    except Exception:
        logger.exception("monday token exchange failed")
        return _monday_settings_redirect("error", "token_exchange_failed")
    access_token = (token_data or {}).get("access_token")
    if not access_token:
        return _monday_settings_redirect("error", "no_token")
    identity = monday_fetch_identity(access_token)
    stored = monday_store_token(
        username,
        access_token,
        scope=token_data.get("scope", ""),
        token_type=token_data.get("token_type", "Bearer"),
        account_id=identity.get("account_id"),
        user_name=identity.get("name"),
    )
    if not stored:
        return _monday_settings_redirect("error", "store_failed")
    write_audit(
        "monday_connect",
        username,
        target="monday",
        details={"scope": token_data.get("scope", "")},
        ip_address=request.client.host if request.client else "",
    )
    return _monday_settings_redirect("connected")


@app.get("/integrations/monday/status")
def monday_status(authorization: str | None = Header(default=None)):
    """Return whether the logged-in user has connected their monday account."""
    username = _get_username(authorization)
    status = monday_get_connection_status(username)
    status["enabled"] = monday_enabled()
    return status


@app.delete("/integrations/monday")
def monday_disconnect(request: Request, authorization: str | None = Header(default=None)):
    """Delete the logged-in user's stored monday token (disconnect)."""
    username = _get_username(authorization)
    disconnected = monday_delete_token(username)
    if disconnected:
        write_audit(
            "monday_disconnect",
            username,
            target="monday",
            ip_address=request.client.host if request.client else "",
        )
    return {"ok": True, "disconnected": disconnected}


def _unwrap_response_content(content) -> str:
    """Strip the structured `{"response_content": "..."}` wrapper down to plain text.

    The agent persists assistant content as a JSON string `{"response_content": ...}`.
    This is the same normalization `_messages_to_history` applies on read, factored
    out so callers (e.g. the persistence guard) can compare persisted turns as plain
    text. Non-wrapped content is returned as a stripped string unchanged.
    """
    if isinstance(content, list):
        # Multi-block content: join text parts the same way history rendering does.
        content = " ".join(
            (c.get("text", "") if isinstance(c, dict) else str(c)) for c in content
        )
    text = str(content or "")
    stripped = text.strip()
    if stripped.startswith("{") and "response_content" in stripped:
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict) and "response_content" in parsed:
                return str(parsed["response_content"] or "").strip()
        except (json.JSONDecodeError, TypeError):
            pass
    return stripped


def _messages_to_history(messages) -> list[dict]:
    """Convert agent state messages to [{role, content}, ...] for frontend (user/assistant only)."""
    def _extract_response_content_fallback(payload) -> str:
        """Try to recover assistant text from structured payload shapes."""
        if payload is None:
            return ""
        if isinstance(payload, str):
            s = payload.strip()
            if s.startswith("{") and "response_content" in s:
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, dict):
                        return str(parsed.get("response_content") or "")
                except (json.JSONDecodeError, TypeError):
                    pass
            return ""

        if isinstance(payload, dict):
            # Most explicit shapes first.
            if isinstance(payload.get("response_content"), str):
                return payload.get("response_content") or ""
            parsed = payload.get("parsed")
            if isinstance(parsed, dict) and isinstance(parsed.get("response_content"), str):
                return parsed.get("response_content") or ""
            for key in ("data", "additional_kwargs", "kwargs", "value"):
                nested = payload.get(key)
                if nested is not None:
                    val = _extract_response_content_fallback(nested)
                    if val:
                        return val
        return ""

    def normalize_role(value: str | None) -> str:
        v = str(value or "").strip().lower()
        if v in {"ai", "assistant"}:
            return "assistant"
        if v in {"human", "user"}:
            return "user"
        # Fallback to user for unknown role labels.
        return "user"

    out = []
    for m in messages or []:
        if isinstance(m, dict):
            if m.get("type") in ("tool", "system") or m.get("role") in ("tool", "system"):
                continue
            role = normalize_role(m.get("role") or m.get("type", "user"))
            content = m.get("content") or m.get("data", {}).get("content", "") or ""
        else:
            if getattr(m, "type", None) in ("tool", "system"):
                continue
            content = getattr(m, "content", "") or ""
            if callable(content):
                content = ""
            role = getattr(m, "type", None) or getattr(m, "role", None)
            if not role and hasattr(m, "__class__"):
                name = m.__class__.__name__.lower()
                role = "assistant" if "ai" in name or "assistant" in name else "user"
            role = normalize_role(role)
            if not content:
                # Some structured-output assistant messages keep text in parsed/kwargs fields.
                content = (
                    _extract_response_content_fallback(getattr(m, "additional_kwargs", None))
                    or _extract_response_content_fallback(getattr(m, "kwargs", None))
                    or _extract_response_content_fallback(getattr(m, "data", None))
                )
        if isinstance(content, list):
            content = " ".join(
                (c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
            )
        if isinstance(content, dict):
            content = (
                _extract_response_content_fallback(content)
                or _extract_response_content_fallback(content.get("data"))
                or content.get("content", "")
                or ""
            )
        content = str(content)
        # Agent may store assistant reply as JSON with response_content; extract plain text for history
        if role == "assistant":
            content = _unwrap_response_content(content)
        if role == "assistant" and not str(content).strip():
            # Skip empty assistant placeholders/tool-call stubs in history UI.
            continue
        out.append({"role": role, "content": content})
    return out


@app.get("/chat/history")
def chat_history(
    authorization: str | None = Header(default=None),
    conversation_id: str | None = Query(default=None),
):
    """Return conversation history for the current user and selected conversation. Requires login."""
    username = _get_username(authorization)
    try:
        thread_id = _make_thread_id(username, conversation_id)
        config = {"configurable": {"thread_id": thread_id}}
        get_state = getattr(get_base_agent(), "get_state", None)
        if not get_state:
            return {"messages": []}
        state = get_state(config)
        values = getattr(state, "values", None) or {}
        messages = values.get("messages", [])
        return {"messages": _messages_to_history(messages)}
    except Exception as e:
        logger.exception("Failed to get chat history")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/conversations")
def chat_conversations(
    authorization: str | None = Header(default=None),
    max_conversations: int = Query(default=200, ge=1, le=2000),
    scan_checkpoints: int = Query(default=5000, ge=100, le=50000),
):
    """Return conversation list for current user, discovered from persisted checkpoints."""
    username = _get_username(authorization)
    cp = getattr(get_base_agent(), "checkpointer", None)
    list_fn = getattr(cp, "list", None) if cp is not None else None
    if not callable(list_fn):
        return {
            "conversations": [{"id": "default", "title": "Основной диалог"}],
            "warning": "Checkpointer does not support thread listing in this runtime.",
        }

    meta_map = list_conversation_meta_for_user(username)
    discovered: list[dict] = []
    seen_ids: set[str] = set()
    scanned = 0
    prefix = f"{username}:"
    for item in list_fn(None, limit=scan_checkpoints):
        scanned += 1
        conf = getattr(item, "config", None) or {}
        confg = conf.get("configurable", {}) if isinstance(conf, dict) else {}
        thread_id = str(confg.get("thread_id") or "").strip()
        if not thread_id:
            continue
        thread_username, conv_id = _parse_thread_id(thread_id)
        if thread_username != username:
            continue
        if thread_id.startswith(prefix) and conv_id in seen_ids:
            continue
        seen_ids.add(conv_id)
        stored = meta_map.get(conv_id) or {}
        discovered.append(
            {
                "id": conv_id,
                "title": str(stored.get("title") or ("Основной диалог" if conv_id == "default" else conv_id)),
                "last_activity_ts": str(
                    getattr(item, "checkpoint", {}).get("ts", "")
                    or stored.get("updated_at")
                    or ""
                ),
            }
        )
        if len(discovered) >= max_conversations:
            break

    for conv_id, stored in meta_map.items():
        if conv_id in seen_ids:
            continue
        seen_ids.add(conv_id)
        discovered.append(
            {
                "id": conv_id,
                "title": str(stored.get("title") or conv_id),
                "last_activity_ts": str(stored.get("updated_at") or stored.get("created_at") or ""),
            }
        )
        if len(discovered) >= max_conversations:
            break

    if "default" not in seen_ids:
        stored_default = meta_map.get("default") or {}
        discovered.insert(
            0,
            {
                "id": "default",
                "title": str(stored_default.get("title") or "Основной диалог"),
                "last_activity_ts": str(stored_default.get("updated_at") or ""),
            },
        )
    return {
        "conversations": discovered,
        "scanned_checkpoints": scanned,
    }


@app.post("/chat/conversations")
def create_chat_conversation(
    body: ConversationCreateRequest,
    authorization: str | None = Header(default=None),
):
    """Create or upsert one conversation title for current user."""
    username = _get_username(authorization)
    conv_id = (body.id or "").strip()
    title = (body.title or "").strip()
    if not conv_id:
        raise HTTPException(status_code=400, detail="Conversation id is required")
    if not title:
        raise HTTPException(status_code=400, detail="Conversation title is required")
    saved = upsert_conversation_meta_for_user(username, conv_id, title)
    return {
        "conversation": {
            "id": saved["id"],
            "title": saved["title"],
            "last_activity_ts": saved["updated_at"],
        }
    }


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(RAG_RATE_LIMIT_CHAT)
def chat(
    request: Request,
    body: ChatRequest,
    authorization: str | None = Header(default=None),
    conversation_id: str | None = Query(default=None),
):
    """
    Send a message; agent reply + sources. Requires login.
    thread_id = your username, so when you come back you get your conversation history (if CHECKPOINT_DB is set).
    """
    username = _get_username(authorization)
    user_message = (body.message or "").strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Сообщение не должно быть пустым")
    if RAG_MAX_USER_MESSAGE_CHARS > 0 and len(user_message) > RAG_MAX_USER_MESSAGE_CHARS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Сообщение слишком длинное ({len(user_message)} символов). "
                f"Максимум: {RAG_MAX_USER_MESSAGE_CHARS}."
            ),
        )
    try:
        run_id = f"chat_{int(time.time() * 1000)}"
        tool_events: list[dict] = []
        def on_tool_event(event: dict):
            # Keep event payload explicit and safe for UI.
            tool_events.append(
                {
                    "source": str(event.get("source") or ""),
                    "tool_name": str(event.get("tool_name") or ""),
                    "status": str(event.get("status") or ""),
                    "message": str(event.get("message") or ""),
                    "ts": int(event.get("ts") or 0),
                }
            )

        # NOTE: monday tools are intentionally NOT added here. They are MCP (async-only) tools
        # and this sync endpoint executes tools synchronously via runtime_agent.invoke(), which
        # cannot drive a coroutine-only tool. The UI uses /chat/stream (async), which injects
        # them. A programmatic caller of /chat gets the RAG-only agent.
        selected_model_name: str | None = None

        thread_id = _make_thread_id(username, conversation_id)
        # Long-term memory tools ARE sync, so (unlike monday) they work on this path too.
        user_id = get_user_id(username)
        extra_tools = list(get_memory_tools_for_user(user_id, thread_id=thread_id)) if user_id else []
        memory_suffix = user_memory.build_memory_block(user_id) if user_id else None
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": RAG_MAX_AGENT_RECURSION_LIMIT,
        }
        runtime_agent = build_agent(
            extra_tools=extra_tools,
            model_name=selected_model_name,
            system_prompt_suffix=memory_suffix,
        )
        repaired = _repair_conversation_history_for_provider(runtime_agent, config)
        if repaired:
            on_tool_event(
                {
                    "source": "system",
                    "tool_name": "history_repair",
                    "status": "success",
                    "message": "Conversation history was repaired to remove invalid tool protocol blocks.",
                    "ts": int(time.time() * 1000),
                }
            )
        # Prevent unlimited growth of persisted thread context, which can trigger
        # strict provider TPM limits (especially on Anthropic plans).
        if RAG_MAX_HISTORY_MESSAGES > 0:
            try:
                if _compact_conversation_history(runtime_agent, config, model_name=selected_model_name):
                    on_tool_event(
                        {
                            "source": "system",
                            "tool_name": "history_guard",
                            "status": "success",
                            "message": (
                                "Conversation history was compacted: older turns summarized, recent turns kept."
                            ),
                            "ts": int(time.time() * 1000),
                        }
                    )
            except Exception:
                # Best-effort only; chat should continue even if introspection fails.
                pass
        response = runtime_agent.invoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config=config,
            context=Context(user_id=username),
        )
        content = _extract_agent_response_text(response)
        if not content:
            raise ValueError("Model returned empty response content")
        _ensure_assistant_turn_persisted(runtime_agent, config, content)
        sources = get_last_sources()
        log_append(username=username, question=user_message, answer=content, sources=sources)
        return ChatResponse(response=content, sources=sources, tool_events=tool_events)
    except Exception as e:
        if _is_structured_output_validation_error(e):
            try:
                on_tool_event(
                    {
                        "source": "system",
                        "tool_name": "structured_output_retry",
                        "status": "start",
                        "message": "Structured output validation failed, retrying in unstructured mode",
                        "ts": int(time.time() * 1000),
                    }
                )
                unstructured_agent = build_agent(
                    extra_tools=extra_tools,
                    model_name=selected_model_name,
                    use_response_format=False,
                    system_prompt_suffix=memory_suffix,
                )
                retry_response = unstructured_agent.invoke(
                    {"messages": [{"role": "user", "content": user_message}]},
                    config=config,
                    context=Context(user_id=username),
                )
                content = _extract_agent_response_text(retry_response)
                if not content:
                    raise ValueError("Unstructured retry returned empty response content")
                _ensure_assistant_turn_persisted(unstructured_agent, config, content)
                sources = get_last_sources()
                on_tool_event(
                    {
                        "source": "system",
                        "tool_name": "structured_output_retry",
                        "status": "success",
                        "message": "Recovered response via unstructured retry path",
                        "ts": int(time.time() * 1000),
                    }
                )
                log_append(username=username, question=user_message, answer=content, sources=sources)
                return ChatResponse(response=content, sources=sources, tool_events=tool_events)
            except Exception as structured_retry_error:
                e = structured_retry_error
        if (
            (_is_rate_limit_error(e) or _is_provider_overloaded_error(e))
            and RAG_ENABLE_RATE_LIMIT_FALLBACK
            and RAG_FALLBACK_MODEL
        ):
            try:
                primary_failure_reason = (
                    "Rate limit" if _is_rate_limit_error(e) else "Provider overloaded"
                )
                on_tool_event(
                    {
                        "source": "system",
                        "tool_name": "fallback_model",
                        "status": "start",
                        "message": f"{primary_failure_reason} on primary model, retrying on {RAG_FALLBACK_MODEL}",
                        "ts": int(time.time() * 1000),
                    }
                )
                fallback_agent = build_agent(
                    extra_tools=extra_tools,
                    model_name=RAG_FALLBACK_MODEL,
                    system_prompt_suffix=memory_suffix,
                )
                response = fallback_agent.invoke(
                    {"messages": [{"role": "user", "content": user_message}]},
                    config=config,
                    context=Context(user_id=username),
                )
                content = _extract_agent_response_text(response)
                if not content:
                    raise ValueError("Fallback model returned empty response content")
                _ensure_assistant_turn_persisted(fallback_agent, config, content)
                sources = get_last_sources()
                on_tool_event(
                    {
                        "source": "system",
                        "tool_name": "fallback_model",
                        "status": "success",
                        "message": f"Fallback response generated by {RAG_FALLBACK_MODEL}",
                        "ts": int(time.time() * 1000),
                    }
                )
                log_append(username=username, question=user_message, answer=content, sources=sources)
                return ChatResponse(response=content, sources=sources, tool_events=tool_events)
            except Exception as fallback_error:
                e = fallback_error
        logger.exception("Chat request failed")
        log_append(
            username=username,
            question=user_message,
            answer="",
            sources=[],
            error=str(e),
        )
        if _is_rate_limit_error(e):
            raise HTTPException(
                status_code=429,
                detail=(
                    "Превышен лимит токенов провайдера модели. "
                    "Подождите 30–60 секунд и повторите запрос, "
                    "или уменьшите длину вопроса/контекста."
                ),
            )
        if _is_provider_overloaded_error(e):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Провайдер модели временно перегружен. "
                    "Подождите 10–30 секунд и повторите запрос."
                ),
            )
        if "graphrecursionerror" in str(type(e)).lower() or "recursion limit" in str(e).lower():
            return ChatResponse(
                response=(
                    "Не удалось завершить запрос: модель слишком долго не могла прийти к ответу. "
                    "Сформулируйте вопрос короче и конкретнее и повторите."
                ),
                sources=[],
                tool_events=locals().get("tool_events", []),
            )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
@limiter.limit(RAG_RATE_LIMIT_CHAT)
async def chat_stream(
    request: Request,
    body: ChatRequest,
    authorization: str | None = Header(default=None),
    conversation_id: str | None = Query(default=None),
):
    """Streaming version of /chat using Server-Sent Events.

    Emits SSE messages of the form `data: {json}\\n\\n` with one of these payloads:
      - {"type":"delta","text":"..."}        — chunk of assistant text
      - {"type":"tool_start","name":"..."}   — tool invocation began
      - {"type":"tool_end","name":"...","status":"success|error"}
      - {"type":"done","response":"...","sources":[...],"tool_events":[...]}
      - {"type":"error","message":"..."}     — fatal error; client should show toast
    """
    username = _get_username(authorization)
    user_message = (body.message or "").strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Сообщение не должно быть пустым")
    if RAG_MAX_USER_MESSAGE_CHARS > 0 and len(user_message) > RAG_MAX_USER_MESSAGE_CHARS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Сообщение слишком длинное ({len(user_message)} символов). "
                f"Максимум: {RAG_MAX_USER_MESSAGE_CHARS}."
            ),
        )

    tool_events: list[dict] = []

    def push_event(source: str, tool_name: str, status: str, message: str = "") -> None:
        tool_events.append(
            {
                "source": source,
                "tool_name": tool_name,
                "status": status,
                "message": message,
                "ts": int(time.time() * 1000),
            }
        )

    thread_id = _make_thread_id(username, conversation_id)
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RAG_MAX_AGENT_RECURSION_LIMIT,
    }
    async def event_generator():
        accumulated_parts: list[str] = []
        final_content = ""
        try:
            # monday tools run under the user's OWN monday token, so monday enforces their
            # permissions. Reads run freely; write tools are confirmation-gated (they only
            # mutate monday once the user confirms in chat — see monday_tools). Injected only on
            # this async path: MCP tools are async-only and the sync /chat path cannot execute
            # them. Empty when monday is off / the user has not connected.
            monday_tools = await aget_monday_tools_for_user(username)
            if monday_tools:
                # monday workflows are multi-step; give them a larger step budget.
                config["recursion_limit"] = RAG_MONDAY_AGENT_RECURSION_LIMIT
            # Long-term memory: sync tools + the injected memory block. Offloaded to a thread
            # so the sync DB IO doesn't stall the event loop. Empty/None when memory is off.
            user_id = get_user_id(username)
            memory_tools = (
                await asyncio.to_thread(get_memory_tools_for_user, user_id, thread_id)
                if user_id else []
            )
            memory_suffix = (
                await asyncio.to_thread(user_memory.build_memory_block, user_id)
                if user_id else None
            )
            # build_agent only assembles the graph here; offloaded so the sync
            # build + history prep never stall the event loop.
            runtime_agent = await asyncio.to_thread(
                build_agent,
                extra_tools=[*memory_tools, *monday_tools],
                system_prompt_suffix=compose_system_prompt_suffix(
                    memory_suffix, monday_system_prompt if monday_tools else None
                ),
            )
            try:
                if await asyncio.to_thread(_repair_conversation_history_for_provider, runtime_agent, config):
                    push_event("system", "history_repair", "success", "Conversation history was repaired.")
                if RAG_MAX_HISTORY_MESSAGES > 0:
                    try:
                        if await asyncio.to_thread(_compact_conversation_history, runtime_agent, config):
                            push_event("system", "history_guard", "success", "Conversation history was compacted.")
                    except Exception:
                        pass
            except Exception:
                logger.exception("history prep failed in chat_stream")

            async for ev in runtime_agent.astream_events(
                {"messages": [{"role": "user", "content": user_message}]},
                config=config,
                context=Context(user_id=username),
                version="v2",
            ):
                kind = ev.get("event", "")
                if kind == "on_chat_model_stream":
                    chunk = ev.get("data", {}).get("chunk")
                    content = getattr(chunk, "content", None)
                    text = ""
                    if isinstance(content, str):
                        text = content
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict):
                                if block.get("type") == "text":
                                    text += str(block.get("text") or "")
                            elif isinstance(block, str):
                                text += block
                    if text:
                        accumulated_parts.append(text)
                        yield (
                            "data: "
                            + json.dumps({"type": "delta", "text": text}, ensure_ascii=False)
                            + "\n\n"
                        )
                elif kind == "on_tool_start":
                    tool_name = str(ev.get("name") or "")
                    push_event("tool", tool_name, "start")
                    yield (
                        "data: "
                        + json.dumps({"type": "tool_start", "name": tool_name}, ensure_ascii=False)
                        + "\n\n"
                    )
                elif kind == "on_tool_end":
                    tool_name = str(ev.get("name") or "")
                    output = ev.get("data", {}).get("output")
                    output_str = ""
                    if output is not None:
                        try:
                            output_str = str(output)
                        except Exception:
                            output_str = ""
                    is_error = '"ok":false' in output_str.replace(" ", "") or "error" in output_str[:80].lower()
                    push_event("tool", tool_name, "error" if is_error else "success", output_str[:200])
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "type": "tool_end",
                                "name": tool_name,
                                "status": "error" if is_error else "success",
                            },
                            ensure_ascii=False,
                        )
                        + "\n\n"
                    )

            final_content = "".join(accumulated_parts).strip()
            if not final_content:
                try:
                    state = runtime_agent.get_state(config)
                    msgs = state.values.get("messages", []) if state else []
                    for m in reversed(msgs or []):
                        if getattr(m, "type", "") == "ai" or getattr(m, "role", "") == "assistant":
                            c = getattr(m, "content", None)
                            if isinstance(c, str) and c.strip():
                                final_content = c.strip()
                                break
                            if isinstance(c, list):
                                joined = "".join(
                                    b.get("text", "") if isinstance(b, dict) else str(b) for b in c
                                ).strip()
                                if joined:
                                    final_content = joined
                                    break
                except Exception:
                    pass

            try:
                _ensure_assistant_turn_persisted(runtime_agent, config, final_content or "")
            except Exception:
                logger.exception("Failed to persist assistant turn (stream)")

            sources = get_last_sources()
            log_append(username=username, question=user_message, answer=final_content, sources=sources)
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "done",
                        "response": final_content,
                        "sources": sources,
                        "tool_events": tool_events,
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            )
        except Exception as e:
            logger.exception("Chat stream failed")
            try:
                log_append(username=username, question=user_message, answer="", sources=[], error=str(e))
            except Exception:
                pass
            err_msg = str(e) or "Internal error"
            yield (
                "data: "
                + json.dumps({"type": "error", "message": err_msg}, ensure_ascii=False)
                + "\n\n"
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.delete("/chat/conversation")
def delete_chat_conversation(
    authorization: str | None = Header(default=None),
    conversation_id: str | None = Query(default=None),
):
    """Permanently delete one conversation history for current user."""
    username = _get_username(authorization)
    conv = (conversation_id or "").strip()
    if not conv:
        raise HTTPException(status_code=400, detail="conversation_id is required")
    thread_id = _make_thread_id(username, conv)
    try:
        delete_conversation_state(thread_id)
        delete_conversation_meta_for_user(username, conv)
        return {"ok": True, "conversation_id": conv}
    except Exception as e:
        logger.exception("Failed to delete conversation")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/admin/history/conversations")
def admin_delete_all_conversations(
    request: Request,
    authorization: str | None = Header(default=None),
    scan_checkpoints: int = Query(default=50000, ge=100, le=500000),
):
    """Delete all persisted chat conversation threads globally. Admin only."""
    admin_username = _require_admin(authorization)
    cp = getattr(get_base_agent(), "checkpointer", None)
    list_fn = getattr(cp, "list", None) if cp is not None else None
    if not callable(list_fn):
        return {
            "ok": False,
            "scanned_checkpoints": 0,
            "discovered_threads": 0,
            "deleted_threads": 0,
            "failed_threads": 0,
            "warning": "Checkpointer does not support thread listing in this runtime.",
        }

    discovered_thread_ids: list[str] = []
    seen: set[str] = set()
    scanned = 0
    for item in list_fn(None, limit=scan_checkpoints):
        scanned += 1
        conf = getattr(item, "config", None) or {}
        confg = conf.get("configurable", {}) if isinstance(conf, dict) else {}
        thread_id = str(confg.get("thread_id") or "").strip()
        if not thread_id or thread_id in seen:
            continue
        seen.add(thread_id)
        discovered_thread_ids.append(thread_id)

    deleted = 0
    failed = 0
    failed_ids: list[str] = []
    for thread_id in discovered_thread_ids:
        try:
            delete_conversation_state(thread_id)
            deleted += 1
        except Exception:
            failed += 1
            if len(failed_ids) < 20:
                failed_ids.append(thread_id)

    write_audit(
        "delete_all_conversations",
        admin_username,
        details={"discovered": len(discovered_thread_ids), "deleted": deleted, "failed": failed},
        ip_address=request.client.host if request.client else "",
    )
    return {
        "ok": failed == 0,
        "scanned_checkpoints": scanned,
        "discovered_threads": len(discovered_thread_ids),
        "deleted_threads": deleted,
        "failed_threads": failed,
        "failed_thread_ids_sample": failed_ids,
    }


def _safe_relative_path(path_str: str) -> Path | None:
    """Resolve path_str (relative to knowledge_base) and ensure it's under KNOWLEDGE_BASE_DIR. Return Path or None if invalid."""
    path_str = path_str.strip().replace("\\", "/").lstrip("/")
    if not path_str or ".." in path_str or path_str.startswith("/"):
        return None
    if not path_str.lower().endswith(".pdf"):
        return None
    target = (KNOWLEDGE_BASE_DIR / path_str).resolve()
    try:
        target.relative_to(KNOWLEDGE_BASE_DIR.resolve())
    except ValueError:
        return None
    return target


@app.get("/knowledge/files")
def knowledge_list(authorization: str | None = Header(default=None)):
    """List all PDF files in the RAG knowledge base. Requires admin."""
    _require_admin(authorization)
    return {"files": list_knowledge_files()}


@app.get("/knowledge/files/preview")
def knowledge_preview(
    path: str,
    authorization: str | None = Header(default=None),
):
    """Stream a PDF file for preview. Requires admin."""
    _require_admin(authorization)
    target = _safe_relative_path(path)
    if not target:
        raise HTTPException(status_code=400, detail="Недопустимый путь")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Файл не найден")
    return FileResponse(
        path=target,
        media_type="application/pdf",
        filename=target.name,
    )


def _rel_under_knowledge(target: Path) -> str:
    return str(target.relative_to(KNOWLEDGE_BASE_DIR)).replace("\\", "/")


@app.get("/knowledge/files/text")
def knowledge_pdf_text_get(
    path: str,
    authorization: str | None = Header(default=None),
):
    """Return RAG text for a PDF: sidecar override if present, else extracted PDF text."""
    _require_admin(authorization)
    target = _safe_relative_path(path)
    if not target:
        raise HTTPException(status_code=400, detail="Недопустимый путь")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Файл не найден")
    rel = _rel_under_knowledge(target)
    sidecar = rag_sidecar_path(target)
    if sidecar.is_file():
        raw = sidecar.read_text(encoding="utf-8", errors="replace")
        if raw.strip():
            return {"text": raw, "source": "override", "path": rel}
    try:
        extracted = extract_pdf_plain_text(target)
    except Exception as e:
        logger.exception("PDF text extraction failed")
        raise HTTPException(status_code=500, detail=str(e))
    return {"text": extracted, "source": "extracted", "path": rel}


class PdfTextPut(BaseModel):
    path: str = Field(..., min_length=1, max_length=1024)
    text: str = Field(default="", max_length=10_000_000)


@app.put("/knowledge/files/text")
def knowledge_pdf_text_put(
    body: PdfTextPut,
    authorization: str | None = Header(default=None),
):
    """
    Save RAG text override (sidecar .rag.txt) for a PDF without moving the file.
    Empty text removes the sidecar and reverts to PyPDF extraction.
    """
    _require_admin(authorization)
    target = _safe_relative_path(body.path)
    if not target:
        raise HTTPException(status_code=400, detail="Недопустимый путь")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Файл не найден")

    rel = _rel_under_knowledge(target)
    text_stripped = (body.text or "").strip()
    sc = rag_sidecar_path(target)

    if not text_stripped:
        if sc.is_file():
            try:
                sc.unlink()
            except OSError as e:
                raise HTTPException(status_code=500, detail=str(e))
        invalidate_vector_store()
        upsert_pdf_document(rel)
        return {"ok": True, "path": rel, "source": "extracted"}

    try:
        sc.parent.mkdir(parents=True, exist_ok=True)
        sc.write_text(text_stripped, encoding="utf-8")
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e))

    invalidate_vector_store()
    upsert_pdf_document(rel)
    return {"ok": True, "path": rel, "source": "override"}


@app.delete("/knowledge/files")
def knowledge_delete(
    path: str,
    authorization: str | None = Header(default=None),
):
    """Delete a PDF from the knowledge base. path = relative path (e.g. doc.pdf or folder/doc.pdf). Reindexes after. Requires admin."""
    _require_admin(authorization)
    target = _safe_relative_path(path)
    if not target:
        raise HTTPException(status_code=400, detail="Недопустимый путь к файлу")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Файл не найден")
    try:
        rel_path = str(target.relative_to(KNOWLEDGE_BASE_DIR)).replace("\\", "/")
        target.unlink()
        delete_pdf_metadata(rel_path)
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e))
    invalidate_vector_store()
    delete_pdf_document(rel_path)
    return {"ok": True, "files": list_knowledge_files()}


@app.patch("/knowledge/files/metadata")
def knowledge_pdf_metadata_update(
    body: PdfMetadataUpdate,
    authorization: str | None = Header(default=None),
):
    """Update only update_period_days for a PDF. Requires admin."""
    _require_admin(authorization)
    target = _safe_relative_path(body.path)
    if not target:
        raise HTTPException(status_code=400, detail="Недопустимый путь к файлу")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Файл не найден")
    rel_path = str(target.relative_to(KNOWLEDGE_BASE_DIR)).replace("\\", "/")
    meta = set_pdf_update_period(rel_path, update_period_days=body.update_period_days)
    expiry = compute_expiry(meta.get("last_updated_at") or "", meta.get("update_period_days"))
    return {
        "ok": True,
        "path": rel_path,
        "metadata": {
            "last_updated_at": meta.get("last_updated_at") or "",
            "update_period_days": meta.get("update_period_days"),
            "responsible": meta.get("responsible") or "",
            "expires_at": expiry.get("expires_at") or "",
            "expired": bool(expiry.get("expired")),
        },
    }


@app.post("/knowledge/upload")
def knowledge_upload(
    authorization: str | None = Header(default=None),
    file: UploadFile = File(...),
    update_period_days: int | None = Form(default=None),
):
    """Upload a PDF to the knowledge base. Reindexes after. Requires admin."""
    username = _require_admin(authorization)
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Нужен файл .pdf")
    safe_name = "".join(c for c in file.filename if c.isalnum() or c in "._- ").strip() or "document.pdf"
    if not safe_name.lower().endswith(".pdf"):
        safe_name += ".pdf"
    KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
    target = KNOWLEDGE_BASE_DIR / safe_name
    try:
        content = file.file.read()
        if len(content) > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Файл не более 50 МБ")
        target.write_bytes(content)
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e))
    record_pdf_upload(safe_name, responsible=username, update_period_days=update_period_days)
    invalidate_vector_store()
    upsert_pdf_document(safe_name)
    return {"ok": True, "name": safe_name, "files": list_knowledge_files()}


@app.post("/knowledge/reindex")
def knowledge_reindex(authorization: str | None = Header(default=None)):
    """Rebuild the RAG index from PDFs + knowledge items. Requires admin."""
    _require_admin(authorization)
    invalidate_vector_store()
    result = reconcile_all_documents()
    return {"ok": True, "files": list_knowledge_files(), "items": ki_list(), "reconcile": result}


# --- Knowledge items (text blocks) ---

class KnowledgeItemCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=256)
    content: str = Field(default="", max_length=500_000)
    # How often the responsible person should review/replace this item.
    update_period_days: int | None = Field(default=None, ge=1, le=3650)


class KnowledgeItemUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=256)
    content: str | None = Field(default=None, max_length=500_000)
    # If provided, updates only the expiry policy (does not change the responsible person).
    update_period_days: int | None = Field(default=None, ge=1, le=3650)


@app.get("/knowledge/items")
def knowledge_items_list(authorization: str | None = Header(default=None)):
    """List all text knowledge items. Requires admin."""
    _require_admin(authorization)
    return {"items": ki_list()}


@app.post("/knowledge/items")
def knowledge_item_create(body: KnowledgeItemCreate, authorization: str | None = Header(default=None)):
    """Create a text knowledge item. Reindexes after. Requires admin."""
    username = _require_admin(authorization)
    item = ki_add(
        body.name.strip(),
        body.content,
        update_period_days=body.update_period_days,
        responsible=username,
    )
    invalidate_vector_store()
    upsert_knowledge_item(str(item.get("id") or ""))
    return {"ok": True, "item": item}


@app.get("/knowledge/items/{item_id}")
def knowledge_item_get(item_id: str, authorization: str | None = Header(default=None)):
    """Get one knowledge item by id. Requires admin."""
    _require_admin(authorization)
    item = ki_get(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Элемент не найден")
    return item


@app.patch("/knowledge/items/{item_id}")
def knowledge_item_update(
    item_id: str,
    body: KnowledgeItemUpdate,
    authorization: str | None = Header(default=None),
):
    """Update a knowledge item. Reindexes after. Requires admin."""
    _require_admin(authorization)
    fields_set = getattr(body, "model_fields_set", set()) or set()
    touch_last_updated = ("name" in fields_set) or ("content" in fields_set)
    name = body.name if "name" in fields_set else None
    content = body.content if "content" in fields_set else None
    update_period_days = body.update_period_days if "update_period_days" in fields_set else KI_UNSET
    item = ki_update(
        item_id,
        name=name,
        content=content,
        update_period_days=update_period_days,
        touch_last_updated_at=touch_last_updated,
    )
    if not item:
        raise HTTPException(status_code=404, detail="Элемент не найден")
    invalidate_vector_store()
    upsert_knowledge_item(item_id)
    return {"ok": True, "item": item}


@app.delete("/knowledge/items/{item_id}")
def knowledge_item_delete(item_id: str, authorization: str | None = Header(default=None)):
    """Delete a knowledge item. Reindexes after. Requires admin."""
    _require_admin(authorization)
    if not ki_delete(item_id):
        raise HTTPException(status_code=404, detail="Элемент не найден")
    invalidate_vector_store()
    delete_knowledge_item_document(item_id)
    return {"ok": True, "items": ki_list()}


@app.get("/{route_path:path}", response_class=HTMLResponse)
def spa_fallback(route_path: str, accept: str | None = Header(default=None)):
    """
    Serve the React app shell for hard-refreshes on client-side routes.

    Keep asset/API misses as 404s so fetch failures do not silently receive HTML.
    """
    normalized = (route_path or "").strip("/")
    if "." in Path(normalized).name:
        raise HTTPException(status_code=404, detail="Not found")
    if accept and "text/html" not in accept.lower():
        raise HTTPException(status_code=404, detail="Not found")
    return _serve_frontend_or_legacy(STATIC_DIR / "index.html")


def run():
    """Run the API server (e.g. from CLI)."""
    import uvicorn
    uvicorn.run(
        "rag_agent.api:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
    )


if __name__ == "__main__":
    run()
