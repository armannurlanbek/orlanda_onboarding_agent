"""
REST API for the RAG agent. Serves a local chat UI with login; /chat uses your account as thread_id so history is restored.
"""
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form, Query
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field

from rag_agent.agent import (
    Context,
    agent,
    build_agent,
    close_checkpointer,
    delete_conversation_state,
    get_active_model_name,
    set_active_model,
)
from rag_agent.auth import login as auth_login, register as auth_register, resolve_token, get_user_role
from rag_agent.config import (
    API_HOST,
    API_PORT,
    RAG_AGENT_DIR,
    RAG_MAX_HISTORY_MESSAGES,
    require_runtime_keys,
)
from rag_agent.monday_oauth import build_authorize_url, exchange_code_for_token, monday_oauth_enabled
from rag_agent.monday_store import (
    consume_oauth_state,
    create_oauth_state,
    delete_user_credentials as monday_delete_user_credentials,
    get_user_credentials as monday_get_user_credentials,
    save_user_credentials as monday_save_user_credentials,
)
from rag_agent.monday_tools import build_monday_tools
from rag_agent.indexing import (
    KNOWLEDGE_BASE_DIR,
    build_index,
    clear_index,
    extract_pdf_plain_text,
    list_knowledge_files,
    load_all_documents,
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
    rename_pdf_metadata,
    set_pdf_update_period,
)
from rag_agent.rag_tool import get_last_sources, invalidate_vector_store
from rag_agent.chat_log import (
    append as log_append,
    list_entries as log_list_entries,
    count as log_count,
    update_review as log_update_review,
)

STATIC_DIR = RAG_AGENT_DIR / "static"
PROJECT_DIR = RAG_AGENT_DIR.parent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _is_rate_limit_error(err: Exception) -> bool:
    txt = str(err).lower()
    return "rate_limit" in txt or "rate limit" in txt or "error code: 429" in txt


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    # When false (default), chat uses only RAG tools even if Monday is connected.
    use_monday: bool = False


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    response: str
    sources: list[dict] = Field(description="List of {file, page} used for the answer; empty if RAG was not used.")
    tool_events: list[dict] = Field(default_factory=list, description="Operational tool activity events shown in UI.")


class AuthResponse(BaseModel):
    token: str
    username: str


class MondayConnectResponse(BaseModel):
    authorize_url: str


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    require_runtime_keys()
    try:
        yield
    finally:
        close_checkpointer()


app = FastAPI(title="RAG Agent API", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the local chat UI."""
    path = STATIC_DIR / "index.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="index.html not found")
    return path.read_text(encoding="utf-8")


@app.get("/health")
def health():
    """Production health check."""
    return {"status": "ok"}


@app.get("/admin", response_class=HTMLResponse)
def admin_index():
    """Serve the admin panel UI. Log data is protected by auth on /admin/logs."""
    path = STATIC_DIR / "admin.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="admin.html not found")
    return path.read_text(encoding="utf-8")


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


@app.get("/admin/logs")
def admin_logs(
    authorization: str | None = Header(default=None),
    limit: int = 100,
    offset: int = 0,
):
    """Return chat log entries for the admin panel. Requires admin."""
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


@app.patch("/admin/logs/{entry_id}/review")
def admin_log_review_update(
    entry_id: str,
    body: AdminLogReviewUpdate,
    authorization: str | None = Header(default=None),
):
    """Update score/correct_answer for one log entry. Requires admin."""
    _require_admin(authorization)
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
    return {"ok": True, "entry": updated}


@app.get("/admin/model")
def admin_model_get(authorization: str | None = Header(default=None)):
    """Return active chat model. Requires admin."""
    _require_admin(authorization)
    return {"model": get_active_model_name()}


@app.put("/admin/model")
def admin_model_put(
    body: AdminModelUpdate,
    authorization: str | None = Header(default=None),
):
    """Set active chat model at runtime. Requires admin."""
    _require_admin(authorization)
    try:
        model = set_active_model(body.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "model": model}


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


def _get_username(authorization: str | None = Header(default=None)) -> str:
    """Require Bearer token and return username; 401 if invalid."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Требуется вход в аккаунт")
    token = authorization[7:].strip()
    username = resolve_token(token)
    if not username:
        raise HTTPException(status_code=401, detail="Сессия истекла, войдите снова")
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


@app.post("/auth/register", response_model=AuthResponse)
def register(body: LoginRequest):
    """Create account; returns token and username. thread_id = username so history is per user."""
    ok, result = auth_register(body.username.strip(), body.password)
    if not ok:
        raise HTTPException(status_code=400, detail=result)
    return AuthResponse(token=result, username=body.username.strip())


@app.post("/auth/login", response_model=AuthResponse)
def login(body: LoginRequest):
    """Log in; returns token and username."""
    ok, result = auth_login(body.username.strip(), body.password)
    if not ok:
        raise HTTPException(status_code=401, detail=result)
    return AuthResponse(token=result, username=body.username.strip())


@app.get("/auth/me")
def me(authorization: str | None = Header(default=None)):
    """Return current user and role if token valid."""
    username = _get_username(authorization)
    return {"username": username, "role": get_user_role(username)}


@app.get("/integrations/monday/status")
def monday_status(authorization: str | None = Header(default=None)):
    """Return Monday connection status for current user."""
    username = _get_username(authorization)
    creds = monday_get_user_credentials(username)
    connected = bool(creds and creds.get("access_token"))
    return {"connected": connected, "account": creds.get("account") if connected else None}


@app.post("/integrations/monday/connect", response_model=MondayConnectResponse)
def monday_connect(authorization: str | None = Header(default=None)):
    """Start Monday OAuth flow and return authorize URL."""
    username = _get_username(authorization)
    if not monday_oauth_enabled():
        raise HTTPException(status_code=400, detail="Monday integration is not configured.")
    state = create_oauth_state(username)
    return MondayConnectResponse(authorize_url=build_authorize_url(state))


@app.get("/integrations/monday/callback", response_class=HTMLResponse)
def monday_callback(code: str | None = None, state: str | None = None, error: str | None = None):
    """Handle Monday OAuth callback."""
    if error:
        return HTMLResponse(
            content=f"<html><body><script>window.location.href='/?monday_error={error}';</script></body></html>"
        )
    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing code/state.")
    username = consume_oauth_state(state)
    if not username:
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state.")
    token_payload = exchange_code_for_token(code)
    monday_save_user_credentials(
        username=username,
        payload={
            "access_token": token_payload.get("access_token"),
            "refresh_token": token_payload.get("refresh_token"),
            "expires_in": token_payload.get("expires_in"),
            "scope": token_payload.get("scope"),
            "token_type": token_payload.get("token_type"),
            "connected_at": time.time(),
            "account": token_payload.get("account"),
        },
    )
    logger.info("Monday connected for user=%s", username)
    return HTMLResponse(
        content="<html><body><script>window.location.href='/?monday_connected=1';</script></body></html>"
    )


@app.post("/integrations/monday/disconnect")
def monday_disconnect(authorization: str | None = Header(default=None)):
    """Disconnect Monday for current user."""
    username = _get_username(authorization)
    monday_delete_user_credentials(username)
    logger.info("Monday disconnected for user=%s", username)
    return {"ok": True}


def _messages_to_history(messages) -> list[dict]:
    """Convert agent state messages to [{role, content}, ...] for frontend (user/assistant only)."""
    out = []
    for m in messages or []:
        if isinstance(m, dict):
            if m.get("type") in ("tool", "system") or m.get("role") in ("tool", "system"):
                continue
            role = m.get("role") or m.get("type", "user")
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
            role = "assistant" if role in ("ai", "assistant") else "user"
        if isinstance(content, list):
            content = " ".join(
                (c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
            )
        content = str(content)
        # Agent may store assistant reply as JSON with response_content; extract plain text for history
        if role == "assistant" and content.strip().startswith("{"):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "response_content" in parsed:
                    content = str(parsed["response_content"] or "")
            except (json.JSONDecodeError, TypeError):
                pass
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
        get_state = getattr(agent, "get_state", None)
        if not get_state:
            return {"messages": []}
        state = get_state(config)
        values = getattr(state, "values", None) or {}
        messages = values.get("messages", [])
        return {"messages": _messages_to_history(messages)}
    except Exception as e:
        logger.exception("Failed to get chat history")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
def chat(
    body: ChatRequest,
    authorization: str | None = Header(default=None),
    conversation_id: str | None = Query(default=None),
):
    """
    Send a message; agent reply + sources. Requires login.
    thread_id = your username, so when you come back you get your conversation history (if CHECKPOINT_DB is set).
    Set body.use_monday=true to attach Monday tools when the user has connected Monday; default is RAG-only.
    """
    username = _get_username(authorization)
    try:
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

        extra_tools = []
        creds = monday_get_user_credentials(username)
        access_token = str((creds or {}).get("access_token") or "")
        if body.use_monday and access_token:
            on_tool_event(
                {
                    "source": "monday",
                    "tool_name": "integration",
                    "status": "start",
                    "message": "Monday integration connected",
                    "ts": int(time.time() * 1000),
                }
            )
            extra_tools = build_monday_tools(access_token=access_token, on_event=on_tool_event)
            if extra_tools:
                on_tool_event(
                    {
                        "source": "monday",
                        "tool_name": "integration",
                        "status": "success",
                        "message": f"Loaded {len(extra_tools)} Monday tools",
                        "ts": int(time.time() * 1000),
                    }
                )
            else:
                on_tool_event(
                    {
                        "source": "monday",
                        "tool_name": "integration",
                        "status": "error",
                        "message": "Monday tools unavailable",
                        "ts": int(time.time() * 1000),
                    }
                )

        thread_id = _make_thread_id(username, conversation_id)
        config = {"configurable": {"thread_id": thread_id}}
        runtime_agent = build_agent(extra_tools=extra_tools)
        # Prevent unlimited growth of persisted thread context, which can trigger
        # strict provider TPM limits (especially on Anthropic plans).
        if RAG_MAX_HISTORY_MESSAGES > 0:
            try:
                get_state = getattr(runtime_agent, "get_state", None)
                if callable(get_state):
                    state = get_state(config)
                    values = getattr(state, "values", None) or {}
                    history_messages = values.get("messages", []) or []
                    if len(history_messages) > RAG_MAX_HISTORY_MESSAGES:
                        delete_conversation_state(thread_id)
                        on_tool_event(
                            {
                                "source": "system",
                                "tool_name": "history_guard",
                                "status": "success",
                                "message": (
                                    "Conversation history was reset to avoid "
                                    "token limit overflow."
                                ),
                                "ts": int(time.time() * 1000),
                            }
                        )
            except Exception:
                # Best-effort only; chat should continue even if introspection fails.
                pass
        response = runtime_agent.invoke(
            {"messages": [{"role": "user", "content": body.message}]},
            config=config,
            context=Context(user_id=username),
        )
        content = response["structured_response"].response_content
        sources = get_last_sources()
        log_append(username=username, question=body.message, answer=content, sources=sources)
        return ChatResponse(response=content, sources=sources, tool_events=tool_events)
    except Exception as e:
        logger.exception("Chat request failed")
        log_append(
            username=username,
            question=body.message,
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
        raise HTTPException(status_code=500, detail=str(e))


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
        return {"ok": True, "conversation_id": conv}
    except Exception as e:
        logger.exception("Failed to delete conversation")
        raise HTTPException(status_code=500, detail=str(e))


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
    Save RAG text override (sidecar .rag.txt). First non-empty save renames *.pdf to *_changed.pdf
    when the stem does not already end with _changed. Empty text removes the sidecar and reindexes from PDF.
    """
    _require_admin(authorization)
    target = _safe_relative_path(body.path)
    if not target:
        raise HTTPException(status_code=400, detail="Недопустимый путь")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Файл не найден")

    text_stripped = (body.text or "").strip()

    if not text_stripped:
        sc = rag_sidecar_path(target)
        if sc.is_file():
            try:
                sc.unlink()
            except OSError as e:
                raise HTTPException(status_code=500, detail=str(e))
        invalidate_vector_store()
        if not load_all_documents():
            clear_index()
        else:
            build_index()
        return {"ok": True, "path": _rel_under_knowledge(target), "source": "extracted"}

    final_pdf = target
    stem = target.stem
    if not stem.endswith("_changed"):
        new_pdf = target.parent / f"{stem}_changed.pdf"
        if new_pdf.exists():
            raise HTTPException(
                status_code=400,
                detail="Файл с таким именем уже существует; удалите или переименуйте вручную.",
            )
        rel_old = _rel_under_knowledge(target)
        sc_old = rag_sidecar_path(target)
        try:
            target.rename(new_pdf)
        except OSError as e:
            raise HTTPException(status_code=500, detail=str(e))
        if sc_old.is_file():
            sc_new = rag_sidecar_path(new_pdf)
            try:
                sc_old.rename(sc_new)
            except OSError as e:
                raise HTTPException(status_code=500, detail=str(e))
        rename_pdf_metadata(rel_old, _rel_under_knowledge(new_pdf))
        final_pdf = new_pdf

    sc_final = rag_sidecar_path(final_pdf)
    try:
        sc_final.parent.mkdir(parents=True, exist_ok=True)
        sc_final.write_text(text_stripped, encoding="utf-8")
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e))

    invalidate_vector_store()
    build_index()
    return {"ok": True, "path": _rel_under_knowledge(final_pdf), "source": "override"}


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
    if not load_all_documents():
        clear_index()
    else:
        build_index()
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
    build_index()
    return {"ok": True, "name": safe_name, "files": list_knowledge_files()}


@app.post("/knowledge/reindex")
def knowledge_reindex(authorization: str | None = Header(default=None)):
    """Rebuild the RAG index from PDFs + knowledge items. Requires admin."""
    _require_admin(authorization)
    invalidate_vector_store()
    if not load_all_documents():
        clear_index()
        return {"ok": True, "message": "Нет документов", "files": list_knowledge_files(), "items": ki_list()}
    build_index()
    return {"ok": True, "files": list_knowledge_files(), "items": ki_list()}


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
    build_index()
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
    build_index()
    return {"ok": True, "item": item}


@app.delete("/knowledge/items/{item_id}")
def knowledge_item_delete(item_id: str, authorization: str | None = Header(default=None)):
    """Delete a knowledge item. Reindexes after. Requires admin."""
    _require_admin(authorization)
    if not ki_delete(item_id):
        raise HTTPException(status_code=404, detail="Элемент не найден")
    invalidate_vector_store()
    if not load_all_documents():
        clear_index()
    else:
        build_index()
    return {"ok": True, "items": ki_list()}


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
