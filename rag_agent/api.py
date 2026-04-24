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
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
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
from rag_agent.auth import (
    change_password as auth_change_password,
    get_user_role,
    get_user_auth_flags,
    invalidate_token,
    is_password_change_required,
    login as auth_login,
    provision_user_with_temp_password,
    register as auth_register,
    resolve_token,
)
from rag_agent.config import (
    API_HOST,
    API_PORT,
    RAG_ENABLE_RATE_LIMIT_FALLBACK,
    RAG_FALLBACK_MODEL,
    RAG_HISTORY_KEEP_LAST_MESSAGES,
    RAG_HISTORY_SUMMARY_MAX_TOKEN_LIMIT,
    RAG_AGENT_DIR,
    RAG_MAX_HISTORY_MESSAGES,
    RAG_MAX_PASSWORD_LENGTH,
    RAG_MAX_USER_MESSAGE_CHARS,
    RAG_MIN_PASSWORD_LENGTH,
    RAG_USERNAME_MAX_LEN,
    require_runtime_keys,
)
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
    rename_pdf_metadata,
    set_pdf_update_period,
)
from rag_agent.rag_tool import get_last_sources, invalidate_vector_store, retrieval_debug
from rag_agent.chat_log import (
    append as log_append,
    list_entries as log_list_entries,
    count as log_count,
    update_review as log_update_review,
)

STATIC_DIR = RAG_AGENT_DIR / "static"
PROJECT_DIR = RAG_AGENT_DIR.parent
FRONTEND_DIR = RAG_AGENT_DIR / "frontend"
FRONTEND_DIST_DIR = FRONTEND_DIR / "dist"
FRONTEND_INDEX_PATH = FRONTEND_DIST_DIR / "index.html"

logging.basicConfig(level=logging.INFO)
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    require_runtime_keys()
    try:
        yield
    finally:
        close_checkpointer()


app = FastAPI(title="RAG Agent API", lifespan=lifespan)


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


@app.get("/health")
def health():
    """Production health check."""
    return {"status": "ok"}


@app.get("/admin", response_class=HTMLResponse)
def admin_index():
    """Serve admin route shell for the frontend app."""
    return _serve_frontend_or_legacy(STATIC_DIR / "admin.html")


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
    cp = getattr(agent, "checkpointer", None)
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
            state = agent.get_state(cfg)
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
        normalized.append(
            {
                "role": "assistant" if role in {"assistant", "ai"} else "user",
                "content": str(content_raw or "").strip(),
            }
        )
    return [m for m in normalized if m["content"]]


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

    try:
        if callable(get_state):
            state = get_state(config)
            values = getattr(state, "values", None) or {}
            messages = values.get("messages", []) or []
            if messages:
                last = messages[-1]
                if isinstance(last, dict):
                    last_role = str(last.get("role") or last.get("type") or "").lower()
                    last_content = str(last.get("content") or "")
                else:
                    last_role = str(getattr(last, "role", None) or getattr(last, "type", None) or "").lower()
                    last_content = str(getattr(last, "content", "") or "")
                if last_role in {"assistant", "ai"} and last_content.strip() == text:
                    return
        # Most common langgraph signature.
        try:
            update_state(config, {"messages": [{"role": "assistant", "content": text}]})
        except TypeError:
            # Compatibility fallback for other signatures.
            update_state({"messages": [{"role": "assistant", "content": text}]}, config=config)
    except Exception:
        # Best-effort only; never break chat response.
        return


@app.post("/auth/register", response_model=AuthResponse)
def register(body: RegisterRequest):
    """Create account; returns token and username. thread_id = username so history is per user."""
    ok, result = auth_register(body.username.strip(), body.password)
    if not ok:
        raise HTTPException(status_code=400, detail=result)
    username = resolve_token(result) or body.username.strip()
    flags = get_user_auth_flags(username)
    role = get_user_role(username)
    return AuthResponse(token=result, username=username, role=role, must_change_password=bool(flags.get("must_change_password")))


@app.post("/auth/login", response_model=AuthResponse)
def login(body: LoginRequest):
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


@app.post("/admin/users/provision")
def admin_user_provision(
    body: AdminProvisionUserRequest,
    authorization: str | None = Header(default=None),
):
    """
    Create employee account with a random temporary password.
    User must change password on first login.
    """
    admin_username = _require_admin(authorization)
    ok, result = provision_user_with_temp_password(
        created_by_username=admin_username,
        username=body.username.strip(),
        role=body.role.strip().lower(),
    )
    if not ok:
        raise HTTPException(status_code=400, detail=result)
    return {"ok": True, "user": result}


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
        if role == "assistant" and content.strip().startswith("{"):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "response_content" in parsed:
                    content = str(parsed["response_content"] or "")
            except (json.JSONDecodeError, TypeError):
                pass
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
        selected_model_name: str | None = None

        thread_id = _make_thread_id(username, conversation_id)
        config = {"configurable": {"thread_id": thread_id}}
        runtime_agent = build_agent(extra_tools=extra_tools, model_name=selected_model_name)
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
                fallback_agent = build_agent(extra_tools=extra_tools, model_name=RAG_FALLBACK_MODEL)
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
        upsert_pdf_document(_rel_under_knowledge(target))
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
    upsert_pdf_document(_rel_under_knowledge(final_pdf))
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
