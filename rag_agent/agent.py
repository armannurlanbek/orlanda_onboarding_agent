"""
LangChain agent instance with switchable chat model.
"""
import json
import logging
import yaml
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

from .config import (
    CHECKPOINT_BACKEND,
    CHECKPOINT_DB,
    CHECKPOINT_POSTGRES_URL,
    DATABASE_URL,
    MAX_TOKENS,
    MODEL_NAME,
    RAG_ENABLE_MONDAY_MCP,
    RAG_AGENT_DIR,
    TEMPERATURE,
    TIMEOUT,
)
from .monday_auth import get_monday_mcp_tools_for_user
from .rag_tool import retrieve_context

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_PATH = RAG_AGENT_DIR / "system_prompt.yaml"
_RUNTIME_SETTINGS_PATH = RAG_AGENT_DIR / "data" / "runtime_settings.json"


def _load_system_prompt() -> str:
    """Load system prompt from YAML file."""
    with open(_SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["system_prompt"].strip()


_checkpointer_cm = None
_checkpointer = None


def _postgres_checkpoint_dsn(raw_url: str) -> str:
    url = (raw_url or "").strip()
    if url.startswith("postgresql+psycopg://"):
        return "postgresql://" + url[len("postgresql+psycopg://") :]
    return url


def _get_checkpointer():
    """Create a valid checkpoint saver instance for configured backend."""
    global _checkpointer_cm, _checkpointer
    if _checkpointer is not None:
        return _checkpointer

    backend = (CHECKPOINT_BACKEND or "").strip().lower()
    if backend == "memory":
        _checkpointer = InMemorySaver()
        return _checkpointer

    if backend == "postgres":
        dsn = _postgres_checkpoint_dsn(CHECKPOINT_POSTGRES_URL or DATABASE_URL or "")
        if not dsn:
            raise RuntimeError(
                "CHECKPOINT_BACKEND=postgres requires CHECKPOINT_POSTGRES_URL or DATABASE_URL."
            )
        try:
            from langgraph.checkpoint.postgres import PostgresSaver  # type: ignore[reportMissingImports]
        except ImportError:
            raise ImportError(
                "CHECKPOINT_BACKEND=postgres requires langgraph-checkpoint-postgres. "
                "Run: pip install langgraph-checkpoint-postgres"
            ) from None
        _checkpointer_cm = PostgresSaver.from_conn_string(dsn)
        _checkpointer = _checkpointer_cm.__enter__()
        setup_fn = getattr(_checkpointer, "setup", None)
        if callable(setup_fn):
            setup_fn()
        return _checkpointer

    # sqlite backend
    if not CHECKPOINT_DB:
        _checkpointer = InMemorySaver()
        return _checkpointer
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        raise ImportError(
            "CHECKPOINT_BACKEND=sqlite requires langgraph-checkpoint-sqlite. "
            "Run: pip install langgraph-checkpoint-sqlite"
        ) from None
    _checkpointer_cm = SqliteSaver.from_conn_string(CHECKPOINT_DB)
    _checkpointer = _checkpointer_cm.__enter__()
    return _checkpointer


def close_checkpointer() -> None:
    """Close checkpointer context if we opened one."""
    global _checkpointer_cm, _checkpointer
    if _checkpointer_cm is not None:
        try:
            _checkpointer_cm.__exit__(None, None, None)
        except Exception:
            # Best-effort cleanup; shutdown should still succeed.
            pass
    _checkpointer_cm = None
    _checkpointer = None


def delete_conversation_state(thread_id: str) -> None:
    """Delete all persisted checkpoints for a thread id, if supported."""
    cp = _get_checkpointer()
    delete_fn = getattr(cp, "delete_thread", None)
    if callable(delete_fn):
        delete_fn(thread_id)


_active_model_name = MODEL_NAME


_SUPPORTED_CHAT_PROVIDERS = {
    "openai",
    "anthropic",
    "google_genai",
    "groq",
    "cohere",
    "mistralai",
    "ollama",
    "together",
    "fireworks",
}

_MODEL_ALIASES = {
    # Anthropic model aliases: normalize dot-version form to canonical id.
    "anthropic:claude-sonnet-4.6": "anthropic:claude-sonnet-4-6",
}


def _save_runtime_settings(model_name: str) -> None:
    """Persist active model so runtime switches survive process restarts."""
    try:
        _RUNTIME_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _RUNTIME_SETTINGS_PATH.write_text(
            json.dumps({"active_model": model_name}, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
    except Exception:
        # Persistence is best-effort: chat runtime should still work.
        return


def _load_runtime_model_name() -> str | None:
    """Load last persisted model from runtime settings file."""
    if not _RUNTIME_SETTINGS_PATH.is_file():
        return None
    try:
        raw = _RUNTIME_SETTINGS_PATH.read_text(encoding="utf-8")
        payload = json.loads(raw)
        model_name = str((payload or {}).get("active_model") or "").strip()
        return model_name or None
    except Exception:
        return None


def _normalize_model_name(model_name: str) -> str:
    raw = (model_name or "").strip()
    if not raw:
        raise ValueError("model_name must be a non-empty string")
    if ":" not in raw:
        raw = f"openai:{raw}"
    provider, _, model = raw.partition(":")
    provider = provider.strip().lower()
    model = model.strip()
    if not model:
        raise ValueError("Model must include provider and model id, e.g. 'openai:gpt-4o-mini'.")
    if provider not in _SUPPORTED_CHAT_PROVIDERS:
        raise ValueError(
            "Unsupported provider. Use one of: "
            + ", ".join(sorted(_SUPPORTED_CHAT_PROVIDERS))
            + "."
        )
    normalized = f"{provider}:{model}"
    return _MODEL_ALIASES.get(normalized, normalized)


def get_active_model_name() -> str:
    """Return the current chat model identifier."""
    loaded = _load_runtime_model_name()
    if loaded:
        try:
            return _normalize_model_name(loaded)
        except ValueError:
            pass
    return _normalize_model_name(_active_model_name)


def set_active_model(model_name: str) -> str:
    """
    Change the active chat model at runtime.
    Supports provider-prefixed model ids, e.g.:
    - openai:gpt-4o-mini
    - anthropic:claude-3-5-sonnet-latest
    - google_genai:gemini-1.5-pro
    For backward compatibility, plain model ids default to OpenAI.
    Returns the normalized model name that will be used.
    """
    global _active_model_name
    normalized = _normalize_model_name(model_name)
    _active_model_name = normalized
    _save_runtime_settings(normalized)
    return _active_model_name


def _build_chat_model(model_name: str | None = None):
    selected_model = _normalize_model_name(model_name or get_active_model_name())
    return init_chat_model(
        model=selected_model,
        temperature=TEMPERATURE,
        timeout=TIMEOUT,
        max_tokens=MAX_TOKENS,
    )


def _bootstrap_active_model() -> None:
    """Use persisted active model if available and valid."""
    global _active_model_name
    loaded = _load_runtime_model_name()
    if not loaded:
        return
    try:
        _active_model_name = _normalize_model_name(loaded)
    except ValueError:
        # Ignore invalid persisted value and keep env default.
        return

system_prompt = _load_system_prompt()
_bootstrap_active_model()


@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str


@dataclass
class ResponseFormat:
    """Response format for the agent."""
    response_content: str


def build_agent(
    extra_tools: list | None = None,
    model_name: str | None = None,
    use_response_format: bool = True,
    monday_username: str | None = None,
    include_monday_tools: bool = False,
    include_retrieve_context: bool = True,
):
    tools = [retrieve_context] if include_retrieve_context else []
    if include_monday_tools and RAG_ENABLE_MONDAY_MCP and monday_username:
        try:
            tools.extend(get_monday_mcp_tools_for_user(monday_username))
        except Exception:
            logger.exception("Failed to load per-user monday MCP tools")
    if extra_tools:
        tools.extend(extra_tools)
    kwargs = {
        "model": _build_chat_model(model_name=model_name),
        "tools": tools,
        "system_prompt": system_prompt,
        "checkpointer": _get_checkpointer(),
        "context_schema": Context,
    }
    if use_response_format:
        kwargs["response_format"] = ResponseFormat
    return create_agent(
        **kwargs,
    )


agent = build_agent()
