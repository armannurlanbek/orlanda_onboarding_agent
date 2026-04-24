"""
LangChain agent instance with switchable chat model.
"""
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
    RAG_AGENT_DIR,
    TEMPERATURE,
    TIMEOUT,
)
from .rag_tool import retrieve_context

_SYSTEM_PROMPT_PATH = RAG_AGENT_DIR / "system_prompt.yaml"


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


def _normalize_openai_model_name(model_name: str) -> str:
    raw = (model_name or "").strip()
    if not raw:
        raise ValueError("model_name must be a non-empty string")
    if ":" not in raw:
        raw = f"openai:{raw}"
    provider, _, model = raw.partition(":")
    if provider.strip().lower() != "openai" or not model.strip():
        raise ValueError(
            "Only OpenAI models are supported. Use value like 'openai:gpt-4o-mini' or 'gpt-4o-mini'."
        )
    return f"openai:{model.strip()}"


def get_active_model_name() -> str:
    """Return the current chat model identifier."""
    return _active_model_name


def set_active_model(model_name: str) -> str:
    """
    Change the active chat model at runtime.
    Supports OpenAI model ids, e.g. `openai:gpt-4o-mini` or `gpt-4o-mini`.
    Returns the normalized model name that will be used.
    """
    global _active_model_name
    normalized = _normalize_openai_model_name(model_name)
    _active_model_name = normalized
    return _active_model_name


def _build_chat_model(model_name: str | None = None):
    selected_model = _normalize_openai_model_name(model_name or _active_model_name)
    return init_chat_model(
        model=selected_model,
        temperature=TEMPERATURE,
        timeout=TIMEOUT,
        max_tokens=MAX_TOKENS,
    )

system_prompt = _load_system_prompt()


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
):
    tools = [retrieve_context]
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
