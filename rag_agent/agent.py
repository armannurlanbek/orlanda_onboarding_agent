"""
LangChain agent instance with switchable chat model.
"""
import yaml
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

from .config import CHECKPOINT_DB, MAX_TOKENS, MODEL_NAME, RAG_AGENT_DIR, TEMPERATURE, TIMEOUT
from .rag_tool import retrieve_context

_SYSTEM_PROMPT_PATH = RAG_AGENT_DIR / "system_prompt.yaml"


def _load_system_prompt() -> str:
    """Load system prompt from YAML file."""
    with open(_SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["system_prompt"].strip()


_sqlite_checkpointer_cm = None
_checkpointer = None


def _get_checkpointer():
    """Create a valid checkpoint saver instance.

    Note: in this langgraph-checkpoint-sqlite version, `SqliteSaver.from_conn_string`
    returns a context manager, so we must enter it and pass the yielded saver.
    """
    global _sqlite_checkpointer_cm, _checkpointer
    if _checkpointer is not None:
        return _checkpointer

    # Development / fallback
    if not CHECKPOINT_DB:
        _checkpointer = InMemorySaver()
        return _checkpointer

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        raise ImportError(
            "CHECKPOINT_DB is set but langgraph-checkpoint-sqlite is not installed. "
            "Run: pip install langgraph-checkpoint-sqlite"
        ) from None

    _sqlite_checkpointer_cm = SqliteSaver.from_conn_string(CHECKPOINT_DB)
    _checkpointer = _sqlite_checkpointer_cm.__enter__()
    return _checkpointer


def close_checkpointer() -> None:
    """Close sqlite checkpointer context if we opened one."""
    global _sqlite_checkpointer_cm, _checkpointer
    if _sqlite_checkpointer_cm is not None:
        try:
            _sqlite_checkpointer_cm.__exit__(None, None, None)
        except Exception:
            # Best-effort cleanup; shutdown should still succeed.
            pass
    _sqlite_checkpointer_cm = None
    _checkpointer = None


def delete_conversation_state(thread_id: str) -> None:
    """Delete all persisted checkpoints for a thread id, if supported."""
    cp = _get_checkpointer()
    delete_fn = getattr(cp, "delete_thread", None)
    if callable(delete_fn):
        delete_fn(thread_id)


_active_model_name = MODEL_NAME


def get_active_model_name() -> str:
    """Return the current chat model identifier."""
    return _active_model_name


def set_active_model(model_name: str) -> str:
    """
    Change the active chat model at runtime.
    Supports provider-prefixed model ids, e.g. `anthropic:claude-sonnet-4-6`.
    Returns the normalized model name that will be used.
    """
    global _active_model_name
    normalized = (model_name or "").strip()
    if not normalized:
        raise ValueError("model_name must be a non-empty string")
    _active_model_name = normalized
    return _active_model_name


def _build_chat_model(model_name: str | None = None):
    selected_model = (model_name or _active_model_name).strip()
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


def build_agent(extra_tools: list | None = None):
    tools = [retrieve_context]
    if extra_tools:
        tools.extend(extra_tools)
    return create_agent(
        model=_build_chat_model(),
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=_get_checkpointer(),
        context_schema=Context,
        response_format=ResponseFormat,
    )


agent = build_agent()
