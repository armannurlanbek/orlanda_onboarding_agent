"""
Central config from environment. Required in production.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths (project root = parent of rag_agent)
RAG_AGENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RAG_AGENT_DIR.parent
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / os.environ.get("KNOWLEDGE_BASE_DIR", "knowledge_base")

# Model
# Prefix with provider to switch safely, e.g.:
# - anthropic:claude-sonnet-4-6
# - openai:gpt-4o-mini
MODEL_NAME = os.environ.get("RAG_AGENT_MODEL", "anthropic:claude-sonnet-4-6").strip()
TEMPERATURE = float(os.environ.get("RAG_AGENT_TEMPERATURE", "0.5"))
MAX_TOKENS = int(os.environ.get("RAG_AGENT_MAX_TOKENS", "4096"))
# Timeout for OpenAI API (seconds). Increase if prompts are large or answers long.
TIMEOUT = int(os.environ.get("RAG_AGENT_TIMEOUT", "120"))

# RAG context budget controls to avoid oversized prompts/rate-limit spikes.
RAG_RETRIEVE_TOP_K = int(os.environ.get("RAG_RETRIEVE_TOP_K", "4"))
RAG_MAX_CHARS_PER_CHUNK = int(os.environ.get("RAG_MAX_CHARS_PER_CHUNK", "1200"))
RAG_MAX_TOTAL_CONTEXT_CHARS = int(os.environ.get("RAG_MAX_TOTAL_CONTEXT_CHARS", "6000"))
# Hard cap for persisted conversation messages in one thread. When exceeded,
# API resets that thread to avoid prompt/token explosion with strict TPM limits.
RAG_MAX_HISTORY_MESSAGES = int(os.environ.get("RAG_MAX_HISTORY_MESSAGES", "16"))

# Persistent checkpointer: set CHECKPOINT_DB to a file path (e.g. ./data/checkpoints.db) for production
CHECKPOINT_DB = os.environ.get("CHECKPOINT_DB", "").strip() or None

# API (when running as web service). PORT is set by Railway, Render, Fly.io, etc.
API_HOST = os.environ.get("RAG_AGENT_API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("PORT", os.environ.get("RAG_AGENT_API_PORT", "8000")))

# Auth: secret for password hashing and token signing (set in production)
SECRET_KEY = os.environ.get("RAG_AGENT_SECRET_KEY", "change-me-in-production")
USERS_FILE = RAG_AGENT_DIR / "data" / "users.json"
# Comma-separated usernames that have admin access (can open admin panel, manage docs, see logs)
ADMIN_USERNAMES = {u.strip().lower() for u in os.environ.get("RAG_AGENT_ADMIN_USERNAMES", "").split(",") if u.strip()}

# Monday OAuth + direct API integration
MONDAY_CLIENT_ID = os.environ.get("MONDAY_CLIENT_ID", "").strip()
MONDAY_CLIENT_SECRET = os.environ.get("MONDAY_CLIENT_SECRET", "").strip()
MONDAY_OAUTH_REDIRECT_URI = os.environ.get("MONDAY_OAUTH_REDIRECT_URI", "").strip()
MONDAY_OAUTH_SCOPES = os.environ.get("MONDAY_OAUTH_SCOPES", "me:read boards:read boards:write").strip()
MONDAY_OAUTH_AUTHORIZE_URL = os.environ.get("MONDAY_OAUTH_AUTHORIZE_URL", "https://auth.monday.com/oauth2/authorize").strip()
MONDAY_OAUTH_TOKEN_URL = os.environ.get("MONDAY_OAUTH_TOKEN_URL", "https://auth.monday.com/oauth2/token").strip()
MONDAY_CREDENTIALS_FILE = RAG_AGENT_DIR / "data" / "monday_credentials.json"
MONDAY_OAUTH_STATE_TTL_SECONDS = int(os.environ.get("MONDAY_OAUTH_STATE_TTL_SECONDS", "600"))
MONDAY_ENCRYPTION_KEY = os.environ.get("MONDAY_ENCRYPTION_KEY", "").strip()


def _provider_from_model(model_name: str) -> str:
    """Return provider prefix from model name (`provider:model`) or openai by default."""
    if ":" in model_name:
        return model_name.split(":", 1)[0].strip().lower()
    return "openai"


def require_runtime_keys() -> None:
    """Validate API keys needed by runtime: chat provider + OpenAI embeddings for RAG."""
    provider = _provider_from_model(MODEL_NAME)
    if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Set it in .env or the environment for production."
        )
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in .env or the environment for production."
        )
    # Retrieval currently uses OpenAIEmbeddings in rag_agent/indexing.py for query embeddings.
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is required for RAG embeddings and is not set."
        )
