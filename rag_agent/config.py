"""
Central config from environment. Required in production.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Paths (project root = parent of rag_agent)
RAG_AGENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RAG_AGENT_DIR.parent
# Load `.env` from project root (same folder as `alembic.ini`) so DATABASE_URL works everywhere.
load_dotenv(PROJECT_ROOT / ".env")
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / os.environ.get("KNOWLEDGE_BASE_DIR", "knowledge_base")

# Model
# Prefix with provider to switch safely, e.g.:
# - openai:gpt-4o-mini
MODEL_NAME = os.environ.get("RAG_AGENT_MODEL", "openai:gpt-4o-mini").strip()
TEMPERATURE = float(os.environ.get("RAG_AGENT_TEMPERATURE", "0.5"))
MAX_TOKENS = int(os.environ.get("RAG_AGENT_MAX_TOKENS", "4096"))
# Timeout for OpenAI API (seconds). Increase if prompts are large or answers long.
TIMEOUT = int(os.environ.get("RAG_AGENT_TIMEOUT", "120"))

# RAG context budget controls to avoid oversized prompts/rate-limit spikes.
RAG_EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "text-embedding-3-small").strip()
RAG_VECTOR_DIM = int(os.environ.get("RAG_VECTOR_DIM", "1536"))
RAG_RETRIEVE_TOP_K = int(os.environ.get("RAG_RETRIEVE_TOP_K", "4"))
RAG_RETRIEVE_FETCH_K = int(os.environ.get("RAG_RETRIEVE_FETCH_K", "24"))
RAG_ENABLE_HYBRID_RETRIEVAL = os.environ.get("RAG_ENABLE_HYBRID_RETRIEVAL", "true").strip().lower() in {
    "1", "true", "yes", "on"
}
RAG_BM25_TOP_K = int(os.environ.get("RAG_BM25_TOP_K", "24"))
RAG_ENABLE_MMR = os.environ.get("RAG_ENABLE_MMR", "true").strip().lower() in {
    "1", "true", "yes", "on"
}
# 0.0 -> maximum diversity, 1.0 -> pure relevance.
RAG_MMR_LAMBDA = float(os.environ.get("RAG_MMR_LAMBDA", "0.35"))
RAG_RERANK_CANDIDATES_K = int(os.environ.get("RAG_RERANK_CANDIDATES_K", "18"))
RAG_RRF_K = int(os.environ.get("RAG_RRF_K", "60"))
RAG_ENABLE_CROSS_ENCODER_RERANK = os.environ.get("RAG_ENABLE_CROSS_ENCODER_RERANK", "false").strip().lower() in {
    "1", "true", "yes", "on"
}
RAG_CROSS_ENCODER_MODEL = os.environ.get(
    "RAG_CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
).strip()
RAG_MAX_CHARS_PER_CHUNK = int(os.environ.get("RAG_MAX_CHARS_PER_CHUNK", "1200"))
RAG_MAX_TOTAL_CONTEXT_CHARS = int(os.environ.get("RAG_MAX_TOTAL_CONTEXT_CHARS", "6000"))
RAG_NEIGHBOR_PAGE_WINDOW = int(os.environ.get("RAG_NEIGHBOR_PAGE_WINDOW", "1"))
RAG_NEIGHBOR_MAX_CHUNKS = int(os.environ.get("RAG_NEIGHBOR_MAX_CHUNKS", "4"))
RAG_QUERY_REWRITE_MAX = int(os.environ.get("RAG_QUERY_REWRITE_MAX", "3"))
RAG_RETRIEVAL_LOG_TOP = int(os.environ.get("RAG_RETRIEVAL_LOG_TOP", "12"))
# Hard cap for semantic conversation messages (user/assistant) in one thread.
# When exceeded, API compacts history (summary of old turns + keep latest turns).
RAG_MAX_HISTORY_MESSAGES = int(os.environ.get("RAG_MAX_HISTORY_MESSAGES", "16"))
# Keep this many latest semantic turns (user/assistant) after history compaction.
RAG_HISTORY_KEEP_LAST_MESSAGES = int(os.environ.get("RAG_HISTORY_KEEP_LAST_MESSAGES", "8"))
# Token budget used by the history summarization prompt for old turns.
RAG_HISTORY_SUMMARY_MAX_TOKEN_LIMIT = int(
    os.environ.get("RAG_HISTORY_SUMMARY_MAX_TOKEN_LIMIT", "1200")
)
# Hard cap for one inbound user message body (characters).
RAG_MAX_USER_MESSAGE_CHARS = int(os.environ.get("RAG_MAX_USER_MESSAGE_CHARS", "2500"))
# When provider returns 429, optionally retry once on a lighter fallback model.
RAG_ENABLE_RATE_LIMIT_FALLBACK = os.environ.get("RAG_ENABLE_RATE_LIMIT_FALLBACK", "true").strip().lower() in {
    "1", "true", "yes", "on"
}
RAG_FALLBACK_MODEL = os.environ.get("RAG_FALLBACK_MODEL", "openai:gpt-4o-mini").strip()

# Persistent checkpointer: set CHECKPOINT_DB to a file path (e.g. ./data/checkpoints.db) for production
CHECKPOINT_DB = os.environ.get("CHECKPOINT_DB", "").strip() or None
# Checkpoint backend: postgres | sqlite | memory.
CHECKPOINT_BACKEND = os.environ.get("CHECKPOINT_BACKEND", "postgres").strip().lower()
# Optional DSN override for postgres checkpoint backend; defaults to DATABASE_URL.
CHECKPOINT_POSTGRES_URL = os.environ.get("CHECKPOINT_POSTGRES_URL", "").strip() or None

# API (when running as web service). PORT is set by Railway, Render, Fly.io, etc.
API_HOST = os.environ.get("RAG_AGENT_API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("PORT", os.environ.get("RAG_AGENT_API_PORT", "8000")))

# PostgreSQL (SQLAlchemy + Alembic). Example:
# postgresql+psycopg://user:password@localhost:5432/rag_agent
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip() or None

# Auth: secret for password hashing and token signing (set in production)
SECRET_KEY = os.environ.get("RAG_AGENT_SECRET_KEY", "change-me-in-production")
# New registrations only: minimum length and composition (login still allows old passwords).
RAG_MIN_PASSWORD_LENGTH = max(8, int(os.environ.get("RAG_MIN_PASSWORD_LENGTH", "12")))
RAG_MAX_PASSWORD_LENGTH = min(256, int(os.environ.get("RAG_MAX_PASSWORD_LENGTH", "128")))
# Bearer sessions: validity when using PostgreSQL-backed auth_sessions (and in-memory TTL without DB).
RAG_SESSION_EXPIRY_DAYS = max(1, min(365, int(os.environ.get("RAG_SESSION_EXPIRY_DAYS", "7"))))
# Legacy import only (`python -m rag_agent.import_json_users`). Not used by runtime auth.
USERS_FILE = RAG_AGENT_DIR / "data" / "users.json"
# Comma-separated usernames that have admin access (can open admin panel, manage docs, see logs)
ADMIN_USERNAMES = {u.strip().lower() for u in os.environ.get("RAG_AGENT_ADMIN_USERNAMES", "").split(",") if u.strip()}
# Allowed non-email logins: only these short names (no "@"). Others must use this email domain.
RAG_ALLOWED_EMAIL_DOMAIN = os.environ.get("RAG_ALLOWED_EMAIL_DOMAIN", "orlanda.info").strip().lower()
# Max stored username length (emails need more than 64 characters).
RAG_USERNAME_MAX_LEN = min(255, max(64, int(os.environ.get("RAG_USERNAME_MAX_LEN", "255"))))

def _provider_from_model(model_name: str) -> str:
    """Return provider prefix from model name (`provider:model`) or openai by default."""
    if ":" in model_name:
        return model_name.split(":", 1)[0].strip().lower()
    return "openai"


def require_runtime_keys() -> None:
    """Validate API keys needed by runtime (OpenAI-only chat + embeddings for RAG)."""
    provider = _provider_from_model(MODEL_NAME)
    if provider != "openai":
        raise RuntimeError(
            "Only OpenAI models are supported in this deployment. "
            "Set RAG_AGENT_MODEL to an openai:* model (recommended: openai:gpt-4o-mini)."
        )
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Set it in .env or the environment for production.")
    # Retrieval currently uses OpenAIEmbeddings in rag_agent/indexing.py for query embeddings.
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is required for RAG embeddings and is not set."
        )
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is required. Authentication uses PostgreSQL only (users + auth_sessions); "
            "users.json is no longer used. Set DATABASE_URL in .env or the environment."
        )
