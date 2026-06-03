"""
Central config from environment. Required in production.
"""
import ipaddress
import logging
import os
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

# Paths (project root = parent of rag_agent)
RAG_AGENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RAG_AGENT_DIR.parent
# Load `.env` from project root (same folder as `alembic.ini`) so DATABASE_URL works everywhere.
load_dotenv(PROJECT_ROOT / ".env")
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

# Optional Monday MCP integration.
RAG_ENABLE_MONDAY_MCP = os.environ.get("RAG_ENABLE_MONDAY_MCP", "false").strip().lower() in {
    "1", "true", "yes", "on"
}
RAG_MONDAY_MCP_URL = os.environ.get("RAG_MONDAY_MCP_URL", "https://mcp.monday.com/mcp").strip()
RAG_MONDAY_MCP_TRANSPORT = os.environ.get("RAG_MONDAY_MCP_TRANSPORT", "streamable_http").strip()
RAG_MONDAY_MCP_TIMEOUT_SECONDS = int(os.environ.get("RAG_MONDAY_MCP_TIMEOUT_SECONDS", "25"))
RAG_MONDAY_MCP_TOOL_ALLOWLIST = {
    name.strip() for name in os.environ.get("RAG_MONDAY_MCP_TOOL_ALLOWLIST", "").split(",") if name.strip()
}
# Max Monday tools bound per request. Big enough to keep the full essential
# read/discovery set (get_board_info, search, list_workspaces, all_monday_api,
# get_column_type_info, ...) PLUS the core write tools. Anthropic/OpenAI both
# handle dozens of tools; the old default (18) combined with a write-biased
# ranking silently evicted the discovery tools the agent needs to find anything.
RAG_MONDAY_MCP_MAX_TOOLS = max(
    1,
    min(128, int(os.environ.get("RAG_MONDAY_MCP_MAX_TOOLS", "25"))),
)
# Optional-parameter budget across selected monday tools. OFF by default (0).
# The old default (22) was catastrophic: a single heavy tool
# (get_board_items_page ~15 optional params) exhausted the budget and the greedy
# walk DROPPED every tool after it, leaving the agent with ~6 mostly-write tools
# and no way to search boards or read column ids. There is no real provider limit
# this low. Set >0 only if a specific provider actually rejects the schema set.
RAG_MONDAY_MCP_MAX_OPTIONAL_PARAMS = max(
    0,
    min(1000, int(os.environ.get("RAG_MONDAY_MCP_MAX_OPTIONAL_PARAMS", "0"))),
)
RAG_MONDAY_MCP_TOOLS_CACHE_TTL_SECONDS = max(
    30,
    int(os.environ.get("RAG_MONDAY_MCP_TOOLS_CACHE_TTL_SECONDS", "600")),
)
# Bounded retry for transient Monday MCP failures (connection reset / 5xx /
# session-termination 500) on both tool loading and tool invocation. Keep small;
# the goal is to ride out blips, not to hammer a degraded upstream.
RAG_MONDAY_MCP_MAX_RETRIES = max(
    0,
    min(5, int(os.environ.get("RAG_MONDAY_MCP_MAX_RETRIES", "2"))),
)
# Base backoff (seconds) between retries; grows linearly with the attempt number.
RAG_MONDAY_MCP_RETRY_BACKOFF_SECONDS = max(
    0.0,
    min(5.0, float(os.environ.get("RAG_MONDAY_MCP_RETRY_BACKOFF_SECONDS", "0.5"))),
)
RAG_MONDAY_MCP_SUPPRESS_TERMINATION_500_WARNINGS = os.environ.get(
    "RAG_MONDAY_MCP_SUPPRESS_TERMINATION_500_WARNINGS",
    "true",
).strip().lower() in {"1", "true", "yes", "on"}
RAG_MAX_AGENT_RECURSION_LIMIT = max(
    6,
    int(os.environ.get("RAG_MAX_AGENT_RECURSION_LIMIT", "12")),
)
RAG_MONDAY_MCP_OAUTH_ENABLED = os.environ.get("RAG_MONDAY_MCP_OAUTH_ENABLED", "true").strip().lower() in {
    "1", "true", "yes", "on"
}
# When false (default): a CONNECTED monday user always has monday tools bound,
# regardless of keyword intent — so phrasing never hides the tools. Set to true
# as an opt-in to gate tools behind the keyword intent classifier instead.
RAG_MONDAY_MCP_USE_FOR_INTENT_ONLY = os.environ.get("RAG_MONDAY_MCP_USE_FOR_INTENT_ONLY", "false").strip().lower() in {
    "1", "true", "yes", "on"
}
# monday OAuth settings (hosted MCP user auth).
RAG_MONDAY_OAUTH_CLIENT_ID = os.environ.get("RAG_MONDAY_OAUTH_CLIENT_ID", "").strip()
RAG_MONDAY_OAUTH_CLIENT_SECRET = os.environ.get("RAG_MONDAY_OAUTH_CLIENT_SECRET", "").strip()
RAG_MONDAY_OAUTH_REDIRECT_URI = os.environ.get("RAG_MONDAY_OAUTH_REDIRECT_URI", "").strip()
RAG_MONDAY_OAUTH_AUTHORIZE_URL = os.environ.get(
    "RAG_MONDAY_OAUTH_AUTHORIZE_URL",
    "https://auth.monday.com/oauth2/authorize",
).strip()
RAG_MONDAY_OAUTH_TOKEN_URL = os.environ.get(
    "RAG_MONDAY_OAUTH_TOKEN_URL",
    "https://auth.monday.com/oauth2/token",
).strip()
RAG_MONDAY_OAUTH_SCOPES = os.environ.get("RAG_MONDAY_OAUTH_SCOPES", "").strip()
RAG_MONDAY_OAUTH_STATE_TTL_SECONDS = max(
    60,
    int(os.environ.get("RAG_MONDAY_OAUTH_STATE_TTL_SECONDS", "600")),
)
MONDAY_ENCRYPTION_KEY = os.environ.get("MONDAY_ENCRYPTION_KEY", "").strip()

# Persistent checkpointer: set CHECKPOINT_DB to a file path (e.g. ./data/checkpoints.db) for production
CHECKPOINT_DB = os.environ.get("CHECKPOINT_DB", "").strip() or None
# Checkpoint backend: postgres | sqlite | memory.
CHECKPOINT_BACKEND = os.environ.get("CHECKPOINT_BACKEND", "postgres").strip().lower()
# Optional DSN override for postgres checkpoint backend; defaults to DATABASE_URL.
CHECKPOINT_POSTGRES_URL = os.environ.get("CHECKPOINT_POSTGRES_URL", "").strip() or None

# API (when running as web service). PORT is set by Railway, Render, Fly.io, etc.
API_HOST = os.environ.get("RAG_AGENT_API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("PORT", os.environ.get("RAG_AGENT_API_PORT", "8000")))
# Optional absolute frontend URL used for OAuth callback redirects in split dev/prod setups.
# In production this MUST be your public https origin (e.g. https://platform.n8norlanda.com)
# or left EMPTY. Leaving it empty makes the post-OAuth 302 a same-origin relative path
# ("/chat?..."), which is correct when the API and frontend share the public host.
# NEVER set this to a private/LAN address (192.168.x.x / 10.x / 172.16-31.x) — the
# user's browser cannot reach it and OAuth will appear to "not log in".
RAG_FRONTEND_BASE_URL = os.environ.get("RAG_FRONTEND_BASE_URL", "").strip().rstrip("/")

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

# CORS: comma-separated list of allowed origins for browser requests.
RAG_CORS_ALLOWED_ORIGINS: list[str] = [
    o.strip()
    for o in os.environ.get(
        "RAG_CORS_ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://localhost:8000",
    ).split(",")
    if o.strip()
]

# Rate limiting (slowapi format: "N/period" where period = second|minute|hour|day).
RAG_RATE_LIMIT_LOGIN = os.environ.get("RAG_RATE_LIMIT_LOGIN", "10/minute").strip()
RAG_RATE_LIMIT_REGISTER = os.environ.get("RAG_RATE_LIMIT_REGISTER", "5/minute").strip()
RAG_RATE_LIMIT_CHAT = os.environ.get("RAG_RATE_LIMIT_CHAT", "60/minute").strip()

# Logging format: "json" (default, structured) or "text" (human-readable dev output).
RAG_LOG_FORMAT = os.environ.get("RAG_LOG_FORMAT", "json").strip().lower()


def _provider_from_model(model_name: str) -> str:
    """Return provider prefix from model name (`provider:model`) or openai by default."""
    if ":" in model_name:
        return model_name.split(":", 1)[0].strip().lower()
    return "openai"


def require_runtime_keys() -> None:
    """Validate API keys needed by runtime chat model and embeddings."""
    provider = _provider_from_model(MODEL_NAME)
    provider_key_requirements = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google_genai": "GOOGLE_API_KEY",
        "groq": "GROQ_API_KEY",
        "cohere": "COHERE_API_KEY",
        "mistralai": "MISTRAL_API_KEY",
        "together": "TOGETHER_API_KEY",
        "fireworks": "FIREWORKS_API_KEY",
        # Local Ollama runtime does not require a cloud API key.
        "ollama": None,
    }
    required_key = provider_key_requirements.get(provider)
    if required_key is None and provider != "ollama":
        supported = ", ".join(sorted(provider_key_requirements.keys()))
        raise RuntimeError(
            f"Unsupported chat model provider '{provider}'. "
            f"Use one of: {supported}."
        )
    if required_key and not os.environ.get(required_key):
        raise RuntimeError(
            f"{required_key} is not set. "
            "Set it in .env or the environment for production."
        )
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
    if SECRET_KEY == "change-me-in-production":
        raise RuntimeError(
            "RAG_AGENT_SECRET_KEY is set to the insecure default value. "
            "Generate a strong secret with: python -c \"import secrets; print(secrets.token_hex(32))\" "
            "and set it as RAG_AGENT_SECRET_KEY in .env or the environment before starting."
        )


def _is_private_or_lan_url(url: str) -> bool:
    """True if the URL host is a private/LAN/loopback IP (192.168/10/172.16-31/127)."""
    host = (urlsplit(url).hostname or "").strip()
    if not host:
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local


def warn_oauth_redirect_misconfig() -> None:
    """Log (never raise) when OAuth/frontend URLs look like dev/LAN misconfig.

    Catches the classic "Authorize sends me to a 192.168.x.x address" bug early:
    the Monday redirect_uri (sent to Monday at authorize time, and what the browser
    is bounced back to) and the post-callback frontend redirect base must be your
    PUBLIC https origin, never a private/LAN IP or plain http.
    """
    log = logging.getLogger(__name__)
    redirect = (RAG_MONDAY_OAUTH_REDIRECT_URI or "").strip()
    if redirect:
        if _is_private_or_lan_url(redirect):
            log.warning(
                "RAG_MONDAY_OAUTH_REDIRECT_URI points at a private/LAN/loopback address "
                "(%s). Monday will bounce users' browsers there after Authorize and login "
                "will fail. Set it to your public https callback, e.g. "
                "https://platform.n8norlanda.com/auth/monday/callback (and register the same "
                "URL in the Monday developer app).",
                redirect,
            )
        elif redirect.lower().startswith("http://"):
            log.warning(
                "RAG_MONDAY_OAUTH_REDIRECT_URI uses plain http (%s); use https in production.",
                redirect,
            )
    if RAG_FRONTEND_BASE_URL:
        if _is_private_or_lan_url(RAG_FRONTEND_BASE_URL):
            log.warning(
                "RAG_FRONTEND_BASE_URL points at a private/LAN/loopback address (%s); the "
                "post-OAuth redirect will land users on an unreachable host. Set it to your "
                "public https origin (e.g. https://platform.n8norlanda.com) or leave it empty "
                "to use a same-origin relative redirect.",
                RAG_FRONTEND_BASE_URL,
            )
        elif RAG_FRONTEND_BASE_URL.lower().startswith("http://"):
            log.warning(
                "RAG_FRONTEND_BASE_URL uses plain http (%s); use https in production.",
                RAG_FRONTEND_BASE_URL,
            )
