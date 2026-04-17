"""
Step 2: Configuration and environment.
Loads paths and LLM settings from env and defaults.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (parent of src/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def _str(value: str | None, default: str) -> str:
    return (value or "").strip() or default


# Paths
KNOWLEDGE_DIR = Path(_str(os.getenv("KNOWLEDGE_DIR"), "company_knowledge"))
if not KNOWLEDGE_DIR.is_absolute():
    KNOWLEDGE_DIR = _PROJECT_ROOT / KNOWLEDGE_DIR

CHROMA_DIR = Path(_str(os.getenv("CHROMA_DIR"), "data/chroma"))
if not CHROMA_DIR.is_absolute():
    CHROMA_DIR = _PROJECT_ROOT / CHROMA_DIR

# LLM (OpenAI-compatible API)
OPENAI_API_BASE = _str(os.getenv("OPENAI_API_BASE"), "http://localhost:1234/v1")
OPENAI_API_KEY = _str(os.getenv("OPENAI_API_KEY"), "not-needed-for-local")
OPENAI_MODEL = _str(os.getenv("OPENAI_MODEL"), "llama3.2")

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# RAG
TOP_K = 5
COLLECTION_NAME = "orlanda_onboarding"
