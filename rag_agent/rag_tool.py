"""
RAG tool: retrieves context from the indexed knowledge base for the agent.
"""
from langchain.tools import tool

from rag_agent.config import (
    RAG_MAX_CHARS_PER_CHUNK,
    RAG_MAX_TOTAL_CONTEXT_CHARS,
    RAG_RETRIEVE_TOP_K,
)
from rag_agent.indexing import load_vector_store

# Lazy-loaded vector store: loads from disk (index_store) when present;
# indexing runs only when the index is missing or you run: python -m rag_agent.indexing
_vector_store = None

# Sources from the last retrieve_context call(s) in this turn (for chat to display)
_last_sources: list[dict] = []


def get_last_sources() -> list[dict]:
    """Return and clear the list of document sources used in the last agent turn."""
    global _last_sources
    out = list(_last_sources)
    _last_sources = []
    return out


def invalidate_vector_store() -> None:
    """Clear the cached vector store so the next request will reload (e.g. after reindex)."""
    global _vector_store
    _vector_store = None


def _get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = load_vector_store()
    return _vector_store


def _trim_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


@tool(response_format="content_and_artifact")
def retrieve_context(query: str) -> tuple[str, str]:
    """
    Retrieve relevant context from the knowledge base to answer the user's query.
    Use this when you need information from the company's PDF documents.
    """
    try:
        store = _get_vector_store()
    except Exception as e:
        return (
            "Ошибка загрузки базы знаний. Проверьте подключение и наличие папки knowledge_base. "
            f"Детали: {e!s}",
            "",
        )

    if store is None:
        return (
            "Документы не проиндексированы. Добавьте PDF в папку knowledge_base и выполните: python -m rag_agent.indexing",
            "",
        )

    try:
        docs = store.similarity_search(query, k=max(1, RAG_RETRIEVE_TOP_K))
    except Exception as e:
        return (
            f"Ошибка поиска по базе знаний: {e!s}. Попробуйте переформулировать вопрос или переиндексировать документы.",
            "",
        )

    global _last_sources
    for doc in docs:
        _last_sources.append({
            "file": doc.metadata.get("source_file", doc.metadata.get("source", "?")),
            "page": doc.metadata.get("page", "?"),
        })
    parts: list[str] = []
    total_chars = 0
    max_total = max(1, RAG_MAX_TOTAL_CONTEXT_CHARS)
    max_per_chunk = max(1, RAG_MAX_CHARS_PER_CHUNK)
    for doc in docs:
        meta = doc.metadata
        content = _trim_text(doc.page_content, max_per_chunk)
        block = f"Source: {meta}\nContent: {content}"
        if parts and total_chars + len(block) + 2 > max_total:
            break
        if not parts and len(block) > max_total:
            block = _trim_text(block, max_total)
        parts.append(block)
        total_chars += len(block) + 2

    serialized = "\n\n".join(parts)
    return (serialized, serialized)
