"""
RAG tool: retrieves context from the indexed knowledge base for the agent.
"""
import logging
import re
from collections import defaultdict
from typing import Any

from langchain.tools import tool

from rag_agent.config import (
    RAG_BM25_TOP_K,
    RAG_CROSS_ENCODER_MODEL,
    RAG_ENABLE_CROSS_ENCODER_RERANK,
    RAG_ENABLE_HYBRID_RETRIEVAL,
    RAG_ENABLE_MMR,
    RAG_MAX_CHARS_PER_CHUNK,
    RAG_MAX_TOTAL_CONTEXT_CHARS,
    RAG_MMR_LAMBDA,
    RAG_NEIGHBOR_MAX_CHUNKS,
    RAG_NEIGHBOR_PAGE_WINDOW,
    RAG_QUERY_REWRITE_MAX,
    RAG_RERANK_CANDIDATES_K,
    RAG_RETRIEVAL_LOG_TOP,
    RAG_RETRIEVE_FETCH_K,
    RAG_RETRIEVE_TOP_K,
    RAG_RRF_K,
)
from rag_agent.indexing import load_vector_store

# Lazy-loaded vector store: loads from disk (index_store) when present;
# indexing runs only when the index is missing or you run: python -m rag_agent.indexing
_vector_store = None
_bm25_retriever = None
_bm25_corpus_size = 0
_cross_encoder = None
_cross_encoder_load_failed = False
logger = logging.getLogger(__name__)

# Sources from the last retrieve_context call(s) in this turn (for chat to display)
_last_sources: list[dict] = []


def get_last_sources() -> list[dict]:
    """Return and clear the list of document sources used in the last agent turn."""
    global _last_sources
    out = list(_last_sources)
    _last_sources = []
    return out


def invalidate_vector_store() -> None:
    """Clear caches so next request reloads stores (e.g. after reindex)."""
    global _vector_store, _bm25_retriever, _bm25_corpus_size
    _vector_store = None
    _bm25_retriever = None
    _bm25_corpus_size = 0


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


def _tokenize_simple(text: str) -> list[str]:
    return re.findall(r"[a-zA-Zа-яА-Я0-9_]+", (text or "").lower())


def _lexical_overlap_score(query: str, content: str) -> float:
    q_tokens = set(_tokenize_simple(query))
    if not q_tokens:
        return 0.0
    c_tokens = set(_tokenize_simple(content))
    if not c_tokens:
        return 0.0
    return len(q_tokens & c_tokens) / max(1.0, float(len(q_tokens)))


def _is_sequence_question(query: str) -> bool:
    q = (query or "").strip().lower()
    patterns = [
        r"\bwhat\s+next\b",
        r"\bwhat\s+comes\s+next\b",
        r"\bnext\s+step\b",
        r"\bafter\b",
        r"\bbefore\b",
        r"\bprevious\s+step\b",
        r"\bworkflow\b",
        r"\bsequence\b",
        r"\bэтап\b",
        r"\bшаг\b",
        r"\bследующ\w*\b",
        r"\bпосле\b",
        r"\bперед\b",
        r"\bдо\s+этого\b",
    ]
    return any(re.search(p, q) for p in patterns)


def _sequence_transition_score(query: str, content: str) -> float:
    if not _is_sequence_question(query):
        return 0.0
    c = (content or "").lower()
    hints = [
        "next step",
        "what comes next",
        "after",
        "before",
        "then",
        "followed by",
        "step ",
        "этап",
        "шаг",
        "следующ",
        "после",
        "перед",
        "далее",
        "затем",
    ]
    hits = sum(1 for h in hints if h in c)
    return min(1.0, hits / 4.0)


def _build_query_variants(query: str) -> list[str]:
    base = (query or "").strip()
    if not base:
        return []

    variants: list[str] = [base]
    if _is_sequence_question(base):
        variants.extend(
            [
                f"{base}. include next and previous process steps",
                f"{base}. workflow order and step sequence",
                f"{base}. what comes after and before this stage",
                f"{base}. последовательность шагов процесса",
            ]
        )
    deduped: list[str] = []
    seen = set()
    for v in variants:
        key = v.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(v.strip())
    return deduped[: max(1, RAG_QUERY_REWRITE_MAX)]


def _doc_key(doc) -> str:
    meta = getattr(doc, "metadata", {}) or {}
    source = str(meta.get("source_file", meta.get("source", "?")))
    page = str(meta.get("page", "?"))
    content = str(getattr(doc, "page_content", "") or "")
    return f"{source}|{page}|{content[:220]}"


def _sort_candidates(candidates: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        candidates.values(),
        key=lambda x: (
            x.get("dense_score") is None,
            x.get("dense_score") if x.get("dense_score") is not None else float("inf"),
            x.get("key", ""),
        ),
    )


def _all_docs_from_store(store) -> list:
    docstore = getattr(store, "docstore", None)
    if docstore is None:
        return []
    docs_map = getattr(docstore, "_dict", None)
    if isinstance(docs_map, dict):
        return [d for d in docs_map.values() if d is not None]
    return []


def _get_bm25_retriever(store):
    global _bm25_retriever, _bm25_corpus_size
    docs = _all_docs_from_store(store)
    if not docs:
        return None
    if _bm25_retriever is not None and _bm25_corpus_size == len(docs):
        return _bm25_retriever
    try:
        from langchain_community.retrievers import BM25Retriever
    except Exception:
        logger.warning("BM25 retriever unavailable: import failed")
        return None
    try:
        retriever = BM25Retriever.from_documents(docs)
        retriever.k = max(1, RAG_BM25_TOP_K)
        _bm25_retriever = retriever
        _bm25_corpus_size = len(docs)
        return _bm25_retriever
    except Exception:
        logger.exception("Failed to build BM25 retriever; continuing with dense retrieval only")
        return None


def _bm25_search(retriever, query: str) -> list:
    if retriever is None:
        return []
    try:
        docs = retriever.invoke(query)
        if isinstance(docs, list):
            return docs
    except Exception:
        pass
    try:
        docs = retriever.get_relevant_documents(query)
        return docs if isinstance(docs, list) else []
    except Exception:
        return []


def _get_cross_encoder():
    global _cross_encoder, _cross_encoder_load_failed
    if not RAG_ENABLE_CROSS_ENCODER_RERANK:
        return None
    if _cross_encoder is not None:
        return _cross_encoder
    if _cross_encoder_load_failed:
        return None
    try:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(RAG_CROSS_ENCODER_MODEL)
        return _cross_encoder
    except Exception:
        _cross_encoder_load_failed = True
        logger.warning(
            "Cross-encoder reranker unavailable for model '%s'; using heuristic reranker",
            RAG_CROSS_ENCODER_MODEL,
        )
        return None


def _upsert_candidate(
    candidates: dict[str, dict[str, Any]],
    *,
    doc,
    origin: str,
    rank: int | None = None,
    dense_score: float | None = None,
) -> None:
    key = _doc_key(doc)
    row = candidates.get(key)
    if row is None:
        row = {
            "key": key,
            "doc": doc,
            "dense_score": None,
            "dense_best_rank": None,
            "bm25_best_rank": None,
            "origins": set(),
            "rrf": 0.0,
            "rerank_score": 0.0,
            "rerank_reason": "pending",
        }
        candidates[key] = row

    row["origins"].add(origin)
    rrf_k = max(1, RAG_RRF_K)
    if isinstance(rank, int) and rank >= 1:
        row["rrf"] += 1.0 / (rrf_k + rank)
        if origin.startswith("dense") or origin.startswith("mmr"):
            prev = row.get("dense_best_rank")
            row["dense_best_rank"] = rank if prev is None else min(prev, rank)
        if origin.startswith("bm25"):
            prev = row.get("bm25_best_rank")
            row["bm25_best_rank"] = rank if prev is None else min(prev, rank)
    if isinstance(dense_score, (float, int)):
        val = float(dense_score)
        prev_score = row.get("dense_score")
        if prev_score is None or val < prev_score:
            row["dense_score"] = val


def _rerank_candidates(
    query: str,
    query_variants: list[str],
    rows: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    ce = _get_cross_encoder()
    if ce is not None:
        try:
            pairs = [(query, str(getattr(r["doc"], "page_content", "") or "")[:2500]) for r in rows]
            scores = ce.predict(pairs)
            for i, r in enumerate(rows):
                score = float(scores[i]) if i < len(scores) else 0.0
                r["rerank_score"] = score
                r["rerank_reason"] = "cross_encoder"
            rows.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            return rows[:limit]
        except Exception:
            logger.exception("Cross-encoder reranking failed; fallback to heuristic reranker")

    dense_scores = [r["dense_score"] for r in rows if isinstance(r.get("dense_score"), float)]
    dense_min = min(dense_scores) if dense_scores else 0.0
    dense_max = max(dense_scores) if dense_scores else 0.0
    dense_range = dense_max - dense_min

    variants = query_variants or [query]
    for r in rows:
        content = str(getattr(r["doc"], "page_content", "") or "")
        overlap = max(_lexical_overlap_score(v, content) for v in variants)
        transition = _sequence_transition_score(query, content)
        dense_score = r.get("dense_score")
        if isinstance(dense_score, float) and dense_range > 1e-9:
            dense_norm = 1.0 - ((dense_score - dense_min) / dense_range)
        elif isinstance(dense_score, float):
            dense_norm = 0.5
        else:
            dense_norm = 0.0

        rrf = float(r.get("rrf", 0.0))
        bm25_bonus = 0.06 if r.get("bm25_best_rank") is not None else 0.0
        score = (0.34 * dense_norm) + (0.38 * overlap) + (0.22 * rrf) + (0.12 * transition) + bm25_bonus
        r["rerank_score"] = score
        r["rerank_reason"] = "hybrid_heuristic"

    rows.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    return rows[:limit]


def _collect_neighbor_docs(store, seed_docs: list, *, page_window: int, max_chunks: int) -> list:
    if page_window <= 0 or max_chunks <= 0 or not seed_docs:
        return []
    all_docs = _all_docs_from_store(store)
    if not all_docs:
        return []

    by_source_page: dict[tuple[str, int], list] = defaultdict(list)
    for d in all_docs:
        meta = getattr(d, "metadata", {}) or {}
        source = str(meta.get("source_file", meta.get("source", ""))).strip()
        page_raw = meta.get("page", None)
        if not source or not isinstance(page_raw, int):
            continue
        by_source_page[(source, page_raw)].append(d)

    for k in by_source_page:
        by_source_page[k].sort(key=lambda d: len(str(getattr(d, "page_content", "") or "")), reverse=True)

    existing = {_doc_key(d) for d in seed_docs}
    neighbors: list = []
    for seed in seed_docs:
        meta = getattr(seed, "metadata", {}) or {}
        source = str(meta.get("source_file", meta.get("source", ""))).strip()
        page = meta.get("page", None)
        if not source or not isinstance(page, int):
            continue
        for offset in range(1, page_window + 1):
            for target_page in (page - offset, page + offset):
                bucket = by_source_page.get((source, target_page), [])
                for d in bucket:
                    k = _doc_key(d)
                    if k in existing:
                        continue
                    existing.add(k)
                    neighbors.append(d)
                    break
                if len(neighbors) >= max_chunks:
                    return neighbors
    return neighbors


def _log_retrieval_diagnostics(rows: list[dict]) -> None:
    if not rows:
        return
    limit = max(1, RAG_RETRIEVAL_LOG_TOP)
    shown = rows[:limit]
    parts = []
    for i, row in enumerate(shown, start=1):
        doc = row.get("doc")
        meta = getattr(doc, "metadata", {}) or {}
        source = str(meta.get("source_file", meta.get("source", "?")))
        page = meta.get("page", "?")
        dense_score = row.get("dense_score")
        dense_txt = f"{dense_score:.5f}" if isinstance(dense_score, (int, float)) else "n/a"
        rerank_score = row.get("rerank_score")
        rerank_txt = f"{rerank_score:.5f}" if isinstance(rerank_score, (int, float)) else "n/a"
        origins = ",".join(sorted(list(row.get("origins") or []))) or "?"
        parts.append(
            " ".join(
                [
                    f"{i}. source={source}",
                    f"page={page}",
                    f"dense={dense_txt}",
                    f"rerank={rerank_txt}",
                    f"origins={origins}",
                    f"reason={row.get('rerank_reason', 'n/a')}",
                ]
            )
        )
    logger.info("RAG retrieval top candidates:\n%s", "\n".join(parts))


def _run_retrieval_core(query: str) -> tuple[list, list[dict[str, Any]], list[str], str | None]:
    """
    Run retrieval pipeline and return:
    - final_docs,
    - ranked_rows,
    - query_variants,
    - error_text (if any).
    """
    try:
        store = _get_vector_store()
    except Exception as e:
        return (
            [],
            [],
            [],
            "Ошибка загрузки базы знаний. Проверьте подключение и наличие папки knowledge_base. "
            f"Детали: {e!s}",
        )

    if store is None:
        return (
            [],
            [],
            [],
            "Документы не проиндексированы. Добавьте PDF в папку knowledge_base и выполните: python -m rag_agent.indexing",
        )

    top_k = max(1, RAG_RETRIEVE_TOP_K)
    fetch_k = max(top_k, RAG_RETRIEVE_FETCH_K)
    query_variants = _build_query_variants(query)
    candidates: dict[str, dict[str, Any]] = {}

    try:
        for i, q in enumerate(query_variants):
            dense_origin = "dense_original" if i == 0 else "dense_rewrite"
            try:
                scored = store.similarity_search_with_score(q, k=fetch_k)
                for rank_i, pair in enumerate(scored, start=1):
                    doc, score = pair
                    _upsert_candidate(
                        candidates,
                        doc=doc,
                        origin=dense_origin,
                        rank=rank_i,
                        dense_score=float(score) if isinstance(score, (int, float)) else None,
                    )
            except Exception:
                docs_fallback = store.similarity_search(q, k=fetch_k)
                for rank_i, doc in enumerate(docs_fallback, start=1):
                    _upsert_candidate(candidates, doc=doc, origin=dense_origin, rank=rank_i)

        if RAG_ENABLE_MMR:
            lambda_mult = max(0.0, min(1.0, RAG_MMR_LAMBDA))
            for i, q in enumerate(query_variants):
                mmr_origin = "mmr_original" if i == 0 else "mmr_rewrite"
                mmr_docs = store.max_marginal_relevance_search(
                    q,
                    k=min(fetch_k, max(top_k * 2, top_k)),
                    fetch_k=fetch_k,
                    lambda_mult=lambda_mult,
                )
                for rank_i, doc in enumerate(mmr_docs, start=1):
                    _upsert_candidate(candidates, doc=doc, origin=mmr_origin, rank=rank_i)

        if RAG_ENABLE_HYBRID_RETRIEVAL:
            bm25 = _get_bm25_retriever(store)
            for i, q in enumerate(query_variants):
                bm25_origin = "bm25_original" if i == 0 else "bm25_rewrite"
                bm25_docs = _bm25_search(bm25, q)
                for rank_i, doc in enumerate(bm25_docs, start=1):
                    _upsert_candidate(candidates, doc=doc, origin=bm25_origin, rank=rank_i)
    except Exception as e:
        return (
            [],
            [],
            query_variants,
            f"Ошибка поиска по базе знаний: {e!s}. Попробуйте переформулировать вопрос или переиндексировать документы.",
        )

    pre_rank_rows = _sort_candidates(candidates)
    rerank_limit = max(top_k, RAG_RERANK_CANDIDATES_K)
    ranked_rows = _rerank_candidates(
        query=query,
        query_variants=query_variants,
        rows=pre_rank_rows[:rerank_limit],
        limit=rerank_limit,
    )
    ranked_docs = [row["doc"] for row in ranked_rows]

    seed_docs = ranked_docs[: min(len(ranked_docs), top_k)]
    neighbor_docs = _collect_neighbor_docs(
        store,
        seed_docs,
        page_window=max(0, RAG_NEIGHBOR_PAGE_WINDOW),
        max_chunks=max(0, RAG_NEIGHBOR_MAX_CHUNKS),
    )

    reserve_for_neighbors = min(len(neighbor_docs), max(1, top_k // 3)) if neighbor_docs else 0
    primary_take = max(1, top_k - reserve_for_neighbors)
    final_docs: list = []
    seen = set()
    for doc in ranked_docs[:primary_take]:
        k = _doc_key(doc)
        if k in seen:
            continue
        seen.add(k)
        final_docs.append(doc)
    for doc in neighbor_docs:
        if len(final_docs) >= top_k:
            break
        k = _doc_key(doc)
        if k in seen:
            continue
        seen.add(k)
        final_docs.append(doc)
    for doc in ranked_docs[primary_take:]:
        if len(final_docs) >= top_k:
            break
        k = _doc_key(doc)
        if k in seen:
            continue
        seen.add(k)
        final_docs.append(doc)

    if not final_docs:
        final_docs = ranked_docs[:top_k]

    return final_docs, ranked_rows, query_variants, None


def retrieval_debug(query: str, limit: int = 12) -> dict[str, Any]:
    """
    Return retrieval internals for diagnostics and evaluation.
    """
    final_docs, ranked_rows, query_variants, err = _run_retrieval_core(query)
    if err:
        return {"ok": False, "error": err, "query_variants": query_variants, "rows": []}

    selected_keys = {_doc_key(d) for d in final_docs}
    out_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(ranked_rows[: max(1, limit)], start=1):
        doc = row.get("doc")
        meta = getattr(doc, "metadata", {}) or {}
        text = str(getattr(doc, "page_content", "") or "")
        out_rows.append(
            {
                "rank": idx,
                "selected": _doc_key(doc) in selected_keys,
                "source": str(meta.get("source_file", meta.get("source", "?"))),
                "page": meta.get("page", "?"),
                "dense_score": row.get("dense_score"),
                "rerank_score": row.get("rerank_score"),
                "origins": sorted(list(row.get("origins") or [])),
                "reason": row.get("rerank_reason", ""),
                "snippet": _trim_text(text, 280),
            }
        )
    return {
        "ok": True,
        "query_variants": query_variants,
        "rows": out_rows,
        "selected_sources": [
            {
                "file": d.metadata.get("source_file", d.metadata.get("source", "?")),
                "page": d.metadata.get("page", "?"),
            }
            for d in final_docs
        ],
    }


@tool(response_format="content_and_artifact")
def retrieve_context(query: str) -> tuple[str, str]:
    """
    Retrieve relevant context from the knowledge base to answer the user's query.
    Use this when you need information from the company's PDF documents.
    """
    final_docs, ranked_rows, _query_variants, err = _run_retrieval_core(query)
    if err:
        return (err, "")

    _log_retrieval_diagnostics(ranked_rows)

    global _last_sources
    for doc in final_docs:
        _last_sources.append(
            {
                "file": doc.metadata.get("source_file", doc.metadata.get("source", "?")),
                "page": doc.metadata.get("page", "?"),
            }
        )

    parts: list[str] = []
    total_chars = 0
    max_total = max(1, RAG_MAX_TOTAL_CONTEXT_CHARS)
    max_per_chunk = max(1, RAG_MAX_CHARS_PER_CHUNK)
    for doc in final_docs:
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
