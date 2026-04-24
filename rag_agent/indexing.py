"""
Incremental PostgreSQL + pgvector indexing and retrieval adapter.

This module replaces FAISS persistence with document-level upsert/delete flows.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from rag_agent.config import RAG_EMBEDDING_MODEL, RAG_VECTOR_DIM
from rag_agent.db.models import DocumentChunk, DocumentIndexRecord
from rag_agent.db.session import get_session_factory

load_dotenv()

# Paths: knowledge_base at project root.
RAG_AGENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RAG_AGENT_DIR.parent
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 250
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
)
_embeddings = OpenAIEmbeddings(model=RAG_EMBEDDING_MODEL)


class _DocStore:
    """Minimal docstore-like object expected by rag_tool BM25 helper."""

    def __init__(self, docs: list[Document]):
        self._dict = {str(i): d for i, d in enumerate(docs)}


class PostgresVectorStoreAdapter:
    """FAISS-like adapter used by rag_tool retrieval pipeline."""

    def __init__(self) -> None:
        self._session_factory = get_session_factory()
        self._docstore: _DocStore | None = None

    @property
    def docstore(self):
        if self._docstore is None:
            docs = _load_all_chunk_documents()
            self._docstore = _DocStore(docs)
        return self._docstore

    def _query_rows(self, query: str, k: int) -> list[tuple[DocumentChunk, float]]:
        q_vec = _embeddings.embed_query((query or "").strip())
        with self._session_factory() as db:
            distance = DocumentChunk.embedding.cosine_distance(q_vec)
            stmt = (
                select(DocumentChunk, distance.label("distance"))
                .order_by(distance.asc())
                .limit(max(1, int(k)))
            )
            rows = db.execute(stmt).all()
        return rows

    def similarity_search_with_score(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        rows = self._query_rows(query, k)
        out: list[tuple[Document, float]] = []
        for chunk, distance in rows:
            meta = _normalized_meta(chunk)
            out.append((Document(page_content=chunk.chunk_text, metadata=meta), float(distance)))
        return out

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k)]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ) -> list[Document]:
        q_vec = _embeddings.embed_query((query or "").strip())
        with self._session_factory() as db:
            distance = DocumentChunk.embedding.cosine_distance(q_vec)
            stmt = (
                select(DocumentChunk, distance.label("distance"))
                .order_by(distance.asc())
                .limit(max(int(k), int(fetch_k), 1))
            )
            rows = db.execute(stmt).all()
        chunks = [row[0] for row in rows]
        if not chunks:
            return []
        candidate_vecs = [list(c.embedding) if c.embedding is not None else [] for c in chunks]
        selected_idx = _mmr_select(
            query_embedding=q_vec,
            candidate_embeddings=candidate_vecs,
            k=max(1, int(k)),
            lambda_mult=max(0.0, min(1.0, float(lambda_mult))),
        )
        return [Document(page_content=chunks[i].chunk_text, metadata=_normalized_meta(chunks[i])) for i in selected_idx]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if a is None or b is None:
        return 0.0
    if len(a) == 0 or len(b) == 0 or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (na * nb)


def _mmr_select(
    query_embedding: list[float],
    candidate_embeddings: list[list[float]],
    k: int,
    lambda_mult: float,
) -> list[int]:
    if not candidate_embeddings:
        return []
    selected: list[int] = []
    remaining = list(range(len(candidate_embeddings)))
    while remaining and len(selected) < k:
        best_idx = None
        best_score = -float("inf")
        for idx in remaining:
            relevance = _cosine_similarity(query_embedding, candidate_embeddings[idx])
            diversity = 0.0
            if selected:
                diversity = max(
                    _cosine_similarity(candidate_embeddings[idx], candidate_embeddings[s])
                    for s in selected
                )
            mmr = (lambda_mult * relevance) - ((1.0 - lambda_mult) * diversity)
            if mmr > best_score:
                best_score = mmr
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected


def _normalized_meta(chunk: DocumentChunk) -> dict:
    meta = dict(chunk.metadata_json or {})
    meta.setdefault("source_file", chunk.source_file)
    meta.setdefault("page", chunk.page)
    meta.setdefault("doc_id", chunk.doc_id)
    meta.setdefault("chunk_no", chunk.chunk_no)
    return meta


def _hash_documents(documents: list[Document]) -> str:
    h = hashlib.sha256()
    for doc in documents:
        content = (doc.page_content or "").strip()
        h.update(content.encode("utf-8", errors="replace"))
        meta = doc.metadata or {}
        keys = sorted(str(k) for k in meta.keys())
        for k in keys:
            h.update(k.encode("utf-8"))
            h.update(str(meta.get(k, "")).encode("utf-8", errors="replace"))
    return h.hexdigest()


def _normalize_rel(path: str | Path) -> str:
    if isinstance(path, Path):
        path = str(path)
    return str(path or "").strip().replace("\\", "/").lstrip("/")


def _pdf_doc_id(rel_path: str) -> str:
    return f"pdf:{_normalize_rel(rel_path)}"


def _item_doc_id(item_id: str) -> str:
    return f"item:{(item_id or '').strip()}"


def get_pdf_doc_id(rel_path: str) -> str:
    return _pdf_doc_id(rel_path)


def get_item_doc_id(item_id: str) -> str:
    return _item_doc_id(item_id)


def get_pdf_paths() -> list[Path]:
    """Return list of PDF paths in knowledge_base."""
    if not KNOWLEDGE_BASE_DIR.is_dir():
        return []
    return list(KNOWLEDGE_BASE_DIR.glob("**/*.pdf"))


def rag_sidecar_path(pdf_path: Path) -> Path:
    """Sibling file `{stem}.rag.txt` overrides PyPDF extraction for RAG."""
    return pdf_path.with_name(pdf_path.stem + ".rag.txt")


def extract_pdf_plain_text(pdf_path: Path) -> str:
    """Concatenate page text from a PDF (for API preview when no override exists)."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    for page in reader.pages:
        try:
            t = page.extract_text()
            if t:
                parts.append(t)
        except Exception:
            continue
    return "\n\n".join(parts).strip()


def _load_pdf_documents(pdf_path: Path) -> list[Document]:
    rel = str(pdf_path.relative_to(KNOWLEDGE_BASE_DIR)).replace("\\", "/")
    sidecar = rag_sidecar_path(pdf_path)
    if sidecar.is_file():
        raw = sidecar.read_text(encoding="utf-8", errors="replace")
        if raw.strip():
            return [
                Document(
                    page_content=raw.strip(),
                    metadata={"source_file": rel, "page": 0, "type": "pdf", "path": rel},
                )
            ]
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    for d in docs:
        page_raw = d.metadata.get("page", 0)
        page = int(page_raw) if isinstance(page_raw, int) else 0
        d.metadata["source_file"] = rel
        d.metadata["page"] = page
        d.metadata["type"] = "pdf"
        d.metadata["path"] = rel
    return docs


def load_pdfs() -> list[Document]:
    """Load all PDFs from knowledge_base into LangChain documents."""
    documents: list[Document] = []
    for path in get_pdf_paths():
        try:
            documents.extend(_load_pdf_documents(path))
        except Exception:
            continue
    return documents


def load_all_documents() -> list[Document]:
    """Load PDFs and text knowledge items into one list of LangChain documents."""
    from rag_agent.knowledge_items import items_to_documents

    return load_pdfs() + items_to_documents()


def list_knowledge_files() -> list[dict]:
    """Return list of PDFs in knowledge_base: [{path, name, size}, ...]."""
    if not KNOWLEDGE_BASE_DIR.is_dir():
        return []
    out = []
    for p in KNOWLEDGE_BASE_DIR.glob("**/*.pdf"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(KNOWLEDGE_BASE_DIR)).replace("\\", "/")
        out.append({"path": rel, "name": p.name, "size": p.stat().st_size})
    return sorted(out, key=lambda x: x["path"])


def _chunks_from_documents(documents: list[Document]) -> list[Document]:
    if not documents:
        return []
    return _splitter.split_documents(documents)


def _upsert_document(
    db: Session,
    *,
    doc_id: str,
    doc_type: str,
    source_ref: str,
    source_name: str,
    documents: list[Document],
) -> dict:
    doc_hash = _hash_documents(documents)
    existing = db.get(DocumentIndexRecord, doc_id)
    if existing and existing.content_hash == doc_hash:
        return {"status": "unchanged", "doc_id": doc_id, "chunks": existing.chunk_count}

    chunks = _chunks_from_documents(documents)
    if not chunks:
        delete_document(doc_id, db=db)
        return {"status": "deleted_empty", "doc_id": doc_id, "chunks": 0}

    texts = [(c.page_content or "").strip() for c in chunks]
    vectors = _embeddings.embed_documents(texts)
    for vec in vectors:
        if len(vec) != RAG_VECTOR_DIM:
            raise RuntimeError(
                f"Embedding dimension mismatch: got {len(vec)}, expected {RAG_VECTOR_DIM}. "
                "Adjust RAG_VECTOR_DIM or embedding model."
            )

    db.execute(delete(DocumentChunk).where(DocumentChunk.doc_id == doc_id))
    if existing is None:
        existing = DocumentIndexRecord(
            doc_id=doc_id,
            doc_type=doc_type,
            source_ref=source_ref,
            source_name=source_name,
            content_hash=doc_hash,
            chunk_count=len(chunks),
        )
        db.add(existing)
        # Ensure parent row exists before inserting child chunk rows (FK rag_document_chunks.doc_id).
        db.flush()
    else:
        existing.doc_type = doc_type
        existing.source_ref = source_ref
        existing.source_name = source_name
        existing.content_hash = doc_hash
        existing.chunk_count = len(chunks)

    for i, chunk in enumerate(chunks):
        meta = dict(chunk.metadata or {})
        source_file = str(meta.get("source_file", source_name))
        page = int(meta.get("page", 0) or 0)
        row = DocumentChunk(
            doc_id=doc_id,
            chunk_no=i,
            chunk_text=texts[i],
            search_text=texts[i],
            source_file=source_file,
            page=page,
            metadata_json=meta,
            embedding=vectors[i],
        )
        db.add(row)

    return {"status": "upserted", "doc_id": doc_id, "chunks": len(chunks)}


def delete_document(doc_id: str, *, db: Session | None = None) -> bool:
    own = db is None
    session = db or get_session_factory()()
    try:
        existed = session.get(DocumentIndexRecord, doc_id)
        if not existed:
            return False
        session.execute(delete(DocumentChunk).where(DocumentChunk.doc_id == doc_id))
        session.execute(delete(DocumentIndexRecord).where(DocumentIndexRecord.doc_id == doc_id))
        if own:
            session.commit()
        return True
    finally:
        if own:
            session.close()


def upsert_pdf_document(rel_path: str) -> dict:
    rel = _normalize_rel(rel_path)
    target = (KNOWLEDGE_BASE_DIR / rel).resolve()
    target.relative_to(KNOWLEDGE_BASE_DIR.resolve())
    if not target.is_file():
        raise FileNotFoundError(rel)
    docs = _load_pdf_documents(target)
    doc_id = _pdf_doc_id(rel)
    with get_session_factory()() as db:
        result = _upsert_document(
            db,
            doc_id=doc_id,
            doc_type="pdf",
            source_ref=rel,
            source_name=target.name,
            documents=docs,
        )
        db.commit()
    return result


def upsert_knowledge_item(item_id: str) -> dict:
    from rag_agent.knowledge_items import get_item

    item = get_item(item_id)
    if not item:
        deleted = delete_document(_item_doc_id(item_id))
        return {"status": "deleted_missing", "doc_id": _item_doc_id(item_id), "deleted": deleted}
    content = (item.get("content") or "").strip()
    if not content:
        deleted = delete_document(_item_doc_id(item_id))
        return {"status": "deleted_empty", "doc_id": _item_doc_id(item_id), "deleted": deleted}
    name = (item.get("name") or "Без названия").strip() or "Без названия"
    doc = Document(
        page_content=content,
        metadata={
            "source_file": name,
            "page": 0,
            "type": "knowledge_item",
            "id": item.get("id", ""),
        },
    )
    with get_session_factory()() as db:
        result = _upsert_document(
            db,
            doc_id=_item_doc_id(item_id),
            doc_type="knowledge_item",
            source_ref=str(item.get("id") or ""),
            source_name=name,
            documents=[doc],
        )
        db.commit()
    return result


def delete_pdf_document(rel_path: str) -> bool:
    return delete_document(_pdf_doc_id(_normalize_rel(rel_path)))


def delete_knowledge_item_document(item_id: str) -> bool:
    return delete_document(_item_doc_id(item_id))


def reconcile_all_documents() -> dict:
    from rag_agent.knowledge_items import list_items

    touched: list[dict] = []
    expected_ids: set[str] = set()

    for pdf in list_knowledge_files():
        rel = str(pdf.get("path") or "")
        if not rel:
            continue
        result = upsert_pdf_document(rel)
        touched.append(result)
        expected_ids.add(_pdf_doc_id(rel))

    for item in list_items():
        item_id = str(item.get("id") or "").strip()
        if not item_id:
            continue
        result = upsert_knowledge_item(item_id)
        touched.append(result)
        # keep ids only for non-empty existing items
        content = (item.get("content") or "").strip()
        if content:
            expected_ids.add(_item_doc_id(item_id))

    removed = 0
    with get_session_factory()() as db:
        rows = db.execute(select(DocumentIndexRecord.doc_id)).all()
        stale = [doc_id for (doc_id,) in rows if doc_id not in expected_ids]
        for doc_id in stale:
            delete_document(doc_id, db=db)
            removed += 1
        db.commit()

    return {"ok": True, "touched": touched, "removed": removed, "expected_docs": len(expected_ids)}


def build_index():
    """
    Compatibility entrypoint: reconcile all sources into pgvector tables.
    """
    return reconcile_all_documents()


def clear_index() -> None:
    """Clear all indexed documents/chunks from PostgreSQL tables."""
    with get_session_factory()() as db:
        db.execute(delete(DocumentChunk))
        db.execute(delete(DocumentIndexRecord))
        db.commit()


def _load_all_chunk_documents() -> list[Document]:
    with get_session_factory()() as db:
        rows = db.execute(
            select(DocumentChunk).order_by(
                DocumentChunk.source_file.asc(),
                DocumentChunk.page.asc(),
                DocumentChunk.chunk_no.asc(),
            )
        ).scalars().all()
    docs: list[Document] = []
    for row in rows:
        docs.append(Document(page_content=row.chunk_text, metadata=_normalized_meta(row)))
    return docs


def load_vector_store():
    """Return pgvector-backed adapter used by retrieval pipeline."""
    return PostgresVectorStoreAdapter()


if __name__ == "__main__":
    result = reconcile_all_documents()
    print(f"Indexed documents: {result.get('expected_docs', 0)}")
