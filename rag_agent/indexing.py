"""
Index PDFs from the knowledge_base folder into a FAISS vector store.
Run this script to build or rebuild the index.

FAISS (Facebook AI Similarity Search) is a library for fast similarity search
on vectors; we use it to store document embeddings and retrieve relevant chunks.
Install faiss-gpu for GPU (requires NVIDIA GPU + CUDA); if install fails, use faiss-cpu.

Indexing PDFs in different languages is supported: OpenAI embeddings are
multilingual, so mixed-language documents in the same index work fine.
"""
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

# Paths: knowledge_base at project root, index stored inside rag_agent
RAG_AGENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RAG_AGENT_DIR.parent
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
INDEX_DIR = RAG_AGENT_DIR / "index_store"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 250


def get_pdf_paths():
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


def load_pdfs():
    """Load all PDFs from knowledge_base into LangChain documents."""
    pdf_paths = get_pdf_paths()
    if not pdf_paths:
        return []

    documents = []
    for path in pdf_paths:
        try:
            sidecar = rag_sidecar_path(path)
            if sidecar.is_file():
                raw = sidecar.read_text(encoding="utf-8", errors="replace")
                if raw.strip():
                    doc = Document(
                        page_content=raw.strip(),
                        metadata={"source_file": path.name, "page": 0},
                    )
                    documents.append(doc)
                    continue
            loader = PyPDFLoader(str(path))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_file"] = path.name
            documents.extend(docs)
        except Exception as e:
            print(f"Warning: could not load {path}: {e}", file=sys.stderr)

    return documents


def load_all_documents():
    """Load PDFs and text knowledge items into one list of LangChain documents."""
    from rag_agent.knowledge_items import items_to_documents
    docs = load_pdfs() + items_to_documents()
    return docs


def build_index():
    """Load PDFs + knowledge items, chunk, embed, and persist FAISS index."""
    documents = load_all_documents()
    if not documents:
        print("No documents: add PDFs to knowledge_base or create text knowledge items.")
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(INDEX_DIR))
    print(f"Indexed {len(chunks)} chunks from {len(documents)} pages into {INDEX_DIR}")
    return vector_store


def load_vector_store():
    """Load the persisted FAISS index. Builds it if missing."""
    embeddings = OpenAIEmbeddings()
    if INDEX_DIR.is_dir() and any(INDEX_DIR.iterdir()):
        return FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return build_index()


def clear_index() -> None:
    """Remove persisted index files (e.g. after deleting all PDFs)."""
    import shutil
    if INDEX_DIR.is_dir():
        shutil.rmtree(INDEX_DIR, ignore_errors=True)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)


def list_knowledge_files() -> list[dict]:
    """Return list of PDFs in knowledge_base: [{path, name, size}, ...]. path is relative to knowledge_base."""
    if not KNOWLEDGE_BASE_DIR.is_dir():
        return []
    out = []
    for p in KNOWLEDGE_BASE_DIR.glob("**/*.pdf"):
        if not p.is_file():
            continue
        try:
            rel = p.relative_to(KNOWLEDGE_BASE_DIR)
            out.append({
                "path": str(rel).replace("\\", "/"),
                "name": p.name,
                "size": p.stat().st_size,
            })
        except ValueError:
            continue
    return sorted(out, key=lambda x: x["path"])


if __name__ == "__main__":
    build_index()
