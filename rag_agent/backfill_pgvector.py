"""
One-time backfill utility for PostgreSQL + pgvector RAG index.

Usage:
    python -m rag_agent.backfill_pgvector
"""
from rag_agent.indexing import reconcile_all_documents


def main() -> None:
    result = reconcile_all_documents()
    touched = result.get("touched") or []
    upserted = sum(1 for x in touched if x.get("status") == "upserted")
    unchanged = sum(1 for x in touched if x.get("status") == "unchanged")
    removed = int(result.get("removed") or 0)
    expected = int(result.get("expected_docs") or 0)
    print(f"Backfill complete. expected_docs={expected} upserted={upserted} unchanged={unchanged} removed={removed}")


if __name__ == "__main__":
    main()
