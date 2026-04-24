"""
Simple retrieval evaluation script for the current RAG pipeline.

Input format: JSONL (one case per line), e.g.
{"query":"what is next step after estimation","must_include_any":["shop drawing"]}
{"query":"who approves X","must_include_all":["approval","manager"]}

Run:
python -m rag_agent.eval_retrieval --dataset rag_agent/data/retrieval_eval.jsonl --k 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_agent.rag_tool import retrieval_debug


def _load_cases(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    cases: list[dict] = []
    # Use utf-8-sig so files saved with BOM on Windows still parse correctly.
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            # Keep evaluation running even when one line is malformed.
            continue
        if isinstance(obj, dict) and str(obj.get("query", "")).strip():
            cases.append(obj)
    return cases


def _contains_all(text: str, tokens: list[str]) -> bool:
    t = text.lower()
    return all(str(x).strip().lower() in t for x in (tokens or []) if str(x).strip())


def _contains_any(text: str, tokens: list[str]) -> bool:
    t = text.lower()
    checks = [str(x).strip().lower() for x in (tokens or []) if str(x).strip()]
    if not checks:
        return False
    return any(x in t for x in checks)


def _score_case(case: dict, k: int) -> dict:
    query = str(case.get("query", "")).strip()
    must_any = list(case.get("must_include_any") or [])
    must_all = list(case.get("must_include_all") or [])
    res = retrieval_debug(query, limit=max(1, k))
    rows = list(res.get("rows") or [])[: max(1, k)]

    if not res.get("ok"):
        return {
            "query": query,
            "ok": False,
            "hit": False,
            "mrr": 0.0,
            "error": res.get("error", "unknown retrieval error"),
        }

    first_hit_rank = None
    for idx, row in enumerate(rows, start=1):
        joined = " ".join(
            [
                str(row.get("source", "")),
                str(row.get("snippet", "")),
            ]
        )
        hit_any = _contains_any(joined, must_any) if must_any else True
        hit_all = _contains_all(joined, must_all) if must_all else True
        if hit_any and hit_all:
            first_hit_rank = idx
            break

    hit = first_hit_rank is not None
    mrr = (1.0 / float(first_hit_rank)) if first_hit_rank else 0.0
    return {
        "query": query,
        "ok": True,
        "hit": hit,
        "mrr": mrr,
        "first_hit_rank": first_hit_rank,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality on a JSONL golden set.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="rag_agent/data/retrieval_eval.jsonl",
        help="Path to JSONL file with eval cases",
    )
    parser.add_argument("--k", type=int, default=8, help="Top-K rows to inspect per query")
    parser.add_argument("--show-failures", action="store_true", help="Print failed queries")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    cases = _load_cases(dataset_path)
    if not cases:
        print("No valid cases found.")
        return

    scored = [_score_case(c, k=max(1, args.k)) for c in cases]
    total = len(scored)
    ok_count = sum(1 for s in scored if s.get("ok"))
    hits = [s for s in scored if s.get("ok") and s.get("hit")]
    hit_rate = (len(hits) / ok_count) if ok_count else 0.0
    mrr = (sum(float(s.get("mrr", 0.0)) for s in scored if s.get("ok")) / ok_count) if ok_count else 0.0
    failures = [s for s in scored if not s.get("hit")]

    print(f"Dataset: {dataset_path}")
    print(f"Cases: {total}")
    print(f"Evaluated: {ok_count}")
    print(f"Hit@{max(1, args.k)}: {hit_rate:.3f}")
    print(f"MRR@{max(1, args.k)}: {mrr:.3f}")
    print(f"Failures: {len(failures)}")

    if args.show_failures and failures:
        print("\nFailed queries:")
        for f in failures:
            q = str(f.get("query", "")).strip()
            err = str(f.get("error", "")).strip()
            if err:
                print(f"- {q}  [error: {err}]")
            else:
                print(f"- {q}")


if __name__ == "__main__":
    main()
