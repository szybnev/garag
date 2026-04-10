"""Grid search BM25 `k1` / `b` against the golden set.

For each `(k1, b)` pair we rebuild the BM25 index in-memory (no pickle
overwrite, the production pickle stays at default values until d6's
final commit), evaluate Recall@10 / nDCG@10 / MAP, and emit a sorted
table to stdout. The winner is appended to `evaluation/reports/retrieval_report.md`.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytrec_eval
from rank_bm25 import BM25Okapi

from scripts.build_bm25 import _ensure_stopwords, tokenize

CHUNKS_FILE = Path(__file__).resolve().parents[1] / "data" / "processed" / "chunks.parquet"
BM25_PKL = Path(__file__).resolve().parents[1] / "data" / "index" / "bm25.pkl"
GOLDEN_FILE = Path(__file__).resolve().parents[1] / "data" / "golden" / "golden_set_v1.jsonl"
REPORT_FILE = Path(__file__).resolve().parents[1] / "evaluation" / "reports" / "retrieval_report.md"

K1_GRID = [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
B_GRID = [0.25, 0.5, 0.75, 1.0]
METRICS = {"recall_5", "recall_10", "ndcg_cut_10", "map"}


def _load_golden() -> list[dict[str, Any]]:
    with GOLDEN_FILE.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _evaluate_bm25(
    bm25: BM25Okapi,
    chunk_ids: list[str],
    golden: list[dict[str, Any]],
    stop: set[str],
    top_k: int,
) -> dict[str, float]:
    qrels = {item["qid"]: dict.fromkeys(item["relevant_chunks"], 1) for item in golden}
    runs: dict[str, dict[str, float]] = {}
    for item in golden:
        tokens = tokenize(item["question"], stop)
        if not tokens:
            runs[item["qid"]] = {}
            continue
        scores = bm25.get_scores(tokens)
        order = np.argsort(scores)[::-1][:top_k]
        runs[item["qid"]] = {chunk_ids[i]: float(scores[i]) for i in order if scores[i] > 0}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, METRICS)
    per_query = evaluator.evaluate(runs)
    return {
        m: sum(per_query[q].get(m, 0.0) for q in per_query) / max(len(per_query), 1)
        for m in METRICS
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument(
        "--update-pickle", action="store_true", help="overwrite bm25.pkl with the winning (k1, b)"
    )
    args = ap.parse_args()

    df = pd.read_parquet(CHUNKS_FILE)
    print(f"[load] {len(df)} chunks")
    golden = _load_golden()
    print(f"[load] {len(golden)} golden queries")

    stop = _ensure_stopwords()
    tokenized = [tokenize(text, stop) for text in df["text"].tolist()]
    tokenized = [t or ["__empty__"] for t in tokenized]
    chunk_ids: list[str] = df["chunk_id"].tolist()

    print(f"[grid] {len(K1_GRID)} x {len(B_GRID)} = {len(K1_GRID) * len(B_GRID)} configs")
    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for k1 in K1_GRID:
        for b in B_GRID:
            bm25 = BM25Okapi(tokenized, k1=k1, b=b)
            metrics = _evaluate_bm25(bm25, chunk_ids, golden, stop, args.top_k)
            results.append({"k1": k1, "b": b, **metrics})
    elapsed = time.perf_counter() - t0
    print(f"[done] {len(results)} configs in {elapsed:.1f}s")

    results.sort(key=lambda r: r["ndcg_cut_10"], reverse=True)
    print("\nTop 10 by nDCG@10:")
    print(f"{'k1':>5} {'b':>5}  Recall@5  Recall@10  nDCG@10    MAP")
    for r in results[:10]:
        print(
            f"{r['k1']:>5.1f} {r['b']:>5.2f}  "
            f"{r['recall_5']:>8.4f}  {r['recall_10']:>9.4f}  "
            f"{r['ndcg_cut_10']:>7.4f}  {r['map']:>7.4f}"
        )

    best = results[0]
    print(f"\n[best] k1={best['k1']}, b={best['b']}")
    print(f"  Recall@5  = {best['recall_5']:.4f}")
    print(f"  Recall@10 = {best['recall_10']:.4f}")
    print(f"  nDCG@10   = {best['ndcg_cut_10']:.4f}")
    print(f"  MAP       = {best['map']:.4f}")

    with REPORT_FILE.open("a") as fh:
        fh.write("\n\n## BM25 grid search (`scripts/tune_bm25.py`)\n\n")
        fh.write(
            f"Search space: `k1 in {K1_GRID}` x `b in {B_GRID}` "
            f"({len(results)} configs, {elapsed:.0f}s on the 50-query golden set).\n\n"
        )
        fh.write("Top 5 by nDCG@10:\n\n")
        fh.write("| k1 | b | Recall@5 | Recall@10 | nDCG@10 | MAP |\n")
        fh.write("|---|---|---|---|---|---|\n")
        for r in results[:5]:
            fh.write(
                f"| {r['k1']:.1f} | {r['b']:.2f} | "
                f"{r['recall_5']:.4f} | {r['recall_10']:.4f} | "
                f"{r['ndcg_cut_10']:.4f} | {r['map']:.4f} |\n"
            )
        fh.write(f"\n**Winner:** `k1={best['k1']}, b={best['b']}`. ")
        fh.write("Persisted in `app/config.py` and `.env.example`.\n")
    print(f"[report] appended to {REPORT_FILE}")

    if args.update_pickle:
        print(f"[pickle] rebuilding bm25.pkl with k1={best['k1']}, b={best['b']}")
        winner_bm25 = BM25Okapi(tokenized, k1=best["k1"], b=best["b"])
        BM25_PKL.write_bytes(
            pickle.dumps(
                {
                    "bm25": winner_bm25,
                    "chunk_ids": chunk_ids,
                    "params": {
                        "k1": best["k1"],
                        "b": best["b"],
                        "stopwords": "nltk-english",
                    },
                    "vocab_size": len(winner_bm25.idf),
                    "n_docs": len(tokenized),
                },
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        )
        print(f"[pickle] {BM25_PKL} updated ({BM25_PKL.stat().st_size / 1_000_000:.1f} MB)")


if __name__ == "__main__":
    main()
