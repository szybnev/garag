"""Grid search for hybrid fusion parameters.

Sweeps `alpha ∈ [0.0, 1.0]` for the alpha-weighted fusion and reports
which value maximises nDCG@10 / MAP. Compares the alpha optimum against
RRF (which has no tunable knob in our setup, `k=60` is fixed) and emits
the winner to `evaluation/reports/retrieval_report.md`.

Run **after** `scripts/tune_bm25.py --update-pickle` so the sparse
retriever uses the tuned BM25 parameters.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pytrec_eval

from app.rag.fusion import alpha_weighted_fusion, reciprocal_rank_fusion
from app.rag.retriever_dense import DenseRetriever
from app.rag.retriever_sparse import SparseRetriever

GOLDEN_FILE = Path(__file__).resolve().parents[1] / "data" / "golden" / "golden_set_v1.jsonl"
REPORT_FILE = Path(__file__).resolve().parents[1] / "evaluation" / "reports" / "retrieval_report.md"

ALPHA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
METRICS = {"recall_5", "recall_10", "ndcg_cut_10", "map"}


def _load_golden() -> list[dict[str, Any]]:
    with GOLDEN_FILE.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _evaluate(
    runs: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
) -> dict[str, float]:
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, METRICS)
    per_query = evaluator.evaluate(runs)
    return {
        m: sum(per_query[q].get(m, 0.0) for q in per_query) / max(len(per_query), 1)
        for m in METRICS
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    golden = _load_golden()
    print(f"[load] {len(golden)} golden queries")
    qrels = {item["qid"]: dict.fromkeys(item["relevant_chunks"], 1) for item in golden}

    print("[init] dense + sparse retrievers")
    dense = DenseRetriever()
    sparse = SparseRetriever()

    print("[cache] running dense + sparse once for all queries")
    t0 = time.perf_counter()
    cache: dict[str, tuple[list[Any], list[Any]]] = {}
    for item in golden:
        d = dense.search(item["question"], top_k=args.top_k)
        s = sparse.search(item["question"], top_k=args.top_k)
        cache[item["qid"]] = (d, s)
    print(f"[cache] {time.perf_counter() - t0:.1f}s")

    # RRF baseline
    rrf_runs: dict[str, dict[str, float]] = {}
    for qid, (d, s) in cache.items():
        fused = reciprocal_rank_fusion(d, s, top_k=args.top_k)
        rrf_runs[qid] = {h.chunk_id: float(h.score) for h in fused}
    rrf_metrics = _evaluate(rrf_runs, qrels)
    print(f"[RRF k=60] nDCG@10={rrf_metrics['ndcg_cut_10']:.4f} MAP={rrf_metrics['map']:.4f}")

    # alpha sweep
    rows: list[dict[str, Any]] = []
    for alpha in ALPHA_GRID:
        runs: dict[str, dict[str, float]] = {}
        for qid, (d, s) in cache.items():
            fused = alpha_weighted_fusion(d, s, alpha=alpha, top_k=args.top_k)
            runs[qid] = {h.chunk_id: float(h.score) for h in fused}
        m = _evaluate(runs, qrels)
        rows.append({"alpha": alpha, **m})
        print(
            f"  alpha={alpha:.1f}  nDCG@10={m['ndcg_cut_10']:.4f}  "
            f"MAP={m['map']:.4f}  Recall@5={m['recall_5']:.4f}"
        )

    rows.sort(key=lambda r: r["ndcg_cut_10"], reverse=True)
    best = rows[0]
    print(f"\n[best alpha] {best['alpha']:.1f} → nDCG@10={best['ndcg_cut_10']:.4f}")

    if best["ndcg_cut_10"] >= rrf_metrics["ndcg_cut_10"]:
        winner = f"alpha-weighted (alpha={best['alpha']:.1f})"
        winner_metrics = best
    else:
        winner = "RRF (k=60)"
        winner_metrics = {"alpha": None, **rrf_metrics}
    print(f"[winner] {winner}")

    with REPORT_FILE.open("a") as fh:
        fh.write("\n\n## Fusion grid search (`scripts/tune_fusion.py`)\n\n")
        fh.write(
            f"Search space: `alpha in {ALPHA_GRID}` plus RRF baseline. "
            f"Run on the 50-query golden set with `top_k={args.top_k}` candidates "
            f"after BM25 tuning (k1=0.8, b=0.5).\n\n"
        )
        fh.write(
            f"**RRF baseline (k=60):** nDCG@10={rrf_metrics['ndcg_cut_10']:.4f}, "
            f"MAP={rrf_metrics['map']:.4f}, "
            f"Recall@10={rrf_metrics['recall_10']:.4f}\n\n"
        )
        fh.write("Alpha sweep top 5 by nDCG@10:\n\n")
        fh.write("| alpha | Recall@5 | Recall@10 | nDCG@10 | MAP |\n")
        fh.write("|---|---|---|---|---|\n")
        for r in rows[:5]:
            fh.write(
                f"| {r['alpha']:.1f} | "
                f"{r['recall_5']:.4f} | {r['recall_10']:.4f} | "
                f"{r['ndcg_cut_10']:.4f} | {r['map']:.4f} |\n"
            )
        fh.write(
            f"\n**Winner:** {winner} → "
            f"nDCG@10={winner_metrics['ndcg_cut_10']:.4f}, "
            f"MAP={winner_metrics['map']:.4f}\n"
        )


if __name__ == "__main__":
    main()
