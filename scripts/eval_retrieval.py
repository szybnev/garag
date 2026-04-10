"""Retrieval evaluation against the golden set.

Runs each retriever (dense, sparse, hybrid via RRF, hybrid via alpha-weighted)
against `data/golden/golden_set_v1.jsonl` and reports `Recall@5/10`,
`nDCG@10`, and `MAP` via `pytrec_eval`.

Output: stdout summary table + `evaluation/reports/retrieval_report.md`.
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

METRICS = {"recall_5", "recall_10", "ndcg_cut_10", "map"}
METRIC_LABELS = {
    "recall_5": "Recall@5",
    "recall_10": "Recall@10",
    "ndcg_cut_10": "nDCG@10",
    "map": "MAP",
}


def _load_golden(path: Path) -> list[dict[str, Any]]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _qrels(golden: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {item["qid"]: dict.fromkeys(item["relevant_chunks"], 1) for item in golden}


def _run_to_dict(hits: list[Any]) -> dict[str, float]:
    """Convert a `list[ScoredChunk]` to the {chunk_id: score} dict pytrec_eval wants."""
    return {h.chunk_id: float(h.score) for h in hits}


def _summarise(
    name: str,
    qrels: dict[str, dict[str, int]],
    runs: dict[str, dict[str, float]],
    elapsed: float,
) -> dict[str, Any]:
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, METRICS)
    per_query = evaluator.evaluate(runs)
    averages: dict[str, float] = {}
    for metric in METRICS:
        scores = [per_query[qid].get(metric, 0.0) for qid in per_query]
        averages[metric] = sum(scores) / max(len(scores), 1)
    return {
        "name": name,
        "n_queries": len(per_query),
        "elapsed_s": elapsed,
        "metrics": averages,
        "per_query": per_query,
    }


def _format_table(rows: list[dict[str, Any]]) -> str:
    header = "| Method | " + " | ".join(METRIC_LABELS[m] for m in METRICS) + " | Latency (s) |"
    sep = "|" + "---|" * (len(METRICS) + 2)
    lines = [header, sep]
    for row in rows:
        cells = [
            row["name"],
            *(f"{row['metrics'][metric]:.4f}" for metric in METRICS),
            f"{row['elapsed_s']:.1f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _run_all_retrievers(
    golden: list[dict[str, Any]],
    dense: DenseRetriever,
    sparse: SparseRetriever,
    candidate_k: int,
    alpha: float,
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    dict[str, float],
]:
    dense_runs: dict[str, dict[str, float]] = {}
    sparse_runs: dict[str, dict[str, float]] = {}
    rrf_runs: dict[str, dict[str, float]] = {}
    alpha_runs: dict[str, dict[str, float]] = {}
    times = {"dense": 0.0, "sparse": 0.0, "rrf": 0.0, "alpha": 0.0}

    for item in golden:
        q = item["question"]
        qid = item["qid"]

        t0 = time.perf_counter()
        d_hits = dense.search(q, top_k=candidate_k)
        times["dense"] += time.perf_counter() - t0
        dense_runs[qid] = _run_to_dict(d_hits)

        t0 = time.perf_counter()
        s_hits = sparse.search(q, top_k=candidate_k)
        times["sparse"] += time.perf_counter() - t0
        sparse_runs[qid] = _run_to_dict(s_hits)

        t0 = time.perf_counter()
        rrf_hits = reciprocal_rank_fusion(d_hits, s_hits, top_k=candidate_k)
        times["rrf"] += time.perf_counter() - t0
        rrf_runs[qid] = _run_to_dict(rrf_hits)

        t0 = time.perf_counter()
        alpha_hits = alpha_weighted_fusion(d_hits, s_hits, alpha=alpha, top_k=candidate_k)
        times["alpha"] += time.perf_counter() - t0
        alpha_runs[qid] = _run_to_dict(alpha_hits)

    return dense_runs, sparse_runs, rrf_runs, alpha_runs, times


def _write_per_category(
    fh: Any,
    golden: list[dict[str, Any]],
    qrels: dict[str, dict[str, int]],
    runs: dict[str, dict[str, dict[str, float]]],
    alpha: float,
) -> None:
    fh.write("\n\n## Per-category breakdown\n\n")
    for category in ("factual", "tool_usage", "multi_hop"):
        cat_qids = {item["qid"] for item in golden if item["category"] == category}
        if not cat_qids:
            continue
        cat_qrels = {q: qrels[q] for q in cat_qids}
        cat_summaries = [
            _summarise("dense", cat_qrels, {q: runs["dense"][q] for q in cat_qids}, 0.0),
            _summarise("sparse", cat_qrels, {q: runs["sparse"][q] for q in cat_qids}, 0.0),
            _summarise("hybrid RRF", cat_qrels, {q: runs["rrf"][q] for q in cat_qids}, 0.0),
            _summarise(
                f"hybrid alpha={alpha}",
                cat_qrels,
                {q: runs["alpha"][q] for q in cat_qids},
                0.0,
            ),
        ]
        fh.write(f"### {category} ({len(cat_qids)} queries)\n\n")
        fh.write(_format_table(cat_summaries))
        fh.write("\n\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--golden", type=Path, default=GOLDEN_FILE)
    ap.add_argument("--candidate-k", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--report", type=Path, default=REPORT_FILE)
    args = ap.parse_args()

    golden = _load_golden(args.golden)
    print(f"[load] {len(golden)} queries from {args.golden}")
    qrels = _qrels(golden)

    print("[init] dense + sparse retrievers")
    dense = DenseRetriever()
    sparse = SparseRetriever()

    dense_runs, sparse_runs, rrf_runs, alpha_runs, times = _run_all_retrievers(
        golden, dense, sparse, args.candidate_k, args.alpha
    )

    summaries = [
        _summarise("dense (bge-m3)", qrels, dense_runs, times["dense"]),
        _summarise("sparse (BM25)", qrels, sparse_runs, times["sparse"]),
        _summarise(
            "hybrid RRF (k=60)", qrels, rrf_runs, times["dense"] + times["sparse"] + times["rrf"]
        ),
        _summarise(
            f"hybrid alpha={args.alpha}",
            qrels,
            alpha_runs,
            times["dense"] + times["sparse"] + times["alpha"],
        ),
    ]

    table = _format_table(summaries)
    print()
    print(table)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w") as fh:
        fh.write("# Retrieval evaluation — `golden_set_v1`\n\n")
        fh.write(
            f"Run on **{len(golden)} queries** with `candidate_k={args.candidate_k}`. "
            f"Hybrid alpha-weighted uses `alpha={args.alpha}`. "
            "Metrics computed via `pytrec_eval`.\n\n"
        )
        fh.write(table)
        _write_per_category(
            fh,
            golden,
            qrels,
            {"dense": dense_runs, "sparse": sparse_runs, "rrf": rrf_runs, "alpha": alpha_runs},
            args.alpha,
        )
    print(f"\n[done] wrote {args.report}")


if __name__ == "__main__":
    main()
