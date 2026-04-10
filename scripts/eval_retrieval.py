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
from app.rag.reranker import Reranker
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
    reranker: Reranker | None,
    *,
    candidate_k: int,
    alpha: float,
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, float]]:
    runs: dict[str, dict[str, dict[str, float]]] = {
        "dense": {},
        "sparse": {},
        "rrf": {},
        "alpha": {},
    }
    if reranker is not None:
        runs["rerank"] = {}
    times = {"dense": 0.0, "sparse": 0.0, "rrf": 0.0, "alpha": 0.0, "rerank": 0.0}

    for item in golden:
        q = item["question"]
        qid = item["qid"]

        t0 = time.perf_counter()
        d_hits = dense.search(q, top_k=candidate_k)
        times["dense"] += time.perf_counter() - t0
        runs["dense"][qid] = _run_to_dict(d_hits)

        t0 = time.perf_counter()
        s_hits = sparse.search(q, top_k=candidate_k)
        times["sparse"] += time.perf_counter() - t0
        runs["sparse"][qid] = _run_to_dict(s_hits)

        t0 = time.perf_counter()
        rrf_hits = reciprocal_rank_fusion(d_hits, s_hits, top_k=candidate_k)
        times["rrf"] += time.perf_counter() - t0
        runs["rrf"][qid] = _run_to_dict(rrf_hits)

        t0 = time.perf_counter()
        alpha_hits = alpha_weighted_fusion(d_hits, s_hits, alpha=alpha, top_k=candidate_k)
        times["alpha"] += time.perf_counter() - t0
        runs["alpha"][qid] = _run_to_dict(alpha_hits)

        if reranker is not None:
            t0 = time.perf_counter()
            reranked = reranker.rerank(q, alpha_hits, top_k=candidate_k)
            times["rerank"] += time.perf_counter() - t0
            runs["rerank"][qid] = _run_to_dict(reranked)

    return runs, times


_RUN_LABELS = {
    "dense": "dense",
    "sparse": "sparse",
    "rrf": "hybrid RRF",
    "alpha": "hybrid alpha",
    "rerank": "hybrid + reranker",
}


def _write_per_category(
    fh: Any,
    golden: list[dict[str, Any]],
    qrels: dict[str, dict[str, int]],
    runs: dict[str, dict[str, dict[str, float]]],
) -> None:
    fh.write("\n\n## Per-category breakdown\n\n")
    for category in ("factual", "tool_usage", "multi_hop"):
        cat_qids = {item["qid"] for item in golden if item["category"] == category}
        if not cat_qids:
            continue
        cat_qrels = {q: qrels[q] for q in cat_qids}
        cat_summaries = [
            _summarise(_RUN_LABELS[name], cat_qrels, {q: runs[name][q] for q in cat_qids}, 0.0)
            for name in ("dense", "sparse", "rrf", "alpha", "rerank")
            if name in runs
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
    ap.add_argument(
        "--rerank", action="store_true", help="add bge-reranker-v2-m3 cross-encoder pass"
    )
    args = ap.parse_args()

    golden = _load_golden(args.golden)
    print(f"[load] {len(golden)} queries from {args.golden}")
    qrels = _qrels(golden)

    print("[init] dense + sparse retrievers")
    dense = DenseRetriever()
    sparse = SparseRetriever()
    reranker: Reranker | None = None
    if args.rerank:
        print("[init] cross-encoder reranker (bge-reranker-v2-m3)")
        reranker = Reranker()

    runs, times = _run_all_retrievers(
        golden,
        dense,
        sparse,
        reranker,
        candidate_k=args.candidate_k,
        alpha=args.alpha,
    )

    summaries = [
        _summarise("dense (bge-m3)", qrels, runs["dense"], times["dense"]),
        _summarise("sparse (BM25)", qrels, runs["sparse"], times["sparse"]),
        _summarise(
            "hybrid RRF (k=60)",
            qrels,
            runs["rrf"],
            times["dense"] + times["sparse"] + times["rrf"],
        ),
        _summarise(
            f"hybrid alpha={args.alpha}",
            qrels,
            runs["alpha"],
            times["dense"] + times["sparse"] + times["alpha"],
        ),
    ]
    if "rerank" in runs:
        summaries.append(
            _summarise(
                "hybrid + reranker",
                qrels,
                runs["rerank"],
                times["dense"] + times["sparse"] + times["alpha"] + times["rerank"],
            )
        )

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
        if reranker is not None:
            fh.write(
                "**Reranker:** `BAAI/bge-reranker-v2-m3` cross-encoder applied "
                f"on top of `hybrid alpha={args.alpha}` candidates.\n\n"
            )
        fh.write(table)
        _write_per_category(fh, golden, qrels, runs)
    print(f"\n[done] wrote {args.report}")


if __name__ == "__main__":
    main()
