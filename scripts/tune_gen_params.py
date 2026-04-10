"""Grid search over qwen3.5 generation decode parameters.

Sweeps `(temperature x top_p x num_predict)` on a subset of
`golden_set_v1.jsonl`, runs the full retrieval → generation pipeline
for each config, and scores each row on a cheap heuristic built from
the observations the generator already exposes (no LLM judge — that
lives in `scripts/eval_generation.py` for `garag-zqc.19`).

Retrieval is cached once per query *before* the config loop, so the
only moving piece across configs is `Generator.generate()`. At 5s per
call on RTX 5090 x 20 queries x 36 configs that's ~60 min.

Output:
- `evaluation/results/gen_params_grid.json` — per-config metrics
- stdout top-10 by heuristic score
- winner is reported but NOT written to `app/config.py` — copy by hand
  after eyeballing the full table.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.rag.generator import GenerationError, Generator
from app.rag.pipeline import HybridRetriever
from app.rag.reranker import Reranker

if TYPE_CHECKING:
    from app.rag import ScoredChunk

GOLDEN_FILE = Path(__file__).resolve().parents[1] / "data" / "golden" / "golden_set_v1.jsonl"
OUTPUT_FILE = (
    Path(__file__).resolve().parents[1] / "evaluation" / "results" / "gen_params_grid.json"
)

DEFAULT_GRID_TEMPERATURE = (0.0, 0.1, 0.2, 0.4)
DEFAULT_GRID_TOP_P = (0.8, 0.9, 1.0)
DEFAULT_GRID_NUM_PREDICT = (400, 800, 1200)


def _load_golden(path: Path, limit: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            items.append(json.loads(line))
            if len(items) >= limit:
                break
    return items


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(len(ordered) * pct), len(ordered) - 1)
    return ordered[idx]


def _score_config(row: dict[str, Any]) -> float:
    """Heuristic rank: favour confident, well-cited, fast answers.

    Pure stand-in for a real judge — cheap to compute, coarse but
    monotonic in the right direction for small grids. `.19` replaces
    this with a proper LLM-judge loop on the winning config.
    """
    return (
        row["mean_confidence"]
        + 0.1 * row["mean_citations"]
        - 0.02 * (row["mean_latency_gen_ms"] / 1000.0)
    )


def _run_config(
    *,
    cached_chunks: dict[str, list[ScoredChunk]],
    golden: list[dict[str, Any]],
    temperature: float,
    top_p: float,
    num_predict: int,
) -> dict[str, Any]:
    with Generator(
        temperature=temperature,
        top_p=top_p,
        num_predict=num_predict,
    ) as gen:
        parsed = 0
        grounded = 0
        failures = 0
        confidences: list[float] = []
        citations: list[int] = []
        latencies: list[float] = []
        for item in golden:
            chunks = cached_chunks[item["qid"]]
            retrieved_ids = {c.chunk_id for c in chunks}
            t0 = time.perf_counter()
            try:
                response = gen.generate(item["question"], chunks)
            except GenerationError:
                failures += 1
                continue
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)
            parsed += 1
            confidences.append(response.confidence)
            citations.append(len(response.citations))
            if all(cid in retrieved_ids for cid in response.used_chunks):
                grounded += 1

    n = len(golden)
    row: dict[str, Any] = {
        "config": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": num_predict,
        },
        "n_queries": n,
        "n_failures": failures,
        "format_rate": round(parsed / n, 4) if n else 0.0,
        "grounded_rate": round(grounded / parsed, 4) if parsed else 0.0,
        "mean_confidence": round(statistics.fmean(confidences), 4) if confidences else 0.0,
        "mean_citations": round(statistics.fmean(citations), 3) if citations else 0.0,
        "mean_latency_gen_ms": round(statistics.fmean(latencies), 1) if latencies else 0.0,
        "p95_latency_gen_ms": round(_percentile(latencies, 0.95), 1) if latencies else 0.0,
    }
    row["heuristic_score"] = round(_score_config(row), 4)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--golden", type=Path, default=GOLDEN_FILE)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--candidate-k", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--output", type=Path, default=OUTPUT_FILE)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument(
        "--temperature",
        type=float,
        nargs="+",
        default=list(DEFAULT_GRID_TEMPERATURE),
    )
    ap.add_argument("--top-p", type=float, nargs="+", default=list(DEFAULT_GRID_TOP_P))
    ap.add_argument(
        "--num-predict",
        type=int,
        nargs="+",
        default=list(DEFAULT_GRID_NUM_PREDICT),
    )
    args = ap.parse_args()

    golden = _load_golden(args.golden, args.limit)
    print(f"[load] {len(golden)} queries from {args.golden}")

    print("[init] hybrid retriever (dense + sparse + reranker)")
    reranker = None if args.no_rerank else Reranker()
    retriever = HybridRetriever(reranker=reranker)

    print("[cache] running retrieval once per query")
    cached_chunks: dict[str, list[ScoredChunk]] = {}
    t0 = time.perf_counter()
    for item in golden:
        cached_chunks[item["qid"]] = retriever.retrieve(
            item["question"], candidate_k=args.candidate_k, top_k=args.top_k
        )
    print(f"[cache] {time.perf_counter() - t0:.1f}s")

    combos = list(itertools.product(args.temperature, args.top_p, args.num_predict))
    print(f"[grid] {len(combos)} configs x {len(golden)} queries")

    rows: list[dict[str, Any]] = []
    for idx, (temperature, top_p, num_predict) in enumerate(combos, start=1):
        t_cfg = time.perf_counter()
        row = _run_config(
            cached_chunks=cached_chunks,
            golden=golden,
            temperature=temperature,
            top_p=top_p,
            num_predict=num_predict,
        )
        elapsed = time.perf_counter() - t_cfg
        print(
            f"[{idx:2d}/{len(combos)}] temp={temperature:.2f} top_p={top_p:.2f} "
            f"num_predict={num_predict:4d} | score={row['heuristic_score']:.4f} "
            f"fmt={row['format_rate']:.2f} gnd={row['grounded_rate']:.2f} "
            f"conf={row['mean_confidence']:.2f} cites={row['mean_citations']:.2f} "
            f"lat_p95={row['p95_latency_gen_ms']:.0f}ms ({elapsed:.0f}s)"
        )
        rows.append(row)

    rows.sort(
        key=lambda r: (
            r["heuristic_score"],
            r["grounded_rate"],
            -r["p95_latency_gen_ms"],
        ),
        reverse=True,
    )

    print("\n" + "=" * 78)
    print("Top 10 configs:")
    print("=" * 78)
    print(
        f"{'rank':>4} | {'temp':>5} {'top_p':>6} {'num_pred':>8} | "
        f"{'score':>6} {'fmt':>5} {'gnd':>5} {'conf':>5} {'cit':>5} {'p95ms':>7}"
    )
    for rank, row in enumerate(rows[:10], start=1):
        cfg = row["config"]
        print(
            f"{rank:>4} | {cfg['temperature']:>5.2f} {cfg['top_p']:>6.2f} "
            f"{cfg['num_predict']:>8d} | {row['heuristic_score']:>6.4f} "
            f"{row['format_rate']:>5.2f} {row['grounded_rate']:>5.2f} "
            f"{row['mean_confidence']:>5.2f} {row['mean_citations']:>5.2f} "
            f"{row['p95_latency_gen_ms']:>7.0f}"
        )

    best = rows[0]
    print("\n[winner]", best["config"], f"score={best['heuristic_score']:.4f}")
    print(
        f"[winner] format={best['format_rate']:.2f} grounded={best['grounded_rate']:.2f} "
        f"conf={best['mean_confidence']:.2f} cites={best['mean_citations']:.2f} "
        f"p95={best['p95_latency_gen_ms']:.0f}ms"
    )
    print("\nNote: copy the chosen config into app/config.py manually after review.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "golden_file": str(args.golden),
        "n_queries": len(golden),
        "candidate_k": args.candidate_k,
        "top_k": args.top_k,
        "rerank": not args.no_rerank,
        "grid": {
            "temperature": list(args.temperature),
            "top_p": list(args.top_p),
            "num_predict": list(args.num_predict),
        },
        "rows": rows,
    }
    with args.output.open("w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[done] wrote {args.output}")


if __name__ == "__main__":
    main()
