"""Full generation evaluation — LLM-as-judge + mechanical metrics.

Closes `garag-zqc.19`. Runs the full `QueryPipeline` over
`data/golden/golden_set_v1.jsonl`, then asks `Judge` (qwen3.5:35b)
to score each candidate answer on three dimensions:

- **faithfulness** — claims supported by context
- **correctness** — matches reference answer on substance
- **citation_support** — cited chunks actually back the quoted claims

Mechanical metrics collected in the same pass:

- **format_rate** — fraction of queries where `QueryResponse` parsed
- **citation_acc** — fraction where all `used_chunks ⊆ retrieved_ids`
- per-stage **latency** (mean, p95) via `QueryPipeline.query().latency_ms`

Writes:

- `evaluation/reports/generation_report.md` — rendered markdown (overall,
  per-category, 10 random manual samples, self-bias caveat)
- `evaluation/results/generation_eval.json` — raw per-query judge verdicts
  and pipeline responses for reproducibility

**Self-bias caveat.** Generator and judge share the same checkpoint
(`qwen3.5:35b`) because it is the largest Ollama-available model on the
d9 host. Literature reports 5-15% upward bias on faithfulness when
generator and judge agree on a checkpoint. Treat absolute numbers as
upper bounds; rely on per-category deltas and the 10-sample manual
review for qualitative signal. Cross-model rerun is deferred.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.rag.generator import GenerationError, Generator
from app.rag.judge import Judge, JudgeError, _JudgeVerdict
from app.rag.pipeline import HybridRetriever
from app.rag.query_pipeline import QueryPipeline
from app.rag.reranker import Reranker

if TYPE_CHECKING:
    from app.schemas import QueryResponse

GOLDEN_FILE = Path(__file__).resolve().parents[1] / "data" / "golden" / "golden_set_v1.jsonl"
REPORT_FILE = (
    Path(__file__).resolve().parents[1] / "evaluation" / "reports" / "generation_report.md"
)
RAW_FILE = Path(__file__).resolve().parents[1] / "evaluation" / "results" / "generation_eval.json"


@dataclass
class EvalRow:
    qid: str
    category: str
    question: str
    golden: str
    parsed: bool
    format_error: str | None = None
    judge_error: str | None = None
    answer: str = ""
    confidence: float = 0.0
    n_citations: int = 0
    grounded: bool = False
    latency_ms: dict[str, float] = field(default_factory=dict)
    faithfulness: int | None = None
    correctness: int | None = None
    citation_support: int | None = None
    rationale: str = ""


def _load_golden(path: Path) -> list[dict[str, Any]]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(len(ordered) * pct), len(ordered) - 1)
    return ordered[idx]


def _aggregate(rows: list[EvalRow]) -> dict[str, float]:
    parsed_rows = [r for r in rows if r.parsed]
    judged_rows = [r for r in parsed_rows if r.faithfulness is not None]
    n = len(rows)
    faith_values = [r.faithfulness for r in judged_rows if r.faithfulness is not None]
    corr_values = [r.correctness for r in judged_rows if r.correctness is not None]
    cite_sup_values = [r.citation_support for r in judged_rows if r.citation_support is not None]
    total_latencies = [r.latency_ms.get("total", 0.0) for r in parsed_rows]
    return {
        "n_queries": float(n),
        "format_rate": round(len(parsed_rows) / n, 4) if n else 0.0,
        "citation_acc": (
            round(sum(r.grounded for r in parsed_rows) / len(parsed_rows), 4)
            if parsed_rows
            else 0.0
        ),
        "faithfulness_mean": (
            round(statistics.fmean(faith_values) / 2.0, 4) if faith_values else 0.0
        ),
        "correctness_mean": (round(statistics.fmean(corr_values) / 2.0, 4) if corr_values else 0.0),
        "citation_support_mean": (
            round(statistics.fmean(cite_sup_values) / 2.0, 4) if cite_sup_values else 0.0
        ),
        "judge_format_failures": float(sum(1 for r in parsed_rows if r.judge_error is not None)),
        "mean_latency_ms": (
            round(statistics.fmean(total_latencies), 1) if total_latencies else 0.0
        ),
        "p95_latency_ms": round(_percentile(total_latencies, 0.95), 1),
    }


def _render_report(
    rows: list[EvalRow],  # noqa: ARG001
    overall: dict[str, float],
    per_category: dict[str, dict[str, float]],
    samples: list[EvalRow],
    *,
    model: str,
    judge_model: str,
) -> str:
    lines: list[str] = []
    lines.append("# Generation evaluation — `golden_set_v1`\n")
    lines.append(
        f"Ran the full `QueryPipeline` over {int(overall['n_queries'])} queries from "
        f"`data/golden/golden_set_v1.jsonl`. Generator: `{model}`. Judge: `{judge_model}`."
    )
    lines.append("")
    lines.append(
        "> **Self-bias caveat.** This report uses `qwen3.5:35b` as both generator "
        "and judge, which is known to inflate faithfulness by 5-15% vs a cross-model "
        "judge. Treat the absolute numbers as upper bounds; rely on per-category "
        "deltas and the manual 10-sample review for qualitative signal. "
        "A cross-model rerun (e.g. GPT-4o or Claude as judge) is deferred."
    )
    lines.append("")
    lines.append("## Overall metrics\n")
    lines.append("| metric | value | NFR target |")
    lines.append("|---|---|---|")
    lines.append(f"| format_rate | **{overall['format_rate']:.3f}** | — |")
    lines.append(f"| citation_acc (mechanical) | **{overall['citation_acc']:.3f}** | ≥ 0.85 |")
    lines.append(
        f"| faithfulness (judge, /2 norm) | **{overall['faithfulness_mean']:.3f}** | ≥ 0.80 |"
    )
    lines.append(
        f"| correctness (judge, /2 norm) | **{overall['correctness_mean']:.3f}** | ≥ 0.70 |"
    )
    lines.append(
        f"| citation_support (judge, /2 norm) | **{overall['citation_support_mean']:.3f}** | — |"
    )
    lines.append(f"| judge format failures | {int(overall['judge_format_failures'])} | — |")
    lines.append(
        f"| mean / p95 latency | "
        f"{overall['mean_latency_ms']:.0f} ms / {overall['p95_latency_ms']:.0f} ms | "
        "p95 ≤ 8000 ms |"
    )
    lines.append("")
    lines.append("## Per-category breakdown\n")
    lines.append("| category | n | format | cit_acc | faith | corr | cit_sup | p95 ms |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for cat in ("factual", "tool_usage", "multi_hop"):
        if cat not in per_category:
            continue
        m = per_category[cat]
        lines.append(
            f"| {cat} | {int(m['n_queries'])} "
            f"| {m['format_rate']:.3f} "
            f"| {m['citation_acc']:.3f} "
            f"| {m['faithfulness_mean']:.3f} "
            f"| {m['correctness_mean']:.3f} "
            f"| {m['citation_support_mean']:.3f} "
            f"| {m['p95_latency_ms']:.0f} |"
        )
    lines.append("")
    lines.append("## Manual review — 10 random samples\n")
    for i, row in enumerate(samples, start=1):
        lines.append(f"### Sample {i} — `{row.qid}` ({row.category})")
        lines.append("")
        lines.append(f"**Question:** {row.question}")
        lines.append("")
        lines.append(f"**Golden:** {row.golden}")
        lines.append("")
        lines.append(f"**Candidate:** {row.answer}")
        lines.append("")
        if row.faithfulness is not None:
            lines.append(
                f"**Judge:** faith={row.faithfulness}/2 "
                f"corr={row.correctness}/2 "
                f"cit_sup={row.citation_support}/2"
            )
            lines.append("")
            lines.append(f"*Rationale:* {row.rationale}")
        else:
            lines.append(f"**Judge:** error — {row.judge_error}")
        lines.append("")
    return "\n".join(lines)


def _run_pipeline_phase(
    *,
    golden: list[dict[str, Any]],
    pipeline: QueryPipeline,
) -> list[tuple[dict[str, Any], QueryResponse | None, str | None]]:
    results: list[tuple[dict[str, Any], QueryResponse | None, str | None]] = []
    for item in golden:
        try:
            response = pipeline.query(item["question"])
        except GenerationError as exc:
            results.append((item, None, str(exc)))
            continue
        results.append((item, response, None))
    return results


def _run_judge_phase(
    *,
    responses: list[tuple[dict[str, Any], QueryResponse | None, str | None]],
    judge: Judge,
    retriever: HybridRetriever,
    candidate_k: int,
    top_k: int,
) -> list[EvalRow]:
    rows: list[EvalRow] = []
    for item, response, format_error in responses:
        retrieved = retriever.retrieve(item["question"], candidate_k=candidate_k, top_k=top_k)
        retrieved_ids = {c.chunk_id for c in retrieved}
        row = EvalRow(
            qid=item["qid"],
            category=item["category"],
            question=item["question"],
            golden=item["answer"],
            parsed=response is not None,
            format_error=format_error,
        )
        if response is None:
            rows.append(row)
            continue
        row.answer = response.answer
        row.confidence = response.confidence
        row.n_citations = len(response.citations)
        row.grounded = all(cid in retrieved_ids for cid in response.used_chunks)
        row.latency_ms = dict(response.latency_ms or {})
        try:
            verdict: _JudgeVerdict = judge.judge(
                question=item["question"],
                golden=item["answer"],
                candidate=response,
                chunks=retrieved,
            )
            row.faithfulness = verdict.faithfulness
            row.correctness = verdict.correctness
            row.citation_support = verdict.citation_support
            row.rationale = verdict.rationale
        except JudgeError as exc:
            row.judge_error = str(exc)
        rows.append(row)
    return rows


def main() -> int:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--golden", type=Path, default=GOLDEN_FILE)
    ap.add_argument("--report", type=Path, default=REPORT_FILE)
    ap.add_argument("--raw", type=Path, default=RAW_FILE)
    ap.add_argument("--candidate-k", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="0 = full golden set")
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args()

    golden = _load_golden(args.golden)
    if args.limit:
        golden = golden[: args.limit]
    print(f"[load] {len(golden)} queries from {args.golden}")

    print("[init] hybrid retriever + generator + judge")
    reranker = None if args.no_rerank else Reranker()
    retriever = HybridRetriever(reranker=reranker)
    pipeline = QueryPipeline(
        retriever=retriever,
        generator=Generator(),
        candidate_k=args.candidate_k,
        top_k=args.top_k,
    )

    print("[run] pipeline phase (retrieve + generate for every query)")
    t0 = time.perf_counter()
    responses = _run_pipeline_phase(golden=golden, pipeline=pipeline)
    print(f"[run] pipeline done in {time.perf_counter() - t0:.1f}s")

    print("[run] judge phase (qwen3.5:35b per parsed response)")
    t0 = time.perf_counter()
    with Judge() as judge:
        rows = _run_judge_phase(
            responses=responses,
            judge=judge,
            retriever=retriever,
            candidate_k=args.candidate_k,
            top_k=args.top_k,
        )
    print(f"[run] judge done in {time.perf_counter() - t0:.1f}s")

    overall = _aggregate(rows)
    per_category: dict[str, dict[str, float]] = {}
    for category in ("factual", "tool_usage", "multi_hop"):
        cat_rows = [r for r in rows if r.category == category]
        if cat_rows:
            per_category[category] = _aggregate(cat_rows)

    rng = random.Random(args.seed)  # noqa: S311 — sampling for report, not crypto
    sample_pool = [r for r in rows if r.parsed]
    samples = rng.sample(sample_pool, min(args.n_samples, len(sample_pool)))

    from app.config import settings  # noqa: PLC0415

    report_md = _render_report(
        rows,
        overall,
        per_category,
        samples,
        model=settings.ollama_model,
        judge_model=settings.ollama_judge_model,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_md)
    print(f"[done] wrote {args.report}")

    args.raw.parent.mkdir(parents=True, exist_ok=True)
    with args.raw.open("w") as fh:
        json.dump(
            {
                "model": settings.ollama_model,
                "judge_model": settings.ollama_judge_model,
                "overall": overall,
                "per_category": per_category,
                "rows": [asdict(r) for r in rows],
            },
            fh,
            indent=2,
        )
    print(f"[done] wrote {args.raw}")

    print()
    print("=" * 60)
    print(f"format_rate        : {overall['format_rate']:.3f}")
    print(f"citation_acc       : {overall['citation_acc']:.3f}   (NFR ≥ 0.85)")
    print(f"faithfulness /2    : {overall['faithfulness_mean']:.3f}   (NFR ≥ 0.80)")
    print(f"correctness /2     : {overall['correctness_mean']:.3f}   (NFR ≥ 0.70)")
    print(f"citation_support/2 : {overall['citation_support_mean']:.3f}")
    mean_lat = overall["mean_latency_ms"]
    p95_lat = overall["p95_latency_ms"]
    print(f"mean / p95 latency : {mean_lat:.0f} / {p95_lat:.0f} ms")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
