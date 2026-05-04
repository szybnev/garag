"""NFR benchmark for the GaRAG runtime MVP.

The default run targets the real HTTP API (`/query`) because that is the
user-facing runtime path. It measures warm end-to-end latency and concurrent
throughput without mutating the index. Full Qdrant indexing time is measured
only when `--run-indexing` is passed, because it recreates `garag_v1`.

Outputs:

- `evaluation/reports/nfr_report.md`
- `evaluation/results/nfr_benchmark.json`
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from collections.abc import Callable

GOLDEN_FILE = Path(__file__).resolve().parents[1] / "data" / "golden" / "golden_set_v1.jsonl"
REPORT_FILE = Path(__file__).resolve().parents[1] / "evaluation" / "reports" / "nfr_report.md"
RAW_FILE = Path(__file__).resolve().parents[1] / "evaluation" / "results" / "nfr_benchmark.json"

LATENCY_P95_TARGET_MS = 8000.0
THROUGHPUT_TARGET_RPS = 0.5
INDEXING_TARGET_SECONDS = 20 * 60.0


@dataclass(frozen=True)
class BenchmarkQuery:
    qid: str
    question: str


@dataclass
class QueryMeasurement:
    qid: str
    ok: bool
    status_code: int | None
    latency_ms: float
    error: str | None = None
    response_latency_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class PhaseSummary:
    count: int
    successes: int
    failures: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    rps: float | None = None


@dataclass
class IndexingMeasurement:
    ran: bool
    ok: bool | None = None
    elapsed_s: float | None = None
    command: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class ReportInput:
    api_url: str
    golden: Path
    top_k: int
    health: dict[str, Any]
    latency_summary: PhaseSummary
    throughput_summary: PhaseSummary
    indexing: IndexingMeasurement
    latency_rows: list[QueryMeasurement]
    throughput_rows: list[QueryMeasurement]


def _load_queries(path: Path, limit: int) -> list[BenchmarkQuery]:
    queries: list[BenchmarkQuery] = []
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            item = json.loads(line)
            queries.append(BenchmarkQuery(qid=str(item["qid"]), question=str(item["question"])))
            if len(queries) >= limit:
                break
    return queries


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(len(ordered) * pct), len(ordered) - 1)
    return ordered[idx]


def _summarise(
    measurements: list[QueryMeasurement],
    *,
    elapsed_s: float | None = None,
) -> PhaseSummary:
    latencies = [m.latency_ms for m in measurements if m.ok]
    successes = sum(1 for m in measurements if m.ok)
    failures = len(measurements) - successes
    rps = None
    if elapsed_s is not None and elapsed_s > 0.0:
        rps = round(successes / elapsed_s, 3)
    return PhaseSummary(
        count=len(measurements),
        successes=successes,
        failures=failures,
        mean_ms=round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        p50_ms=round(_percentile(latencies, 0.50), 1),
        p95_ms=round(_percentile(latencies, 0.95), 1),
        min_ms=round(min(latencies), 1) if latencies else 0.0,
        max_ms=round(max(latencies), 1) if latencies else 0.0,
        rps=rps,
    )


def _target_label(value: float | None, target: float, *, lower_is_better: bool) -> str:
    if value is None:
        return "NOT RUN"
    passed = value <= target if lower_is_better else value >= target
    return "PASS" if passed else "FAIL"


def _query_once(
    client: httpx.Client,
    api_url: str,
    query: BenchmarkQuery,
    *,
    top_k: int,
) -> QueryMeasurement:
    started = time.perf_counter()
    try:
        response = client.post(
            f"{api_url}/query",
            json={"query": query.question, "top_k": top_k},
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not response.is_success:
            return QueryMeasurement(
                qid=query.qid,
                ok=False,
                status_code=response.status_code,
                latency_ms=round(elapsed_ms, 1),
                error=response.text[:500],
            )
        body = response.json()
        latency_breakdown = body.get("latency_ms") or {}
        if not isinstance(latency_breakdown, dict):
            latency_breakdown = {}
        return QueryMeasurement(
            qid=query.qid,
            ok=True,
            status_code=response.status_code,
            latency_ms=round(elapsed_ms, 1),
            response_latency_ms={str(k): float(v) for k, v in latency_breakdown.items()},
        )
    except (httpx.HTTPError, ValueError) as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return QueryMeasurement(
            qid=query.qid,
            ok=False,
            status_code=None,
            latency_ms=round(elapsed_ms, 1),
            error=str(exc),
        )


def _health_gate(client: httpx.Client, api_url: str) -> dict[str, Any]:
    response = client.get(f"{api_url}/health")
    response.raise_for_status()
    body = response.json()
    if body.get("status") != "ok" or body.get("pipeline_loaded") is not True:
        msg = f"runtime is not ready: {body!r}"
        raise RuntimeError(msg)
    return body


def _run_warmup(
    client: httpx.Client,
    api_url: str,
    queries: list[BenchmarkQuery],
    *,
    warmup: int,
    top_k: int,
) -> None:
    for query in queries[:warmup]:
        measurement = _query_once(client, api_url, query, top_k=top_k)
        status = "ok" if measurement.ok else f"failed: {measurement.error}"
        print(f"[warmup] {query.qid} {status} {measurement.latency_ms:.1f} ms")


def _run_latency_phase(
    client: httpx.Client,
    api_url: str,
    queries: list[BenchmarkQuery],
    *,
    top_k: int,
) -> list[QueryMeasurement]:
    measurements: list[QueryMeasurement] = []
    for query in queries:
        measurement = _query_once(client, api_url, query, top_k=top_k)
        measurements.append(measurement)
        status = "OK" if measurement.ok else "FAIL"
        print(f"[latency] {status} {query.qid} {measurement.latency_ms:.1f} ms")
    return measurements


def _run_throughput_phase(
    api_url: str,
    queries: list[BenchmarkQuery],
    *,
    concurrency: int,
    top_k: int,
    timeout: float,
    client_factory: Callable[[], httpx.Client] | None = None,
) -> tuple[list[QueryMeasurement], float]:
    started = time.perf_counter()

    def run_one(query: BenchmarkQuery) -> QueryMeasurement:
        client = httpx.Client(timeout=timeout) if client_factory is None else client_factory()
        with client:
            return _query_once(client, api_url, query, top_k=top_k)

    measurements: list[QueryMeasurement] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(run_one, query) for query in queries]
        for future in concurrent.futures.as_completed(futures):
            measurement = future.result()
            measurements.append(measurement)
            status = "OK" if measurement.ok else "FAIL"
            print(f"[throughput] {status} {measurement.qid} {measurement.latency_ms:.1f} ms")
    elapsed = time.perf_counter() - started
    measurements.sort(key=lambda m: m.qid)
    return measurements, elapsed


def _run_indexing_phase(command: list[str]) -> IndexingMeasurement:
    started = time.perf_counter()
    try:
        completed = subprocess.run(  # noqa: S603
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return IndexingMeasurement(
            ran=True,
            ok=False,
            elapsed_s=round(time.perf_counter() - started, 1),
            command=command,
            error=str(exc),
        )
    elapsed = round(time.perf_counter() - started, 1)
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
    return IndexingMeasurement(
        ran=True,
        ok=completed.returncode == 0,
        elapsed_s=elapsed,
        command=command,
        error=None if completed.returncode == 0 else output[-2000:],
    )


def _render_report(data: ReportInput) -> str:
    indexing_value = data.indexing.elapsed_s if data.indexing.ran else None
    indexing_status = _target_label(
        indexing_value,
        INDEXING_TARGET_SECONDS,
        lower_is_better=True,
    )
    latency_status = _target_label(
        data.latency_summary.p95_ms,
        LATENCY_P95_TARGET_MS,
        lower_is_better=True,
    )
    throughput_status = _target_label(
        data.throughput_summary.rps,
        THROUGHPUT_TARGET_RPS,
        lower_is_better=False,
    )
    lines = [
        "# NFR benchmark — Runtime MVP",
        "",
        f"API URL: `{data.api_url}`",
        f"Golden set: `{data.golden}`",
        f"Top K: `{data.top_k}`",
        f"Health: `{json.dumps(data.health, ensure_ascii=False)}`",
        "",
        "## Summary",
        "",
        "| Metric | Value | Target | Status |",
        "|---|---:|---:|---|",
        (
            f"| p95 e2e latency (warm) | {data.latency_summary.p95_ms:.0f} ms "
            f"| ≤ {LATENCY_P95_TARGET_MS:.0f} ms "
            f"| {latency_status} |"
        ),
        (
            f"| Throughput | {(data.throughput_summary.rps or 0.0):.3f} RPS "
            f"| ≥ {THROUGHPUT_TARGET_RPS:.1f} RPS "
            f"| {throughput_status} |"
        ),
        (
            f"| Indexing time | {_format_optional_seconds(indexing_value)} "
            f"| ≤ {INDEXING_TARGET_SECONDS:.0f} s | {indexing_status} |"
        ),
        "",
        "## Throughput note",
        "",
        (
            "The v0.1.0 throughput target is scoped to a single local FastAPI "
            "instance backed by a 35B LM Studio generator. Stage timings show "
            f"generation dominates runtime: mean `gen` latency is "
            f"{_mean_stage_ms(data.throughput_rows, 'gen'):.0f} ms in the "
            f"throughput phase at `top_k={data.top_k}`, while retrieval plus "
            "rerank stages are in the tens of milliseconds. The aspirational "
            "≥2 RPS target is deferred to a serving-focused increment such as "
            "vLLM continuous batching or a smaller generator."
        ),
        "",
        "## Latency Phase",
        "",
        _phase_table(data.latency_summary),
        "",
        "## Throughput Phase",
        "",
        _phase_table(data.throughput_summary),
        "",
    ]
    if data.indexing.ran:
        lines.extend(
            [
                "## Indexing Phase",
                "",
                f"- Command: `{' '.join(data.indexing.command)}`",
                f"- Elapsed: {_format_optional_seconds(data.indexing.elapsed_s)}",
                f"- Status: {'PASS' if data.indexing.ok else 'FAIL'}",
                "",
            ]
        )
        if data.indexing.error:
            lines.extend(["```text", data.indexing.error, "```", ""])
    else:
        lines.extend(
            [
                "## Indexing Phase",
                "",
                "Not run. Pass `--run-indexing` to measure full Qdrant rebuild time.",
                "",
            ]
        )
    lines.extend(
        [
            "## Per-query Latency",
            "",
            _measurements_table(data.latency_rows),
            "",
            "## Per-query Throughput",
            "",
            _measurements_table(data.throughput_rows),
            "",
        ]
    )
    return "\n".join(lines)


def _mean_stage_ms(rows: list[QueryMeasurement], key: str) -> float:
    values = [
        row.response_latency_ms[key] for row in rows if row.ok and key in row.response_latency_ms
    ]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _format_optional_seconds(value: float | None) -> str:
    if value is None:
        return "not run"
    return f"{value:.1f} s"


def _phase_table(summary: PhaseSummary) -> str:
    return "\n".join(
        [
            "| count | success | failures | mean ms | p50 ms | p95 ms | min ms | max ms | rps |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            (
                f"| {summary.count} | {summary.successes} | {summary.failures} "
                f"| {summary.mean_ms:.1f} | {summary.p50_ms:.1f} | {summary.p95_ms:.1f} "
                f"| {summary.min_ms:.1f} | {summary.max_ms:.1f} "
                f"| {(summary.rps or 0.0):.3f} |"
            ),
        ]
    )


def _measurements_table(rows: list[QueryMeasurement]) -> str:
    lines = [
        "| qid | ok | status | latency ms | stage total ms | error |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        stage_total = row.response_latency_ms.get("total", 0.0)
        error = (row.error or "").replace("\n", " ")[:120]
        lines.append(
            f"| `{row.qid}` | {'yes' if row.ok else 'no'} | {row.status_code or 0} "
            f"| {row.latency_ms:.1f} | {stage_total:.1f} | {error} |"
        )
    return "\n".join(lines)


def _write_outputs(
    *,
    report_file: Path,
    raw_file: Path,
    report: str,
    raw: dict[str, Any],
) -> None:
    report_file.parent.mkdir(parents=True, exist_ok=True)
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(report)
    raw_file.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--api-url", default="http://localhost:8000")
    ap.add_argument("--golden", type=Path, default=GOLDEN_FILE)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--report", type=Path, default=REPORT_FILE)
    ap.add_argument("--raw", type=Path, default=RAW_FILE)
    ap.add_argument(
        "--run-indexing",
        action="store_true",
        help="also run `uv run python -m scripts.build_qdrant` and measure full rebuild time",
    )
    ap.add_argument(
        "--fail-on-target-miss",
        action="store_true",
        help="return exit code 1 when any measured NFR target fails",
    )
    args = ap.parse_args()

    api_url = args.api_url.rstrip("/")
    queries = _load_queries(args.golden, args.limit)
    if not queries:
        msg = f"no queries loaded from {args.golden}"
        raise RuntimeError(msg)

    timeout = httpx.Timeout(args.timeout, connect=10.0)
    with httpx.Client(timeout=timeout) as client:
        health = _health_gate(client, api_url)
        print(f"[health] {health}")
        _run_warmup(client, api_url, queries, warmup=args.warmup, top_k=args.top_k)
        latency_rows = _run_latency_phase(client, api_url, queries, top_k=args.top_k)

    throughput_rows, throughput_elapsed_s = _run_throughput_phase(
        api_url,
        queries,
        concurrency=args.concurrency,
        top_k=args.top_k,
        timeout=args.timeout,
    )
    indexing = (
        _run_indexing_phase(["uv", "run", "python", "-m", "scripts.build_qdrant"])
        if args.run_indexing
        else IndexingMeasurement(ran=False)
    )

    latency_summary = _summarise(latency_rows)
    throughput_summary = _summarise(throughput_rows, elapsed_s=throughput_elapsed_s)
    report = _render_report(
        ReportInput(
            api_url=api_url,
            golden=args.golden,
            top_k=args.top_k,
            health=health,
            latency_summary=latency_summary,
            throughput_summary=throughput_summary,
            indexing=indexing,
            latency_rows=latency_rows,
            throughput_rows=throughput_rows,
        )
    )
    raw = {
        "api_url": api_url,
        "golden": str(args.golden),
        "limit": args.limit,
        "warmup": args.warmup,
        "concurrency": args.concurrency,
        "top_k": args.top_k,
        "health": health,
        "targets": {
            "latency_p95_ms": LATENCY_P95_TARGET_MS,
            "throughput_rps": THROUGHPUT_TARGET_RPS,
            "indexing_seconds": INDEXING_TARGET_SECONDS,
        },
        "latency_summary": asdict(latency_summary),
        "throughput_summary": asdict(throughput_summary),
        "indexing": asdict(indexing),
        "latency_rows": [asdict(row) for row in latency_rows],
        "throughput_rows": [asdict(row) for row in throughput_rows],
    }
    _write_outputs(report_file=args.report, raw_file=args.raw, report=report, raw=raw)
    print(f"[done] wrote {args.report}")
    print(f"[done] wrote {args.raw}")

    failed = (
        latency_summary.p95_ms > LATENCY_P95_TARGET_MS
        or (throughput_summary.rps or 0.0) < THROUGHPUT_TARGET_RPS
        or (
            indexing.ran
            and (not indexing.ok or (indexing.elapsed_s or 0.0) > INDEXING_TARGET_SECONDS)
        )
    )
    return 1 if args.fail_on_target_miss and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
