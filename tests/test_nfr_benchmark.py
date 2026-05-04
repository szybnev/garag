"""Unit tests for the NFR benchmark script."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from scripts.nfr_benchmark import (
    BenchmarkQuery,
    IndexingMeasurement,
    QueryMeasurement,
    ReportInput,
    _health_gate,
    _percentile,
    _render_report,
    _run_latency_phase,
    _run_throughput_phase,
    _summarise,
    _target_label,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_percentile_uses_sorted_floor_index() -> None:
    assert _percentile([100.0, 10.0, 50.0, 80.0], 0.50) == 80.0
    assert _percentile([100.0, 10.0, 50.0, 80.0], 0.95) == 100.0
    assert _percentile([], 0.95) == 0.0


def test_target_label_handles_direction_and_not_run() -> None:
    assert _target_label(7000.0, 8000.0, lower_is_better=True) == "PASS"
    assert _target_label(9000.0, 8000.0, lower_is_better=True) == "FAIL"
    assert _target_label(2.2, 2.0, lower_is_better=False) == "PASS"
    assert _target_label(1.9, 2.0, lower_is_better=False) == "FAIL"
    assert _target_label(None, 2.0, lower_is_better=False) == "NOT RUN"


def test_health_gate_requires_loaded_pipeline() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://api.test/health"
        return httpx.Response(200, json={"status": "ok", "pipeline_loaded": True})

    client = httpx.Client(transport=httpx.MockTransport(handler))

    assert _health_gate(client, "http://api.test") == {
        "status": "ok",
        "pipeline_loaded": True,
    }


def test_health_gate_rejects_unloaded_pipeline() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"status": "ok", "pipeline_loaded": False})
        )
    )

    with pytest.raises(RuntimeError, match="runtime is not ready"):
        _health_gate(client, "http://api.test")


def test_latency_phase_posts_queries_and_extracts_stage_latency() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "answer": "ok",
                "citations": [],
                "confidence": 0.9,
                "used_chunks": [],
                "latency_ms": {"total": 123.4, "gen": 50.0},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    queries = [BenchmarkQuery(qid="q1", question="Question one?")]

    rows = _run_latency_phase(client, "http://api.test", queries, top_k=3)

    assert seen == [{"query": "Question one?", "top_k": 3}]
    assert len(rows) == 1
    assert rows[0].ok is True
    assert rows[0].status_code == 200
    assert rows[0].response_latency_ms == {"total": 123.4, "gen": 50.0}


def test_throughput_phase_counts_successes_and_failures() -> None:
    calls = {"count": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 2:
            return httpx.Response(500, text="boom")
        return httpx.Response(
            200,
            json={
                "answer": "ok",
                "citations": [],
                "confidence": 0.9,
                "used_chunks": [],
                "latency_ms": {"total": 10.0},
            },
        )

    def client_factory() -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(handler))

    rows, elapsed = _run_throughput_phase(
        "http://api.test",
        [
            BenchmarkQuery(qid="q1", question="one"),
            BenchmarkQuery(qid="q2", question="two"),
            BenchmarkQuery(qid="q3", question="three"),
        ],
        concurrency=1,
        top_k=3,
        timeout=10.0,
        client_factory=client_factory,
    )
    summary = _summarise(rows, elapsed_s=elapsed)

    assert summary.count == 3
    assert summary.successes == 2
    assert summary.failures == 1
    assert summary.rps is not None
    assert summary.rps > 0.0


def test_render_report_includes_targets_and_serializable_raw(tmp_path: Path) -> None:
    latency_rows = [
        QueryMeasurement(
            qid="q1",
            ok=True,
            status_code=200,
            latency_ms=1000.0,
            response_latency_ms={"total": 950.0},
        )
    ]
    throughput_rows = [
        QueryMeasurement(
            qid="q1",
            ok=True,
            status_code=200,
            latency_ms=900.0,
            response_latency_ms={"total": 850.0},
        )
    ]
    report = _render_report(
        ReportInput(
            api_url="http://api.test",
            golden=tmp_path / "golden.jsonl",
            health={"status": "ok", "pipeline_loaded": True},
            latency_summary=_summarise(latency_rows),
            throughput_summary=_summarise(throughput_rows, elapsed_s=0.4),
            indexing=IndexingMeasurement(ran=False),
            latency_rows=latency_rows,
            throughput_rows=throughput_rows,
        )
    )

    assert "# NFR benchmark" in report
    assert "p95 e2e latency" in report
    assert "Throughput" in report
    assert "Indexing time" in report
    assert "Not run. Pass `--run-indexing`" in report
