"""FastAPI runtime tests with an injected fake pipeline."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.rag.generator import GenerationError
from app.schemas import QueryResponse


class _FakePipeline:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def query(
        self,
        question: str,
        *,
        candidate_k: int | None = None,
        top_k: int | None = None,
    ) -> QueryResponse:
        self.calls.append({"question": question, "candidate_k": candidate_k, "top_k": top_k})
        if self.error is not None:
            raise self.error
        return QueryResponse(
            answer="PowerShell can be abused by adversaries for execution.",
            citations=[],
            confidence=0.9,
            used_chunks=["mitre_attack:T1059.001::0"],
            latency_ms={"total": 12.3},
        )


@pytest.fixture
def fake_pipeline() -> _FakePipeline:
    return _FakePipeline()


def _client_for(pipeline: _FakePipeline, *, mount_gradio: bool = False) -> TestClient:
    app = create_app(pipeline_factory=lambda: pipeline, mount_gradio=mount_gradio)
    return TestClient(app)


def test_health_reports_loaded_pipeline(fake_pipeline: _FakePipeline) -> None:
    with _client_for(fake_pipeline) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "pipeline_loaded": True, "version": "0.1.0"}


def test_query_returns_query_response_and_passes_top_k(fake_pipeline: _FakePipeline) -> None:
    with _client_for(fake_pipeline) as client:
        response = client.post("/query", json={"query": "What is PowerShell?", "top_k": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("PowerShell")
    assert body["confidence"] == 0.9
    assert body["latency_ms"] == {"total": 12.3}
    assert fake_pipeline.calls == [
        {"question": "What is PowerShell?", "candidate_k": 20, "top_k": 3}
    ]


def test_query_rejects_invalid_payload(fake_pipeline: _FakePipeline) -> None:
    with _client_for(fake_pipeline) as client:
        response = client.post("/query", json={"query": "x", "top_k": 3})

    assert response.status_code == 422
    assert fake_pipeline.calls == []


def test_generation_error_maps_to_502() -> None:
    pipeline = _FakePipeline(error=GenerationError("ollama failed"))
    with _client_for(pipeline) as client:
        response = client.post("/query", json={"query": "What is PowerShell?"})

    assert response.status_code == 502
    assert response.json()["detail"] == "ollama failed"


def test_generic_pipeline_error_maps_to_500() -> None:
    pipeline = _FakePipeline(error=RuntimeError("qdrant failed"))
    with _client_for(pipeline) as client:
        response = client.post("/query", json={"query": "What is PowerShell?"})

    assert response.status_code == 500
    assert response.json()["detail"] == "Query pipeline failed"


def test_metrics_endpoint_exposes_query_counters(fake_pipeline: _FakePipeline) -> None:
    with _client_for(fake_pipeline) as client:
        client.post("/query", json={"query": "What is PowerShell?"})
        response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "garag_query_requests_total" in response.text
    assert 'status="success"' in response.text
    assert "garag_query_latency_seconds" in response.text


def test_gradio_is_mounted(fake_pipeline: _FakePipeline) -> None:
    with _client_for(fake_pipeline, mount_gradio=True) as client:
        response = client.get("/gradio/")

    assert response.status_code != 404
