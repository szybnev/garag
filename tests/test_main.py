"""FastAPI runtime tests with an injected fake pipeline."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import _format_sources, _target_generator_model_label, build_pipeline, create_app
from app.rag.generator import GenerationError
from app.schemas import Citation, QueryResponse


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


def test_format_sources_includes_chunk_source_url_and_quote() -> None:
    rendered = _format_sources(
        [
            Citation(
                chunk_id="mitre_attack:T1059::0",
                source="mitre_attack",
                url="https://example.test/T1059",
                quote="Command and Scripting Interpreter.",
            )
        ]
    )

    assert "mitre_attack:T1059::0" in rendered
    assert "mitre_attack" in rendered
    assert "https://example.test/T1059" in rendered
    assert "Command and Scripting Interpreter." in rendered


def test_format_sources_handles_empty_citations() -> None:
    assert _format_sources([]) == "No explicit sources were returned by the generator."


def test_target_generator_model_label_uses_openai_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "llm_provider", "openai_compat")
    monkeypatch.setattr(settings, "openai_model", "ibm/granite-3.2-8b")

    assert _target_generator_model_label() == (
        "**Target generator model:** `ibm/granite-3.2-8b` (`openai_compat`)"
    )


def test_target_generator_model_label_uses_ollama_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(settings, "ollama_model", "qwen3.5:35b")

    assert _target_generator_model_label() == (
        "**Target generator model:** `qwen3.5:35b` (`ollama`)"
    )


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


def test_build_pipeline_passes_configured_qdrant_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeReranker:
        pass

    class FakeDenseRetriever:
        def __init__(self, *, qdrant_url: str, collection: str) -> None:
            captured["qdrant_url"] = qdrant_url
            captured["collection"] = collection

    class FakeHybridRetriever:
        def __init__(self, **kwargs: Any) -> None:
            captured["dense"] = kwargs["dense"]

    class FakeGenerator:
        pass

    class FakeQueryPipeline:
        def __init__(self, **kwargs: Any) -> None:
            captured["pipeline"] = kwargs

    monkeypatch.setattr(settings, "qdrant_url", "http://qdrant.test:6333")
    monkeypatch.setattr(settings, "qdrant_collection", "garag_test")
    monkeypatch.setattr("app.main.Reranker", FakeReranker)
    monkeypatch.setattr("app.main.DenseRetriever", FakeDenseRetriever)
    monkeypatch.setattr("app.main.HybridRetriever", FakeHybridRetriever)
    monkeypatch.setattr("app.main.Generator", FakeGenerator)
    monkeypatch.setattr("app.main.QueryPipeline", FakeQueryPipeline)

    build_pipeline()

    assert captured["qdrant_url"] == "http://qdrant.test:6333"
    assert captured["collection"] == "garag_test"
    assert captured["pipeline"]["retriever"] is not None
