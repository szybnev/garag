"""Mock-based tests for `QueryPipeline`.

Reuses the stub retriever pattern from `test_pipeline.py` plus the
`httpx.MockTransport` pattern from `test_generator.py`, so the tests
exercise the real `HybridRetriever` + `Generator` wiring without a
running Qdrant, BM25 pickle, or Ollama.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.rag import ScoredChunk
from app.rag.generator import GenerationError, Generator
from app.rag.pipeline import HybridRetriever
from app.rag.query_pipeline import QueryPipeline
from app.schemas import QueryResponse


class _StubRetriever:
    def __init__(self, hits: list[ScoredChunk]) -> None:
        self.hits = hits
        self.last_query: str | None = None
        self.last_top_k: int | None = None

    def search(self, query: str, top_k: int = 20) -> list[ScoredChunk]:
        self.last_query = query
        self.last_top_k = top_k
        return list(self.hits)


class _StubReranker:
    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int = 5,
    ) -> list[ScoredChunk]:
        del query
        return list(candidates)[:top_k]


def _chunk(cid: str) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=cid,
        score=1.0,
        source="mitre_attack",
        title=f"title-{cid}",
        text=f"text of {cid}",
        url=f"https://example.test/{cid}",
        doc_id="mitre_attack:T0",
    )


def _valid_response_json(chunk_id: str = "mitre_attack:T0::0") -> str:
    return json.dumps(
        {
            "answer": "Canned mock answer for the QueryPipeline tests.",
            "citations": [{"chunk_id": chunk_id, "quote": "verbatim quote from the chunk body"}],
            "confidence": 0.9,
            "used_chunks": [chunk_id],
        }
    )


def _make_generator(handler: httpx.MockTransport) -> Generator:
    return Generator(
        base_url="http://ollama.test:11434",
        model="qwen3.5:test",
        client=httpx.Client(transport=handler),
    )


def _make_retriever(*, with_reranker: bool) -> HybridRetriever:
    dense = _StubRetriever([_chunk(f"mitre_attack:T0::{i}") for i in range(3)])
    sparse = _StubRetriever([_chunk(f"mitre_attack:T0::{i}") for i in range(3)])
    reranker = _StubReranker() if with_reranker else None
    return HybridRetriever(
        dense=dense,  # type: ignore[arg-type]
        sparse=sparse,  # type: ignore[arg-type]
        reranker=reranker,  # type: ignore[arg-type]
        fusion="rrf",
    )


def test_query_pipeline_returns_queryresponse_with_latency() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": _valid_response_json()}})

    pipeline = QueryPipeline(
        retriever=_make_retriever(with_reranker=True),
        generator=_make_generator(httpx.MockTransport(handler)),
    )
    response = pipeline.query("example question")
    assert isinstance(response, QueryResponse)
    assert response.latency_ms is not None
    assert response.confidence == 0.9


def test_query_pipeline_latency_keys_include_total_and_gen() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": _valid_response_json()}})

    pipeline = QueryPipeline(
        retriever=_make_retriever(with_reranker=True),
        generator=_make_generator(httpx.MockTransport(handler)),
    )
    response = pipeline.query("q")
    assert response.latency_ms is not None
    expected = {"dense", "sparse", "fusion", "rerank", "gen", "total"}
    assert expected <= set(response.latency_ms)
    for value in response.latency_ms.values():
        assert isinstance(value, float)
        assert value >= 0.0


def test_query_pipeline_without_reranker_omits_rerank_key() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": _valid_response_json()}})

    pipeline = QueryPipeline(
        retriever=_make_retriever(with_reranker=False),
        generator=_make_generator(httpx.MockTransport(handler)),
    )
    response = pipeline.query("q")
    assert response.latency_ms is not None
    assert "rerank" not in response.latency_ms
    assert "gen" in response.latency_ms


def test_query_pipeline_propagates_generation_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    pipeline = QueryPipeline(
        retriever=_make_retriever(with_reranker=True),
        generator=_make_generator(httpx.MockTransport(handler)),
    )
    with pytest.raises(GenerationError):
        pipeline.query("q")


def test_query_pipeline_default_candidate_k_and_top_k_applied() -> None:
    captured: dict[str, Any] = {}

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": _valid_response_json()}})

    dense = _StubRetriever([_chunk(f"mitre_attack:T0::{i}") for i in range(3)])
    sparse = _StubRetriever([_chunk(f"mitre_attack:T0::{i}") for i in range(3)])
    retriever = HybridRetriever(
        dense=dense,  # type: ignore[arg-type]
        sparse=sparse,  # type: ignore[arg-type]
        fusion="rrf",
    )
    pipeline = QueryPipeline(
        retriever=retriever,
        generator=_make_generator(httpx.MockTransport(handler)),
        candidate_k=17,
        top_k=3,
    )
    pipeline.query("q")
    captured["dense_top_k"] = dense.last_top_k
    captured["sparse_top_k"] = sparse.last_top_k
    assert captured["dense_top_k"] == 17
    assert captured["sparse_top_k"] == 17


def test_query_pipeline_call_site_overrides_k() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": _valid_response_json()}})

    dense = _StubRetriever([_chunk(f"mitre_attack:T0::{i}") for i in range(3)])
    sparse = _StubRetriever([_chunk(f"mitre_attack:T0::{i}") for i in range(3)])
    retriever = HybridRetriever(
        dense=dense,  # type: ignore[arg-type]
        sparse=sparse,  # type: ignore[arg-type]
        fusion="rrf",
    )
    pipeline = QueryPipeline(
        retriever=retriever,
        generator=_make_generator(httpx.MockTransport(handler)),
        candidate_k=20,
        top_k=5,
    )
    pipeline.query("q", candidate_k=7, top_k=2)
    assert dense.last_top_k == 7
    assert sparse.last_top_k == 7
