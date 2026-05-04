"""End-to-end query pipeline: retrieve → generate → `QueryResponse`.

Thin composition of `HybridRetriever` + `Generator` that collects
per-stage timings and surfaces them on `QueryResponse.latency_ms`.
This is also the future entry point for FastAPI `/query` on d10.5 /
`garag-zqc.20` — everything above this class is transport (ASGI,
middleware, metrics), and everything below is retrieval + generation.

Dependency injection is deliberate: the pipeline takes a prebuilt
`HybridRetriever` and `Generator`, never constructs them lazily. That
keeps unit tests GPU-free and Qdrant-free (mock both deps) and lets the
FastAPI lifespan handler own the real wiring.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.guardrails import GuardrailClient
    from app.rag.generator import Generator
    from app.rag.pipeline import HybridRetriever
    from app.schemas import QueryResponse


class QueryPipeline:
    """Compose `HybridRetriever` + `Generator` with per-stage latency."""

    def __init__(
        self,
        retriever: HybridRetriever,
        generator: Generator,
        *,
        guardrails: GuardrailClient | None = None,
        candidate_k: int = 20,
        top_k: int = 12,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.guardrails = guardrails
        self.candidate_k = candidate_k
        self.top_k = top_k

    def query(
        self,
        question: str,
        *,
        candidate_k: int | None = None,
        top_k: int | None = None,
    ) -> QueryResponse:
        """Run retrieval + generation and populate `latency_ms`.

        Keys written to `latency_ms`: `guardrails_in` (when enabled),
        `dense`, `sparse`, `fusion`, `rerank` (only if the retriever has
        a reranker), `gen`, `guardrails_out` (when enabled), `total`.
        Values are milliseconds rounded to 1 decimal.
        """
        ck = candidate_k if candidate_k is not None else self.candidate_k
        tk = top_k if top_k is not None else self.top_k

        timings: dict[str, float] = {}
        t_total = time.perf_counter()

        if self.guardrails is not None:
            t_guardrails = time.perf_counter()
            self.guardrails.scan_input(question)
            timings["guardrails_in"] = time.perf_counter() - t_guardrails

        chunks = self.retriever.retrieve(question, candidate_k=ck, top_k=tk, timings=timings)

        t_gen = time.perf_counter()
        response = self.generator.generate(question, chunks)
        timings["gen"] = time.perf_counter() - t_gen

        if self.guardrails is not None:
            t_guardrails = time.perf_counter()
            self.guardrails.scan_output(question=question, chunks=chunks, response=response)
            timings["guardrails_out"] = time.perf_counter() - t_guardrails

        timings["total"] = time.perf_counter() - t_total

        latency_ms = {key: round(value * 1000, 1) for key, value in timings.items()}
        return response.model_copy(update={"latency_ms": latency_ms})
