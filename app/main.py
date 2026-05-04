"""FastAPI runtime entrypoint for the GaRAG MVP."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import gradio as gr
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

from app.config import settings
from app.rag.generator import GenerationError, Generator
from app.rag.pipeline import HybridRetriever
from app.rag.query_pipeline import QueryPipeline
from app.rag.reranker import Reranker
from app.rag.retriever_dense import DenseRetriever
from app.schemas import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

PipelineFactory = Callable[[], Any]


def build_pipeline() -> QueryPipeline:
    """Construct the real retrieval + generation pipeline for runtime use."""
    reranker = Reranker()
    dense = DenseRetriever(
        qdrant_url=settings.qdrant_url,
        collection=settings.qdrant_collection,
    )
    retriever = HybridRetriever(
        dense=dense,
        reranker=reranker,
        fusion=settings.fusion_method,
        alpha=settings.fusion_alpha,
        rrf_k=settings.rrf_k,
    )
    generator = Generator()
    return QueryPipeline(
        retriever=retriever,
        generator=generator,
        candidate_k=settings.retrieve_top_k,
        top_k=settings.rerank_top_k,
    )


def create_app(
    *,
    pipeline_factory: PipelineFactory | None = None,
    mount_gradio: bool = True,
) -> FastAPI:
    """Create the ASGI application.

    `pipeline_factory` is injectable so tests can avoid loading Qdrant, BM25,
    Hugging Face models, and Ollama.
    """
    registry = CollectorRegistry()
    query_requests = Counter(
        "garag_query_requests_total",
        "Total POST /query requests.",
        ["status"],
        registry=registry,
    )
    query_errors = Counter(
        "garag_query_errors_total",
        "Total POST /query failures by error type.",
        ["error_type"],
        registry=registry,
    )
    query_latency = Histogram(
        "garag_query_latency_seconds",
        "POST /query wall-clock latency.",
        ["status"],
        registry=registry,
    )

    factory = pipeline_factory or build_pipeline

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.pipeline = factory()
        yield
        app.state.pipeline = None

    app = FastAPI(title="GaRAG", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        return {
            "status": "ok",
            "pipeline_loaded": _pipeline_loaded(request.app),
            "version": app.version,
        }

    @app.post("/query", response_model=QueryResponse)
    def query(payload: QueryRequest, request: Request) -> QueryResponse:
        pipeline = _get_pipeline(request.app)
        started = time.perf_counter()
        try:
            response = pipeline.query(
                payload.query,
                candidate_k=settings.retrieve_top_k,
                top_k=payload.top_k,
            )
        except GenerationError as exc:
            query_requests.labels(status="error").inc()
            query_errors.labels(error_type="generation").inc()
            query_latency.labels(status="error").observe(time.perf_counter() - started)
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("query pipeline failed")
            query_requests.labels(status="error").inc()
            query_errors.labels(error_type=type(exc).__name__).inc()
            query_latency.labels(status="error").observe(time.perf_counter() - started)
            raise HTTPException(status_code=500, detail="Query pipeline failed") from exc

        query_requests.labels(status="success").inc()
        query_latency.labels(status="success").observe(time.perf_counter() - started)
        return response

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    if mount_gradio:
        demo = _build_gradio_app(app)
        app = gr.mount_gradio_app(app, demo, path="/gradio")

    return app


def _pipeline_loaded(app: FastAPI) -> bool:
    return getattr(app.state, "pipeline", None) is not None


def _get_pipeline(app: FastAPI) -> Any:
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Query pipeline is not loaded")
    return pipeline


def _build_gradio_app(app: FastAPI) -> gr.Blocks:
    def ask(question: str, top_k: int) -> tuple[str, str, str]:
        pipeline = _get_pipeline(app)
        response: QueryResponse = pipeline.query(
            question,
            candidate_k=settings.retrieve_top_k,
            top_k=int(top_k),
        )
        citations = "\n\n".join(
            f"- `{c.chunk_id}` ({c.source})\n  {c.quote}" for c in response.citations
        )
        latency = json.dumps(response.latency_ms or {}, indent=2, ensure_ascii=False)
        return response.answer, citations, latency

    with gr.Blocks(title="GaRAG") as demo:
        gr.Markdown("# GaRAG")
        question = gr.Textbox(label="Question", lines=3)
        top_k = gr.Slider(label="Top K", minimum=1, maximum=20, value=5, step=1)
        submit = gr.Button("Ask")
        answer = gr.Textbox(label="Answer", lines=8)
        citations = gr.Markdown(label="Citations")
        latency = gr.Code(label="Latency", language="json")
        submit.click(ask, inputs=[question, top_k], outputs=[answer, citations, latency])

    return demo


app = create_app()
