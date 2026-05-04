"""Hybrid retrieval orchestrator.

Combines dense (DenseEmbedder + Qdrant) and sparse (BM25Okapi) retrievers via
RRF or alpha-weighted fusion. The reranker (d7) and the LLM generator
(d9) plug in here as additional stages, but for d5 the orchestrator
stops after fusion — it returns the top-k fused chunks ready for
downstream consumption.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Literal

from app.rag.fusion import (
    DEFAULT_RRF_K,
    alpha_weighted_fusion,
    reciprocal_rank_fusion,
)
from app.rag.retriever_dense import DenseRetriever
from app.rag.retriever_sparse import SparseRetriever

if TYPE_CHECKING:
    from app.rag import ScoredChunk
    from app.rag.reranker import Reranker

FusionMethod = Literal["rrf", "alpha"]


class HybridRetriever:
    """Orchestrates dense + sparse retrieval, score fusion, and optional reranking."""

    def __init__(
        self,
        dense: DenseRetriever | None = None,
        sparse: SparseRetriever | None = None,
        reranker: Reranker | None = None,
        *,
        fusion: FusionMethod = "rrf",
        alpha: float = 0.5,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        self.dense = dense or DenseRetriever()
        self.sparse = sparse or SparseRetriever()
        self.reranker = reranker
        self.fusion = fusion
        self.alpha = alpha
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        *,
        candidate_k: int = 20,
        top_k: int = 10,
        timings: dict[str, float] | None = None,
    ) -> list[ScoredChunk]:
        """Run the hybrid pipeline.

        If `timings` is supplied, per-stage elapsed seconds are written
        into it under keys `dense`, `sparse`, `fusion`, and `rerank`
        (the last only when a reranker is attached). The caller owns
        the dict — we only write to it.
        """
        t0 = time.perf_counter()
        dense_hits = self.dense.search(query, top_k=candidate_k)
        if timings is not None:
            timings["dense"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        sparse_hits = self.sparse.search(query, top_k=candidate_k)
        if timings is not None:
            timings["sparse"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        if self.fusion == "rrf":
            fused = reciprocal_rank_fusion(dense_hits, sparse_hits, k=self.rrf_k, top_k=candidate_k)
        else:
            fused = alpha_weighted_fusion(
                dense_hits, sparse_hits, alpha=self.alpha, top_k=candidate_k
            )
        if timings is not None:
            timings["fusion"] = time.perf_counter() - t0

        if self.reranker is not None:
            t0 = time.perf_counter()
            reranked = self.reranker.rerank(query, fused, top_k=top_k)
            if timings is not None:
                timings["rerank"] = time.perf_counter() - t0
            return reranked
        return fused[:top_k]
