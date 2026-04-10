"""Hybrid retrieval orchestrator.

Combines dense (bge-m3 + Qdrant) and sparse (BM25Okapi) retrievers via
RRF or alpha-weighted fusion. The reranker (d7) and the LLM generator
(d9) plug in here as additional stages, but for d5 the orchestrator
stops after fusion — it returns the top-k fused chunks ready for
downstream consumption.
"""

from __future__ import annotations

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

FusionMethod = Literal["rrf", "alpha"]


class HybridRetriever:
    """Orchestrates dense + sparse retrieval and score fusion."""

    def __init__(
        self,
        dense: DenseRetriever | None = None,
        sparse: SparseRetriever | None = None,
        *,
        fusion: FusionMethod = "rrf",
        alpha: float = 0.5,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        self.dense = dense or DenseRetriever()
        self.sparse = sparse or SparseRetriever()
        self.fusion = fusion
        self.alpha = alpha
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        *,
        candidate_k: int = 20,
        top_k: int = 10,
    ) -> list[ScoredChunk]:
        dense_hits = self.dense.search(query, top_k=candidate_k)
        sparse_hits = self.sparse.search(query, top_k=candidate_k)

        if self.fusion == "rrf":
            return reciprocal_rank_fusion(dense_hits, sparse_hits, k=self.rrf_k, top_k=top_k)
        return alpha_weighted_fusion(dense_hits, sparse_hits, alpha=self.alpha, top_k=top_k)
