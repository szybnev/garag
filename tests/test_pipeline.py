"""Mock-based tests for `HybridRetriever`.

We don't want unit tests to depend on a running Qdrant or a populated
BM25 pickle, so we inject fake `dense` / `sparse` / `reranker` objects
that just return canned `ScoredChunk` lists. This exercises the
fusion-and-rerank wiring without touching GPU or disk.
"""

from __future__ import annotations

from app.rag import ScoredChunk
from app.rag.pipeline import HybridRetriever


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
    """Reverses the candidate order so we can verify rerank was applied."""

    def __init__(self) -> None:
        self.last_query: str | None = None
        self.last_candidates: list[ScoredChunk] | None = None

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int = 5,
    ) -> list[ScoredChunk]:
        self.last_query = query
        self.last_candidates = list(candidates)
        return list(reversed(candidates))[:top_k]


def _chunk(cid: str, score: float) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=cid,
        score=score,
        source="mitre_attack",
        title=f"title-{cid}",
        text=f"text of {cid}",
        url=None,
        doc_id="mitre_attack:T0",
    )


def _make_retriever(
    *,
    dense_hits: list[ScoredChunk],
    sparse_hits: list[ScoredChunk],
    reranker: _StubReranker | None = None,
    fusion: str = "rrf",
    alpha: float = 0.5,
) -> HybridRetriever:
    return HybridRetriever(
        dense=_StubRetriever(dense_hits),  # type: ignore[arg-type]
        sparse=_StubRetriever(sparse_hits),  # type: ignore[arg-type]
        reranker=reranker,  # type: ignore[arg-type]
        fusion=fusion,  # type: ignore[arg-type]
        alpha=alpha,
    )


def test_hybrid_rrf_returns_top_k() -> None:
    dense = [_chunk(c, 0.9 - i * 0.1) for i, c in enumerate("abcde")]
    sparse = [_chunk(c, 0.8 - i * 0.1) for i, c in enumerate("abcde")]
    retriever = _make_retriever(dense_hits=dense, sparse_hits=sparse, fusion="rrf")
    out = retriever.retrieve("query", candidate_k=5, top_k=3)
    assert len(out) == 3
    assert {c.chunk_id for c in out} <= {"a", "b", "c", "d", "e"}


def test_hybrid_alpha_returns_top_k() -> None:
    dense = [_chunk(c, 1.0) for c in "abc"]
    sparse = [_chunk(c, 1.0) for c in "abc"]
    retriever = _make_retriever(dense_hits=dense, sparse_hits=sparse, fusion="alpha", alpha=0.3)
    out = retriever.retrieve("query", candidate_k=3, top_k=2)
    assert len(out) == 2


def test_hybrid_passes_query_to_both_retrievers() -> None:
    dense_stub = _StubRetriever([_chunk("a", 1.0)])
    sparse_stub = _StubRetriever([_chunk("a", 1.0)])
    retriever = HybridRetriever(
        dense=dense_stub,  # type: ignore[arg-type]
        sparse=sparse_stub,  # type: ignore[arg-type]
        fusion="rrf",
    )
    retriever.retrieve("hello world", candidate_k=10, top_k=5)
    assert dense_stub.last_query == "hello world"
    assert sparse_stub.last_query == "hello world"
    assert dense_stub.last_top_k == 10
    assert sparse_stub.last_top_k == 10


def test_hybrid_without_reranker_just_truncates() -> None:
    dense = [_chunk(c, 1.0) for c in "abcde"]
    sparse = [_chunk(c, 1.0) for c in "abcde"]
    retriever = _make_retriever(dense_hits=dense, sparse_hits=sparse, fusion="rrf")
    out = retriever.retrieve("q", candidate_k=5, top_k=2)
    assert len(out) == 2


def test_hybrid_with_reranker_reorders_and_truncates() -> None:
    dense = [_chunk(c, 1.0 - i * 0.1) for i, c in enumerate("abcde")]
    sparse = [_chunk(c, 1.0 - i * 0.1) for i, c in enumerate("abcde")]
    reranker = _StubReranker()
    retriever = _make_retriever(
        dense_hits=dense,
        sparse_hits=sparse,
        reranker=reranker,
        fusion="rrf",
    )
    out = retriever.retrieve("q", candidate_k=5, top_k=3)
    assert reranker.last_query == "q"
    assert reranker.last_candidates is not None
    assert len(reranker.last_candidates) == 5
    assert len(out) == 3


def test_hybrid_reranker_sees_fused_not_raw() -> None:
    dense = [_chunk("only-dense", 0.9)]
    sparse = [_chunk("only-sparse", 0.8)]
    reranker = _StubReranker()
    retriever = _make_retriever(
        dense_hits=dense,
        sparse_hits=sparse,
        reranker=reranker,
        fusion="rrf",
    )
    retriever.retrieve("q", candidate_k=5, top_k=5)
    cids = {c.chunk_id for c in (reranker.last_candidates or [])}
    assert cids == {"only-dense", "only-sparse"}


def test_hybrid_retrieve_populates_timings_with_reranker() -> None:
    dense = [_chunk(c, 1.0) for c in "abc"]
    sparse = [_chunk(c, 1.0) for c in "abc"]
    reranker = _StubReranker()
    retriever = _make_retriever(
        dense_hits=dense, sparse_hits=sparse, reranker=reranker, fusion="rrf"
    )
    timings: dict[str, float] = {}
    retriever.retrieve("q", candidate_k=3, top_k=2, timings=timings)
    assert set(timings) == {"dense", "sparse", "fusion", "rerank"}
    for value in timings.values():
        assert isinstance(value, float)
        assert value >= 0.0


def test_hybrid_retrieve_timings_without_reranker_skips_rerank_key() -> None:
    dense = [_chunk(c, 1.0) for c in "ab"]
    sparse = [_chunk(c, 1.0) for c in "ab"]
    retriever = _make_retriever(dense_hits=dense, sparse_hits=sparse, fusion="rrf")
    timings: dict[str, float] = {}
    retriever.retrieve("q", candidate_k=2, top_k=2, timings=timings)
    assert set(timings) == {"dense", "sparse", "fusion"}
    assert "rerank" not in timings


def test_hybrid_retrieve_backward_compat_without_timings_kwarg() -> None:
    """Calling retrieve() without timings= must behave exactly as before."""
    dense = [_chunk(c, 1.0 - i * 0.1) for i, c in enumerate("abcde")]
    sparse = [_chunk(c, 1.0 - i * 0.1) for i, c in enumerate("abcde")]
    retriever = _make_retriever(dense_hits=dense, sparse_hits=sparse, fusion="rrf")
    out = retriever.retrieve("q", candidate_k=5, top_k=3)
    assert len(out) == 3
