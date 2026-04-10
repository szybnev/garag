"""Invariants for `app.rag.fusion`."""

from __future__ import annotations

import pytest

from app.rag import ScoredChunk
from app.rag.fusion import alpha_weighted_fusion, reciprocal_rank_fusion


def _chunk(cid: str, score: float) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=cid,
        score=score,
        source="mitre_attack",
        title=f"title-{cid}",
        text=f"text-{cid}",
        url=None,
        doc_id="mitre_attack:T0",
    )


# ---------- RRF ----------


def test_rrf_single_ranking_preserves_order() -> None:
    a = [_chunk("a", 10.0), _chunk("b", 8.0), _chunk("c", 5.0)]
    fused = reciprocal_rank_fusion(a)
    assert [c.chunk_id for c in fused] == ["a", "b", "c"]


def test_rrf_two_rankings_full_overlap_keeps_order() -> None:
    a = [_chunk("x", 1.0), _chunk("y", 1.0), _chunk("z", 1.0)]
    b = [_chunk("x", 1.0), _chunk("y", 1.0), _chunk("z", 1.0)]
    fused = reciprocal_rank_fusion(a, b)
    assert [c.chunk_id for c in fused] == ["x", "y", "z"]


def test_rrf_chunk_in_both_outranks_chunk_in_one() -> None:
    a = [_chunk("alpha", 1.0), _chunk("beta", 1.0)]
    b = [_chunk("beta", 1.0), _chunk("gamma", 1.0)]
    fused = reciprocal_rank_fusion(a, b)
    # beta appears in both → should rank above alpha and gamma
    assert fused[0].chunk_id == "beta"


def test_rrf_top_k_truncates() -> None:
    a = [_chunk(c, 1.0) for c in "abcdef"]
    fused = reciprocal_rank_fusion(a, top_k=3)
    assert len(fused) == 3
    assert [c.chunk_id for c in fused] == ["a", "b", "c"]


def test_rrf_empty_input_returns_empty() -> None:
    assert reciprocal_rank_fusion() == []
    assert reciprocal_rank_fusion([], []) == []


def test_rrf_score_is_positive_and_decreasing() -> None:
    a = [_chunk(c, 0.0) for c in "abcde"]
    b = [_chunk(c, 0.0) for c in "abcde"]
    fused = reciprocal_rank_fusion(a, b)
    scores = [c.score for c in fused]
    assert all(s > 0 for s in scores)
    assert scores == sorted(scores, reverse=True)


# ---------- alpha-weighted ----------


def test_alpha_one_returns_dense_order() -> None:
    dense = [_chunk("a", 9.0), _chunk("b", 5.0), _chunk("c", 1.0)]
    sparse = [_chunk("c", 9.0), _chunk("b", 5.0), _chunk("a", 1.0)]
    fused = alpha_weighted_fusion(dense, sparse, alpha=1.0)
    assert [c.chunk_id for c in fused[:3]] == ["a", "b", "c"]


def test_alpha_zero_returns_sparse_order() -> None:
    dense = [_chunk("a", 9.0), _chunk("b", 5.0), _chunk("c", 1.0)]
    sparse = [_chunk("c", 9.0), _chunk("b", 5.0), _chunk("a", 1.0)]
    fused = alpha_weighted_fusion(dense, sparse, alpha=0.0)
    assert [c.chunk_id for c in fused[:3]] == ["c", "b", "a"]


def test_alpha_half_combines_lists() -> None:
    dense = [_chunk("d1", 10.0), _chunk("shared", 5.0)]
    sparse = [_chunk("s1", 10.0), _chunk("shared", 5.0)]
    fused = alpha_weighted_fusion(dense, sparse, alpha=0.5)
    ids = {c.chunk_id for c in fused}
    assert ids == {"d1", "shared", "s1"}


def test_alpha_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match="alpha must be in"):
        alpha_weighted_fusion([], [], alpha=1.5)
    with pytest.raises(ValueError, match="alpha must be in"):
        alpha_weighted_fusion([], [], alpha=-0.1)


def test_alpha_top_k_truncates() -> None:
    dense = [_chunk(c, float(10 - i)) for i, c in enumerate("abcdef")]
    sparse = [_chunk(c, float(10 - i)) for i, c in enumerate("ghijkl")]
    fused = alpha_weighted_fusion(dense, sparse, alpha=0.5, top_k=4)
    assert len(fused) == 4


def test_alpha_handles_constant_score_list() -> None:
    # min-max normaliser must not divide by zero
    dense = [_chunk(c, 7.0) for c in "abc"]
    sparse = [_chunk(c, 3.0) for c in "abc"]
    fused = alpha_weighted_fusion(dense, sparse, alpha=0.5)
    assert {c.chunk_id for c in fused} == {"a", "b", "c"}
