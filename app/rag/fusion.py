"""Score-list fusion strategies for hybrid retrieval.

Two strategies are exposed:

- **Reciprocal Rank Fusion (RRF)** — `1 / (k + rank)` summed across rankings.
  Score-agnostic, the canonical hybrid retrieval recipe. `k=60` is the
  default from the original paper.
- **Alpha-weighted min-max fusion** — normalise each list to `[0, 1]` and
  combine with `alpha * dense + (1 - alpha) * sparse`. Sensitive to score
  scale, so we run min-max normalisation per list. Useful when one
  retriever's scores are systematically more informative than the other.

Both functions take **already-sorted** lists of `ScoredChunk` and return
a fused list sorted by descending fused score.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from app.rag import ScoredChunk

if TYPE_CHECKING:
    from collections.abc import Iterable

DEFAULT_RRF_K = 60


def _by_chunk_id(chunks: Iterable[ScoredChunk]) -> dict[str, ScoredChunk]:
    out: dict[str, ScoredChunk] = {}
    for c in chunks:
        # keep first occurrence (highest-ranked)
        out.setdefault(c.chunk_id, c)
    return out


def reciprocal_rank_fusion(
    *rankings: list[ScoredChunk],
    k: int = DEFAULT_RRF_K,
    top_k: int | None = None,
) -> list[ScoredChunk]:
    """Combine multiple rankings via RRF.

    Returns a fused list sorted by descending RRF score, deduped by `chunk_id`.
    Metadata for each surviving chunk is taken from the first list it appears in.
    """
    if not rankings:
        return []

    fused: dict[str, float] = defaultdict(float)
    metadata: dict[str, ScoredChunk] = {}

    for ranking in rankings:
        for rank, chunk in enumerate(ranking):
            fused[chunk.chunk_id] += 1.0 / (k + rank + 1)
            metadata.setdefault(chunk.chunk_id, chunk)

    sorted_ids = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    if top_k is not None:
        sorted_ids = sorted_ids[:top_k]

    out: list[ScoredChunk] = []
    for chunk_id, score in sorted_ids:
        meta = metadata[chunk_id]
        out.append(
            ScoredChunk(
                chunk_id=chunk_id,
                score=float(score),
                source=meta.source,
                title=meta.title,
                text=meta.text,
                url=meta.url,
                doc_id=meta.doc_id,
            )
        )
    return out


def _min_max(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if hi - lo < 1e-12:
        return dict.fromkeys(values, 0.0)
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def alpha_weighted_fusion(
    dense: list[ScoredChunk],
    sparse: list[ScoredChunk],
    alpha: float,
    *,
    top_k: int | None = None,
) -> list[ScoredChunk]:
    """Combine dense and sparse rankings via min-max normalisation + linear blend.

    `alpha=1.0` → dense only, `alpha=0.0` → sparse only.
    """
    if not 0.0 <= alpha <= 1.0:
        msg = f"alpha must be in [0, 1], got {alpha}"
        raise ValueError(msg)

    dense_scores = {c.chunk_id: c.score for c in dense}
    sparse_scores = {c.chunk_id: c.score for c in sparse}
    dense_norm = _min_max(dense_scores)
    sparse_norm = _min_max(sparse_scores)

    metadata: dict[str, ScoredChunk] = {}
    for c in dense:
        metadata.setdefault(c.chunk_id, c)
    for c in sparse:
        metadata.setdefault(c.chunk_id, c)

    all_ids = set(dense_scores) | set(sparse_scores)
    fused: dict[str, float] = {
        cid: alpha * dense_norm.get(cid, 0.0) + (1.0 - alpha) * sparse_norm.get(cid, 0.0)
        for cid in all_ids
    }

    sorted_ids = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    if top_k is not None:
        sorted_ids = sorted_ids[:top_k]

    out: list[ScoredChunk] = []
    for chunk_id, score in sorted_ids:
        meta = metadata[chunk_id]
        out.append(
            ScoredChunk(
                chunk_id=chunk_id,
                score=float(score),
                source=meta.source,
                title=meta.title,
                text=meta.text,
                url=meta.url,
                doc_id=meta.doc_id,
            )
        )
    return out
