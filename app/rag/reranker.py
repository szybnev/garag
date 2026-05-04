"""Cross-encoder reranker — `BAAI/bge-reranker-v2-m3`.

Takes a list of `ScoredChunk` candidates from the hybrid retriever and
re-scores each `(query, chunk.text)` pair with a cross-encoder. The
cross-encoder sees both the query and the chunk in a single forward
pass, so it can model term interactions that a bi-encoder cannot — at
the cost of `O(query x candidates)` cross-encoder inference time.

We use the `BAAI/bge-reranker-v2-m3` checkpoint (multilingual, 567M
params) as the tuned cross-encoder stage for the MVP pipeline.
"""

from __future__ import annotations

from FlagEmbedding import FlagReranker

from app.config import settings
from app.rag import ScoredChunk

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


class Reranker:
    """Wraps `FlagReranker` for the GaRAG retrieval pipeline."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        use_fp16: bool | None = None,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name or settings.reranker_model
        use_fp16 = settings.reranker_use_fp16 if use_fp16 is None else use_fp16
        device = device or settings.reranker_device
        self.model = FlagReranker(
            self.model_name,
            use_fp16=use_fp16,
            normalize=True,
            devices=[device],
            max_length=max_length,
        )

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int = 5,
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        pairs = [(query, c.text) for c in candidates]
        scores = self.model.compute_score(pairs)
        # FlagReranker returns float for one pair, list[float] for many
        if isinstance(scores, float):
            scores = [scores]

        scored: list[tuple[float, ScoredChunk]] = [
            (float(score), candidate) for score, candidate in zip(scores, candidates, strict=True)
        ]
        scored.sort(key=lambda kv: kv[0], reverse=True)

        return [
            ScoredChunk(
                chunk_id=c.chunk_id,
                score=score,
                source=c.source,
                title=c.title,
                text=c.text,
                url=c.url,
                doc_id=c.doc_id,
            )
            for score, c in scored[:top_k]
        ]
