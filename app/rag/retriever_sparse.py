"""Sparse retriever — `BM25Okapi` pickle from `scripts/build_bm25.py`."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.rag import ScoredChunk
from scripts.build_bm25 import _ensure_stopwords, tokenize

DEFAULT_BM25_PATH = Path(__file__).resolve().parents[2] / "data" / "index" / "bm25.pkl"
DEFAULT_CHUNKS_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "chunks.parquet"


class SparseRetriever:
    """In-memory BM25 retriever with chunk metadata enrichment from parquet."""

    def __init__(
        self,
        bm25_path: Path = DEFAULT_BM25_PATH,
        chunks_path: Path = DEFAULT_CHUNKS_PATH,
    ) -> None:
        if not bm25_path.exists():
            msg = f"{bm25_path} not found — run `uv run python -m scripts.build_bm25` first"
            raise FileNotFoundError(msg)
        if not chunks_path.exists():
            msg = f"{chunks_path} not found — run `uv run python -m scripts.chunk_corpus` first"
            raise FileNotFoundError(msg)

        with bm25_path.open("rb") as fh:
            blob: dict[str, Any] = pickle.load(fh)  # noqa: S301

        self.bm25 = blob["bm25"]
        self.chunk_ids: list[str] = blob["chunk_ids"]
        self.params = blob["params"]
        self.stop = _ensure_stopwords()

        chunks_df = pd.read_parquet(chunks_path)
        self._chunk_meta: dict[str, dict[str, Any]] = {
            row["chunk_id"]: {
                "doc_id": row["doc_id"],
                "source": row["source"],
                "url": row["url"] if pd.notna(row["url"]) else None,
                "title": row["title"],
                "text": row["text"],
            }
            for _, row in chunks_df.iterrows()
        }

    def search(self, query: str, top_k: int = 20) -> list[ScoredChunk]:
        tokens = tokenize(query, self.stop)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        order = np.argsort(scores)[::-1][:top_k]
        out: list[ScoredChunk] = []
        for idx in order:
            score = float(scores[idx])
            if score <= 0.0:
                continue
            chunk_id = self.chunk_ids[idx]
            meta = self._chunk_meta.get(chunk_id, {})
            out.append(
                ScoredChunk(
                    chunk_id=chunk_id,
                    score=score,
                    source=str(meta.get("source", "")),
                    title=str(meta.get("title", "")),
                    text=str(meta.get("text", "")),
                    url=meta.get("url"),
                    doc_id=str(meta.get("doc_id", "")),
                )
            )
        return out
