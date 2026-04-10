"""Retrieval pipeline building blocks for GaRAG."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A retrieval hit — what dense / sparse / hybrid all return."""

    chunk_id: str
    score: float
    source: str
    title: str
    text: str
    url: str | None
    doc_id: str
