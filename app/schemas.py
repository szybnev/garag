"""Pydantic schemas for the GaRAG pipeline."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SourceName = Literal[
    "mitre_attack",
    "mitre_atlas",
    "owasp",
    "hackerone",
    "man_pages",
]


class Document(BaseModel):
    """A normalised document from one of the five corpus sources.

    Produced by the `scripts/parsers/*` modules, consumed by `scripts/chunk_corpus.py`
    on d3. The `doc_id` is namespaced by source (`"{source}:{natural_id}"`) so that
    collisions across sources are impossible.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str = Field(min_length=3, pattern=r"^[a-z_]+:[^\s]+$")
    source: SourceName
    url: str | None = None
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("doc_id")
    @classmethod
    def _doc_id_prefix_matches_source(cls, v: str) -> str:
        return v

    def model_post_init(self, _context: Any, /) -> None:
        prefix = self.doc_id.split(":", 1)[0]
        if prefix != self.source:
            raise ValueError(f"doc_id prefix {prefix!r} does not match source {self.source!r}")


class Chunk(BaseModel):
    """A token-bounded slice of a `Document`, produced by `scripts/chunk_corpus.py`.

    `chunk_id` is `f"{doc_id}::{index}"` so chunks are globally unique and stable
    across re-runs. `char_start` / `char_end` index into the parent document's text.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str = Field(min_length=5, pattern=r"^[a-z_]+:[^\s]+::\d+$")
    doc_id: str = Field(min_length=3, pattern=r"^[a-z_]+:[^\s]+$")
    source: SourceName
    url: str | None = None
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    token_count: int = Field(ge=1)

    def model_post_init(self, _context: Any, /) -> None:
        prefix = self.chunk_id.split(":", 1)[0]
        if prefix != self.source:
            raise ValueError(f"chunk_id prefix {prefix!r} does not match source {self.source!r}")
        if not self.chunk_id.startswith(self.doc_id + "::"):
            raise ValueError(
                f"chunk_id {self.chunk_id!r} must start with doc_id {self.doc_id!r} + '::'"
            )
        if self.char_end < self.char_start:
            raise ValueError(
                f"char_end ({self.char_end}) must be >= char_start ({self.char_start})"
            )


# ──────────────────────────────────────────────────────────────────────
#  Public API contracts (used by FastAPI on d10 and the generator on d9)
# ──────────────────────────────────────────────────────────────────────


class Citation(BaseModel):
    """A single citation surfaced alongside a generated answer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str = Field(min_length=3)
    source: SourceName
    url: str | None = None
    quote: str = Field(min_length=1, max_length=600)


class QueryRequest(BaseModel):
    """Public POST `/query` payload."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=12, ge=1, le=20)


class QueryResponse(BaseModel):
    """Public POST `/query` response.

    `confidence` is the model's self-reported confidence in `[0, 1]`,
    `used_chunks` is the ordered list of `chunk_id`s that the model
    actually based the answer on (a subset of the retrieved candidates).
    `latency_ms` is an optional per-stage timing breakdown populated by
    `QueryPipeline`; keys are free-form stage names (`dense`, `sparse`,
    `fusion`, `rerank`, `gen`, `total`, …) and values are milliseconds.
    """

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    used_chunks: list[str] = Field(default_factory=list)
    latency_ms: dict[str, float] | None = None
