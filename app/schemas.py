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
