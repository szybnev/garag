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

    def model_post_init(self, __context: Any) -> None:
        prefix = self.doc_id.split(":", 1)[0]
        if prefix != self.source:
            raise ValueError(f"doc_id prefix {prefix!r} does not match source {self.source!r}")
