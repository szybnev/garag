"""Per-source parsers that convert raw dumps into `Document` lists.

Each submodule exposes a single `parse() -> list[Document]` entry point.
`scripts/parse_sources.py` is a thin orchestrator that calls all of them.
"""

from __future__ import annotations

from pathlib import Path

RAW_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw"
