"""Parse the HackerOne top-N metadata CSV into `Document`s.

Strictly metadata-only: program, title, vuln_type, upvotes, bounty, link.
We never fetch or store the report bodies themselves (see fetch_hackerone_reports.py
module docstring for the ToS reasoning).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from app.schemas import Document

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "hackerone"

_REPORT_ID_RE = re.compile(r"/reports/(\d+)")


def _report_id(link: str | float | None) -> str | None:
    if not isinstance(link, str):
        return None
    m = _REPORT_ID_RE.search(link)
    return m.group(1) if m else None


def _normalise_link(link: str | float | None) -> str | None:
    if not isinstance(link, str) or not link:
        return None
    if link.startswith("http"):
        return link
    return "https://" + link.lstrip("/")


def _clean(val: object) -> str:
    if val is None:
        return ""
    if isinstance(val, float):
        if pd.isna(val):
            return ""
        if val.is_integer():
            return str(int(val))
        return f"{val:.2f}"
    return str(val).strip()


def _compose_text(row: pd.Series) -> str:
    program = _clean(row.get("program"))
    title = _clean(row.get("title"))
    vuln_type = _clean(row.get("vuln_type"))
    bounty = _clean(row.get("bounty"))
    upvotes = _clean(row.get("upvotes"))

    lines: list[str] = []
    header_parts: list[str] = []
    if vuln_type:
        header_parts.append(vuln_type)
    if program:
        header_parts.append(f"in {program}")
    if header_parts:
        lines.append(" ".join(header_parts) + ".")

    if title:
        lines.append(f"Report title: {title}")

    stats: list[str] = []
    if upvotes and upvotes != "0":
        stats.append(f"{upvotes} upvotes")
    if bounty and bounty not in ("", "0", "0.0", "0.00"):
        stats.append(f"${bounty} bounty")
    if stats:
        lines.append("Disclosed on HackerOne, " + ", ".join(stats) + ".")

    return " ".join(lines).strip()


def parse(limit: int = 500) -> list[Document]:
    csv_path = RAW / f"top{limit}_metadata.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found — run "
            f"`uv run python -m scripts.fetch_hackerone_reports --limit {limit}` first"
        )

    df = pd.read_csv(csv_path)
    docs: list[Document] = []
    seen_ids: set[str] = set()

    for _, row in df.iterrows():
        report_id = _report_id(row.get("link"))
        if report_id is None:
            continue
        doc_id = f"hackerone:{report_id}"
        if doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)

        title = _clean(row.get("title")) or f"HackerOne report #{report_id}"
        text = _compose_text(row)
        if not text:
            continue

        upvotes_raw = row.get("upvotes")
        bounty_raw = row.get("bounty")

        try:
            upvotes_val = int(upvotes_raw) if not pd.isna(upvotes_raw) else 0
        except (TypeError, ValueError):
            upvotes_val = 0
        try:
            bounty_val = float(bounty_raw) if not pd.isna(bounty_raw) else 0.0
        except (TypeError, ValueError):
            bounty_val = 0.0

        docs.append(
            Document(
                doc_id=doc_id,
                source="hackerone",
                url=_normalise_link(row.get("link")),
                title=title[:300],
                text=text,
                metadata={
                    "program": _clean(row.get("program")),
                    "vuln_type": _clean(row.get("vuln_type")),
                    "upvotes": upvotes_val,
                    "bounty_usd": bounty_val,
                },
            )
        )

    return docs


if __name__ == "__main__":
    docs = parse()
    print(f"[hackerone] {len(docs)} documents")
    if docs:
        print("  top 3 by upvotes:")
        by_up = sorted(docs, key=lambda d: int(d.metadata.get("upvotes", 0)), reverse=True)[:3]
        for d in by_up:
            print(f"    {d.doc_id}: {d.metadata.get('vuln_type')} ({d.metadata.get('upvotes')} up)")
