"""Orchestrate all per-source parsers into a single `documents.parquet`.

Calls each `scripts.parsers.{source}.parse()` in turn, validates every
`Document`, dedupes by `doc_id`, and writes a single parquet file that
downstream d3 chunking consumes. `metadata` is serialised as JSON strings
because parquet cannot store arbitrary `dict` columns losslessly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from app.schemas import Document
from scripts.parsers import hackerone as h1
from scripts.parsers import man_pages, mitre_atlas, mitre_attack, owasp

OUT_FILE = Path(__file__).resolve().parents[1] / "data" / "raw" / "documents.parquet"


def _all_documents() -> list[Document]:
    docs: list[Document] = []
    print("[parse] mitre_attack ...", end=" ", flush=True)
    dd = mitre_attack.parse()
    print(f"{len(dd)}")
    docs.extend(dd)

    print("[parse] mitre_atlas ...", end=" ", flush=True)
    dd = mitre_atlas.parse()
    print(f"{len(dd)}")
    docs.extend(dd)

    print("[parse] owasp ...", end=" ", flush=True)
    dd = owasp.parse()
    print(f"{len(dd)}")
    docs.extend(dd)

    print("[parse] hackerone ...", end=" ", flush=True)
    dd = h1.parse()
    print(f"{len(dd)}")
    docs.extend(dd)

    print("[parse] man_pages ...", end=" ", flush=True)
    dd = man_pages.parse()
    print(f"{len(dd)}")
    docs.extend(dd)

    return docs


def _dedupe(docs: list[Document]) -> list[Document]:
    seen: set[str] = set()
    result: list[Document] = []
    for d in docs:
        if d.doc_id in seen:
            continue
        seen.add(d.doc_id)
        result.append(d)
    return result


def _to_dataframe(docs: list[Document]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "doc_id": [d.doc_id for d in docs],
            "source": [d.source for d in docs],
            "url": [d.url for d in docs],
            "title": [d.title for d in docs],
            "text": [d.text for d in docs],
            "metadata": [json.dumps(d.metadata, ensure_ascii=False) for d in docs],
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_FILE)
    args = ap.parse_args()

    docs = _all_documents()
    print(f"[total] {len(docs)} before dedupe")
    docs = _dedupe(docs)
    print(f"[total] {len(docs)} after dedupe")

    df = _to_dataframe(docs)
    print("[by source]")
    for src, cnt in df.groupby("source").size().items():
        print(f"  {src}: {cnt}")
    print(
        f"[stats] text length: min={df['text'].str.len().min()} "
        f"mean={int(df['text'].str.len().mean())} "
        f"max={df['text'].str.len().max()}"
    )
    print(f"[stats] urls present: {df['url'].notna().sum()}/{len(df)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, engine="pyarrow", compression="zstd", index=False)
    size_kb = args.out.stat().st_size / 1000
    print(f"[done] wrote {args.out} ({size_kb:.0f} KB, {len(df)} rows)")


if __name__ == "__main__":
    main()
