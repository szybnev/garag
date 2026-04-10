"""Download the HackerOne disclosed reports metadata index.

We intentionally pull only the master `data.csv` from the open-source mirror
`reddelexc/hackerone-reports`, never the individual report bodies. The bodies
belong to HackerOne and the reporting researchers; we respect their ToS by
indexing metadata only (title, program, vuln_type, bounty, upvotes, link).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/reddelexc/hackerone-reports/master/data.csv"

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "hackerone"
RAW_CSV = OUT_DIR / "data_full.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=500, help="top N by upvotes")
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_CSV.exists() or args.force:
        print(f"[fetch] {DATA_URL}")
        resp = httpx.get(DATA_URL, follow_redirects=True, timeout=120.0)
        resp.raise_for_status()
        RAW_CSV.write_bytes(resp.content)
        print(f"[done] wrote {RAW_CSV} ({len(resp.content) / 1000:.0f} KB)")
    else:
        print(f"[cached] {RAW_CSV}")

    df = pd.read_csv(RAW_CSV, on_bad_lines="skip", engine="python")
    print(f"[loaded] {len(df)} rows, columns: {list(df.columns)}")

    if "upvotes" not in df.columns:
        raise RuntimeError(f"'upvotes' column missing; got {list(df.columns)}")

    df["upvotes"] = pd.to_numeric(df["upvotes"], errors="coerce").fillna(0).astype(int)
    if "bounty" in df.columns:
        df["bounty"] = pd.to_numeric(df["bounty"], errors="coerce").fillna(0.0)

    top = df.sort_values("upvotes", ascending=False).head(args.limit).reset_index(drop=True)
    out = OUT_DIR / f"top{args.limit}_metadata.csv"
    top.to_csv(out, index=False)
    print(
        f"[top-{args.limit}] wrote {out} (upvotes range: "
        f"{top['upvotes'].min()}..{top['upvotes'].max()})"
    )


if __name__ == "__main__":
    main()
