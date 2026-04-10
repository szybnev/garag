"""Download the MITRE ATLAS compiled YAML dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx

ATLAS_URL = "https://raw.githubusercontent.com/mitre-atlas/atlas-data/main/dist/ATLAS.yaml"

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "mitre_atlas"
OUT_FILE = OUT_DIR / "ATLAS.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="overwrite existing file")
    args = ap.parse_args()

    if OUT_FILE.exists() and not args.force:
        size_kb = OUT_FILE.stat().st_size / 1_000
        print(f"[skip] {OUT_FILE} already exists ({size_kb:.0f} KB). Use --force to overwrite.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] {ATLAS_URL}")
    resp = httpx.get(ATLAS_URL, follow_redirects=True, timeout=60.0)
    resp.raise_for_status()
    OUT_FILE.write_bytes(resp.content)
    kb = len(resp.content) / 1_000
    print(f"[done] wrote {OUT_FILE} ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
