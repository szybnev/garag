"""Download the MITRE ATT&CK Enterprise STIX bundle.

By default uses the modern STIX 2.1 repo (`mitre-attack/attack-stix-data`).
Pass `--stix2` to fall back to the legacy STIX 2.0 `mitre/cti` mirror.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx

STIX_2_1_URL = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/"
    "master/enterprise-attack/enterprise-attack.json"
)
STIX_2_0_URL = (
    "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
)

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "mitre_attack"
OUT_FILE = OUT_DIR / "enterprise-attack.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stix2", action="store_true", help="use legacy STIX 2.0 mirror")
    ap.add_argument("--force", action="store_true", help="overwrite existing file")
    args = ap.parse_args()

    if OUT_FILE.exists() and not args.force:
        size_mb = OUT_FILE.stat().st_size / 1_000_000
        print(f"[skip] {OUT_FILE} already exists ({size_mb:.1f} MB). Use --force to overwrite.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    url = STIX_2_0_URL if args.stix2 else STIX_2_1_URL
    print(f"[fetch] {url}")
    with httpx.stream("GET", url, follow_redirects=True, timeout=300.0) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        written = 0
        with OUT_FILE.open("wb") as fh:
            for chunk in resp.iter_bytes(chunk_size=65536):
                fh.write(chunk)
                written += len(chunk)
        mb = written / 1_000_000
        if total:
            total_mb = total / 1_000_000
            print(f"[done] wrote {OUT_FILE} ({mb:.1f} / {total_mb:.1f} MB)")
        else:
            print(f"[done] wrote {OUT_FILE} ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
