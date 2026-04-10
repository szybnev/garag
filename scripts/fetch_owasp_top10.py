"""Download OWASP Top 10 (2021, English) markdown files.

Ten real categories A01..A10 are fetched from the OWASP/Top10 repo on GitHub.
A00/A11 intro/next-steps files are skipped — they are not part of the Top 10.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx

BASE_URL = "https://raw.githubusercontent.com/OWASP/Top10/master/2021/docs/en"

FILES = [
    "A01_2021-Broken_Access_Control.md",
    "A02_2021-Cryptographic_Failures.md",
    "A03_2021-Injection.md",
    "A04_2021-Insecure_Design.md",
    "A05_2021-Security_Misconfiguration.md",
    "A06_2021-Vulnerable_and_Outdated_Components.md",
    "A07_2021-Identification_and_Authentication_Failures.md",
    "A08_2021-Software_and_Data_Integrity_Failures.md",
    "A09_2021-Security_Logging_and_Monitoring_Failures.md",
    "A10_2021-Server-Side_Request_Forgery_(SSRF).md",
]

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "owasp"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    fetched = 0
    skipped = 0
    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        for name in FILES:
            out_path = OUT_DIR / name
            if out_path.exists() and not args.force:
                skipped += 1
                continue

            url = f"{BASE_URL}/{name}"
            resp = client.get(url)
            if resp.status_code != 200:
                print(f"[warn] {resp.status_code} {url}")
                continue
            out_path.write_bytes(resp.content)
            total_bytes += len(resp.content)
            fetched += 1
            print(f"[fetch] {name} ({len(resp.content)} B)")

    print(f"[done] {fetched} fetched, {skipped} skipped, {total_bytes / 1000:.0f} KB total")


if __name__ == "__main__":
    main()
