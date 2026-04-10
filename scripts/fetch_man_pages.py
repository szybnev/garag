"""Fetch man-page-equivalent plain text for five security tools.

The five targets have different documentation strategies:

  - nmap, sqlmap, hydra  → Debian manpages (HTML → plain text)
  - nuclei, metasploit   → GitHub README.md (no formal man page upstream)

Each tool yields a single text file in `data/raw/man_pages/{tool}.txt`.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "man_pages"

DEBIAN_MAN_BASE = "https://manpages.debian.org/bookworm"

DEBIAN_TARGETS: list[tuple[str, str]] = [
    ("nmap", f"{DEBIAN_MAN_BASE}/nmap/nmap.1.en.html"),
    ("sqlmap", f"{DEBIAN_MAN_BASE}/sqlmap/sqlmap.1.en.html"),
    ("hydra", f"{DEBIAN_MAN_BASE}/hydra/hydra.1.en.html"),
]

GITHUB_TARGETS: list[tuple[str, str]] = [
    ("nuclei", "https://raw.githubusercontent.com/projectdiscovery/nuclei/dev/README.md"),
    (
        "metasploit",
        "https://raw.githubusercontent.com/rapid7/metasploit-framework/master/README.md",
    ),
]


def _clean_text(raw: str) -> str:
    """Collapse runs of whitespace while keeping paragraph breaks."""
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def _extract_debian_manpage(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # The actual man content sits in a <div id="manpage"> section; fall back to body if absent.
    container = soup.find("div", id="manpage") or soup.body
    if container is None:
        return ""
    for tag in container.find_all(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    text = container.get_text(separator="\n")
    return _clean_text(text)


def _fetch_debian(url: str, client: httpx.Client) -> str:
    resp = client.get(url)
    resp.raise_for_status()
    return _extract_debian_manpage(resp.text)


def _fetch_github_readme(url: str, client: httpx.Client) -> str:
    resp = client.get(url)
    resp.raise_for_status()
    return _clean_text(resp.text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": "garag-fetch/0.1"}
    with httpx.Client(follow_redirects=True, timeout=60.0, headers=headers) as client:
        for tool, url in DEBIAN_TARGETS:
            out = OUT_DIR / f"{tool}.txt"
            if out.exists() and not args.force:
                print(f"[skip] {tool} ({out.stat().st_size} B cached)")
                continue
            try:
                text = _fetch_debian(url, client)
            except Exception as exc:
                print(f"[error] {tool}: {exc}")
                continue
            if len(text) < 500:
                print(f"[warn] {tool}: only {len(text)} bytes extracted, skipping")
                continue
            out.write_text(text, encoding="utf-8")
            print(f"[debian-man] {tool}: {len(text)} B")

        for tool, url in GITHUB_TARGETS:
            out = OUT_DIR / f"{tool}.txt"
            if out.exists() and not args.force:
                print(f"[skip] {tool} ({out.stat().st_size} B cached)")
                continue
            try:
                text = _fetch_github_readme(url, client)
            except Exception as exc:
                print(f"[error] {tool}: {exc}")
                continue
            if len(text) < 500:
                print(f"[warn] {tool}: only {len(text)} bytes, skipping")
                continue
            out.write_text(text, encoding="utf-8")
            print(f"[github-readme] {tool}: {len(text)} B")

    files = sorted(OUT_DIR.glob("*.txt"))
    print(f"[done] {len(files)} files in {OUT_DIR}")


if __name__ == "__main__":
    main()
