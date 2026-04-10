"""Parse the fetched man-page-equivalent text files into `Document`s.

Each tool yields a single Document. No section splitting here — the chunker
on d3 will take care of that. We do keep a canonical-looking URL to the
Debian manpages (or GitHub README for nuclei/metasploit).
"""

from __future__ import annotations

from pathlib import Path

from app.schemas import Document

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "man_pages"

TOOL_URLS = {
    "nmap": "https://manpages.debian.org/bookworm/nmap/nmap.1.en.html",
    "sqlmap": "https://manpages.debian.org/bookworm/sqlmap/sqlmap.1.en.html",
    "hydra": "https://manpages.debian.org/bookworm/hydra/hydra.1.en.html",
    "nuclei": "https://github.com/projectdiscovery/nuclei",
    "metasploit": "https://github.com/rapid7/metasploit-framework",
}

TOOL_TITLES = {
    "nmap": "nmap — Network exploration tool and security / port scanner",
    "sqlmap": "sqlmap — automatic SQL injection and database takeover tool",
    "hydra": "hydra — a very fast network logon cracker",
    "nuclei": "nuclei — fast, template-based vulnerability scanner",
    "metasploit": "metasploit — the world's most used penetration testing framework",
}


def parse() -> list[Document]:
    if not RAW.exists():
        raise FileNotFoundError(
            f"{RAW} not found — run `uv run python -m scripts.fetch_man_pages` first"
        )

    docs: list[Document] = []
    for path in sorted(RAW.glob("*.txt")):
        tool = path.stem
        if tool not in TOOL_URLS:
            continue
        text = path.read_text(encoding="utf-8").strip()
        if len(text) < 500:
            continue
        docs.append(
            Document(
                doc_id=f"man_pages:{tool}",
                source="man_pages",
                url=TOOL_URLS[tool],
                title=TOOL_TITLES[tool],
                text=text,
                metadata={
                    "tool_name": tool,
                    "text_bytes": len(text),
                    "source_type": "debian_man"
                    if tool in ("nmap", "sqlmap", "hydra")
                    else "github_readme",
                },
            )
        )
    return docs


if __name__ == "__main__":
    docs = parse()
    print(f"[man_pages] {len(docs)} documents")
    for d in docs:
        print(f"  {d.doc_id}: {d.metadata.get('text_bytes')} B ({d.metadata.get('source_type')})")
