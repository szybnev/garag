"""Parse OWASP Top 10 (2021) markdown files into `Document`s.

Each category becomes a single Document — the markdown is already small
(<12 KB) and its sections stay together for better context during retrieval.
CWE IDs mentioned in the text are extracted into metadata for faceted filtering.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.schemas import Document

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "owasp"

CWE_RE = re.compile(r"CWE-(\d+)")
H1_RE = re.compile(r"^#\s+(.+?)(?:\s+!\[.*)?$", re.MULTILINE)

# Map from file stem to canonical URL slug on owasp.org
CANONICAL_URLS = {
    "A01_2021-Broken_Access_Control": ("https://owasp.org/Top10/A01_2021-Broken_Access_Control/"),
    "A02_2021-Cryptographic_Failures": ("https://owasp.org/Top10/A02_2021-Cryptographic_Failures/"),
    "A03_2021-Injection": "https://owasp.org/Top10/A03_2021-Injection/",
    "A04_2021-Insecure_Design": "https://owasp.org/Top10/A04_2021-Insecure_Design/",
    "A05_2021-Security_Misconfiguration": (
        "https://owasp.org/Top10/A05_2021-Security_Misconfiguration/"
    ),
    "A06_2021-Vulnerable_and_Outdated_Components": (
        "https://owasp.org/Top10/A06_2021-Vulnerable_and_Outdated_Components/"
    ),
    "A07_2021-Identification_and_Authentication_Failures": (
        "https://owasp.org/Top10/A07_2021-Identification_and_Authentication_Failures/"
    ),
    "A08_2021-Software_and_Data_Integrity_Failures": (
        "https://owasp.org/Top10/A08_2021-Software_and_Data_Integrity_Failures/"
    ),
    "A09_2021-Security_Logging_and_Monitoring_Failures": (
        "https://owasp.org/Top10/A09_2021-Security_Logging_and_Monitoring_Failures/"
    ),
    "A10_2021-Server-Side_Request_Forgery_(SSRF)": (
        "https://owasp.org/Top10/A10_2021-Server-Side_Request_Forgery_%28SSRF%29/"
    ),
}


def _strip_image_suffix(title: str) -> str:
    """Remove inline `![icon](...)` blocks that trail the H1 line."""
    return re.sub(r"\s*!\[.*", "", title).strip()


def _extract_title(md: str, fallback: str) -> str:
    m = H1_RE.search(md)
    if m:
        return _strip_image_suffix(m.group(1))
    return fallback


def parse() -> list[Document]:
    if not RAW.exists():
        raise FileNotFoundError(
            f"{RAW} not found — run `uv run python -m scripts.fetch_owasp_top10` first"
        )

    files = sorted(RAW.glob("A*_2021-*.md"))
    if not files:
        raise FileNotFoundError(f"no OWASP markdown files in {RAW}")

    docs: list[Document] = []
    for path in files:
        stem = path.stem  # e.g. "A01_2021-Broken_Access_Control"
        md = path.read_text(encoding="utf-8")
        text = md.strip()
        title = _extract_title(md, fallback=stem)
        cwes = sorted({int(m) for m in CWE_RE.findall(md)})
        category = stem.split("_", 1)[0]  # "A01"
        url = CANONICAL_URLS.get(stem)

        docs.append(
            Document(
                doc_id=f"owasp:{stem}",
                source="owasp",
                url=url,
                title=title,
                text=text,
                metadata={
                    "category": category,
                    "cwes": cwes,
                    "cwe_count": len(cwes),
                },
            )
        )

    return docs


if __name__ == "__main__":
    docs = parse()
    print(f"[owasp] {len(docs)} documents")
    for d in docs:
        cwes = d.metadata.get("cwes", [])
        print(f"  {d.doc_id}: {d.title[:50]} (CWEs: {len(cwes)})")
