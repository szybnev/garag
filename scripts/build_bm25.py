"""Build the sparse BM25 index for GaRAG.

Reads `data/processed/chunks.parquet`, tokenises each chunk's searchable text
with a small regex + nltk English stopwords, and pickles a `BM25Okapi`
together with the position → `chunk_id` mapping. d5
(`app/rag/retriever_sparse.py`) loads this pickle and uses it for sparse
retrieval.

The default `k1=1.5` and `b=0.75` come straight from `rank_bm25`. d6
(`scripts/tune_bm25.py`, bid `garag-zqc.15`) sweeps `k1`, `b`, and
optionally re-tokenises without stopwords as a single ablation.
"""

from __future__ import annotations

import argparse
import pickle
import re
import time
from pathlib import Path

import nltk
import pandas as pd
from rank_bm25 import BM25Okapi

CHUNKS_FILE = Path(__file__).resolve().parents[1] / "data" / "processed" / "chunks.parquet"
OUT_FILE = Path(__file__).resolve().parents[1] / "data" / "index" / "bm25.pkl"

_URL_RE = re.compile(r"https?://\S+")
_WORD_RE = re.compile(r"[a-z][a-z0-9_\-]{1,}")


def _ensure_stopwords() -> set[str]:
    try:
        from nltk.corpus import stopwords  # noqa: PLC0415

        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords  # noqa: PLC0415

        return set(stopwords.words("english"))


def tokenize(text: str, stop: set[str]) -> list[str]:
    text = _URL_RE.sub(" ", text.lower())
    return [t for t in _WORD_RE.findall(text) if t not in stop]


def searchable_text(row: pd.Series) -> str:
    """Join stable identifiers with chunk text for exact-ID sparse retrieval."""
    source = str(row.get("source", ""))
    return "\n".join(
        str(value)
        for value in (
            row.get("chunk_id", ""),
            row.get("doc_id", ""),
            source,
            source.replace("_", " "),
            row.get("title", ""),
            row.get("text", ""),
        )
        if pd.notna(value) and str(value).strip()
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=CHUNKS_FILE)
    ap.add_argument("--output", type=Path, default=OUT_FILE)
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    args = ap.parse_args()

    if not args.input.exists():
        msg = f"{args.input} not found — run `uv run python -m scripts.chunk_corpus` first"
        raise FileNotFoundError(msg)

    df = pd.read_parquet(args.input)
    print(f"[load] {len(df)} chunks from {args.input}")

    stop = _ensure_stopwords()
    print(f"[stopwords] nltk english, {len(stop)} terms")

    t0 = time.perf_counter()
    tokenized = [tokenize(searchable_text(row), stop) for _, row in df.iterrows()]
    print(f"[tokenize] {time.perf_counter() - t0:.1f}s")

    empty = sum(1 for tokens in tokenized if not tokens)
    if empty:
        print(f"[warn] {empty} chunks tokenized to empty list — adding placeholder token")
        tokenized = [tokens or ["__empty__"] for tokens in tokenized]

    t0 = time.perf_counter()
    bm25 = BM25Okapi(tokenized, k1=args.k1, b=args.b)
    print(f"[bm25] fit in {time.perf_counter() - t0:.1f}s, vocab={len(bm25.idf)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bm25": bm25,
        "chunk_ids": df["chunk_id"].tolist(),
        "params": {"k1": args.k1, "b": args.b, "stopwords": "nltk-english"},
        "vocab_size": len(bm25.idf),
        "n_docs": len(tokenized),
    }
    with args.output.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = args.output.stat().st_size / 1_000_000
    print(f"[done] wrote {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
