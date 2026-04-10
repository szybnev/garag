"""Split `data/raw/documents.parquet` into token-bounded chunks.

Uses `chonkie.RecursiveChunker` with the GPT-2 tokenizer and `chunk_size=256`
tokens. The recursive split falls back through paragraph → sentence →
punctuation → whitespace → character, so most chunks land near the token
budget without breaking semantic units mid-sentence.

Output: `data/processed/chunks.parquet` with one row per `Chunk`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from chonkie import RecursiveChunker

from app.schemas import Chunk

IN_FILE = Path(__file__).resolve().parents[1] / "data" / "raw" / "documents.parquet"
OUT_FILE = Path(__file__).resolve().parents[1] / "data" / "processed" / "chunks.parquet"


def _chunk_one(
    chunker: RecursiveChunker,
    row: pd.Series,
) -> list[Chunk]:
    text = row["text"]
    doc_id = row["doc_id"]
    source = row["source"]
    url = row["url"] if pd.notna(row["url"]) else None
    title = row["title"]

    raw_chunks = chunker.chunk(text)
    out: list[Chunk] = []
    for idx, rc in enumerate(raw_chunks):
        if not rc.text.strip():
            continue
        out.append(
            Chunk(
                chunk_id=f"{doc_id}::{idx}",
                doc_id=doc_id,
                source=source,
                url=url,
                title=title,
                text=rc.text,
                char_start=int(rc.start_index),
                char_end=int(rc.end_index),
                token_count=int(rc.token_count),
            )
        )
    return out


def _to_dataframe(chunks: list[Chunk]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "chunk_id": [c.chunk_id for c in chunks],
            "doc_id": [c.doc_id for c in chunks],
            "source": [c.source for c in chunks],
            "url": [c.url for c in chunks],
            "title": [c.title for c in chunks],
            "text": [c.text for c in chunks],
            "char_start": [c.char_start for c in chunks],
            "char_end": [c.char_end for c in chunks],
            "token_count": [c.token_count for c in chunks],
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunk-size", type=int, default=256, help="target tokens per chunk")
    ap.add_argument("--tokenizer", default="gpt2", help="chonkie tokenizer name")
    ap.add_argument("--input", type=Path, default=IN_FILE)
    ap.add_argument("--output", type=Path, default=OUT_FILE)
    args = ap.parse_args()

    if not args.input.exists():
        msg = f"{args.input} not found — run `uv run python -m scripts.parse_sources` first"
        raise FileNotFoundError(msg)

    df = pd.read_parquet(args.input)
    print(f"[load] {len(df)} documents from {args.input}")

    chunker = RecursiveChunker(tokenizer=args.tokenizer, chunk_size=args.chunk_size)
    print(f"[chunker] RecursiveChunker(tokenizer={args.tokenizer!r}, chunk_size={args.chunk_size})")

    t0 = time.perf_counter()
    all_chunks: list[Chunk] = []
    per_source: dict[str, int] = {}
    for _, row in df.iterrows():
        chunks = _chunk_one(chunker, row)
        all_chunks.extend(chunks)
        per_source[row["source"]] = per_source.get(row["source"], 0) + len(chunks)

    elapsed = time.perf_counter() - t0
    print(
        f"[chunked] {len(all_chunks)} chunks in {elapsed:.1f}s "
        f"({len(all_chunks) / max(elapsed, 1e-6):.0f} chunks/s)"
    )

    print("[by source]")
    for src in sorted(per_source):
        n_docs = int((df["source"] == src).sum())
        n_chunks = per_source[src]
        fanout = n_chunks / max(n_docs, 1)
        print(f"  {src}: {n_docs} docs -> {n_chunks} chunks (x{fanout:.1f})")

    out_df = _to_dataframe(all_chunks)
    print(
        f"[stats] token_count: min={out_df['token_count'].min()} "
        f"mean={int(out_df['token_count'].mean())} "
        f"max={out_df['token_count'].max()}"
    )
    print(
        f"[stats] chunks/doc: "
        f"min={out_df.groupby('doc_id').size().min()} "
        f"max={out_df.groupby('doc_id').size().max()}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output, engine="pyarrow", compression="zstd", index=False)
    size_mb = args.output.stat().st_size / 1_000_000
    print(f"[done] wrote {args.output} ({size_mb:.1f} MB, {len(out_df)} rows)")

    # write a quick config sidecar for reproducibility
    config_path = args.output.with_suffix(".config.json")
    config_path.write_text(
        json.dumps(
            {
                "chunker": "RecursiveChunker",
                "tokenizer": args.tokenizer,
                "chunk_size": args.chunk_size,
                "source_file": str(args.input.name),
                "documents_in": len(df),
                "chunks_out": len(out_df),
                "per_source": dict(per_source),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
