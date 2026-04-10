"""Build the dense Qdrant index for GaRAG.

Reads `data/processed/chunks.parquet`, encodes each chunk with bge-m3
through `app.rag.embedder.DenseEmbedder`, and upserts into a fresh
Qdrant collection `garag_v1` with the HNSW parameters fixed by
`docs/design.md §4.7`.

Idempotent on the collection — by default it deletes and recreates,
because re-running with stale points would mix old vectors with new.
Pass `--no-recreate` to append instead.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, HnswConfigDiff, PointStruct, VectorParams

from app.rag.embedder import DEFAULT_DIM, DenseEmbedder

CHUNKS_FILE = Path(__file__).resolve().parents[1] / "data" / "processed" / "chunks.parquet"

DEFAULT_URL = "http://localhost:6380"
DEFAULT_COLLECTION = "garag_v1"
DEFAULT_BATCH = 64


def _create_collection(client: QdrantClient, name: str, *, recreate: bool) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        if recreate:
            print(f"[collection] {name} exists, deleting")
            client.delete_collection(name)
        else:
            print(f"[collection] {name} exists, appending")
            return

    print(f"[collection] creating {name} (size={DEFAULT_DIM}, distance=COSINE, HNSW m=16 ef_c=200)")
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=DEFAULT_DIM, distance=Distance.COSINE),
        hnsw_config=HnswConfigDiff(
            m=16,
            ef_construct=200,
            full_scan_threshold=10000,
        ),
    )


def _payload(row: pd.Series) -> dict[str, object]:
    return {
        "chunk_id": row["chunk_id"],
        "doc_id": row["doc_id"],
        "source": row["source"],
        "url": row["url"] if pd.notna(row["url"]) else None,
        "title": row["title"],
        "text": row["text"],
        "token_count": int(row["token_count"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=CHUNKS_FILE)
    ap.add_argument("--qdrant-url", default=DEFAULT_URL)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ap.add_argument(
        "--no-recreate",
        dest="recreate",
        action="store_false",
        default=True,
        help="append instead of dropping the collection",
    )
    args = ap.parse_args()

    if not args.input.exists():
        msg = f"{args.input} not found — run `uv run python -m scripts.chunk_corpus` first"
        raise FileNotFoundError(msg)

    df = pd.read_parquet(args.input)
    print(f"[load] {len(df)} chunks from {args.input}")

    client = QdrantClient(url=args.qdrant_url, timeout=60)
    _create_collection(client, args.collection, recreate=args.recreate)

    embedder = DenseEmbedder()
    print(f"[embedder] {embedder.model_name}, dim={embedder.dim}, batch={args.batch_size}")

    encode_seconds = 0.0
    upsert_seconds = 0.0

    n = len(df)
    for start in range(0, n, args.batch_size):
        end = min(start + args.batch_size, n)
        batch = df.iloc[start:end]
        texts = batch["text"].tolist()

        t0 = time.perf_counter()
        vecs = embedder.encode(texts, batch_size=args.batch_size)
        encode_seconds += time.perf_counter() - t0

        if vecs.shape[1] != DEFAULT_DIM:
            msg = f"unexpected embedding dim {vecs.shape[1]}, expected {DEFAULT_DIM}"
            raise RuntimeError(msg)

        points = [
            PointStruct(
                id=int(start + i),
                vector=vecs[i].astype("float32").tolist(),
                payload=_payload(row),
            )
            for i, (_, row) in enumerate(batch.iterrows())
        ]

        t0 = time.perf_counter()
        client.upsert(collection_name=args.collection, points=points, wait=True)
        upsert_seconds += time.perf_counter() - t0

        if (end // args.batch_size) % 10 == 0 or end == n:
            print(f"  [{end:>5}/{n}] encode={encode_seconds:.1f}s upsert={upsert_seconds:.1f}s")

    info = client.get_collection(args.collection)
    print("[done] collection state:")
    print(f"  points_count: {info.points_count}")
    print(
        f"  vectors:      size={info.config.params.vectors.size}, "
        f"distance={info.config.params.vectors.distance}"
    )
    print(
        f"  hnsw:         m={info.config.hnsw_config.m}, "
        f"ef_construct={info.config.hnsw_config.ef_construct}"
    )
    print(
        f"[time] encode={encode_seconds:.1f}s upsert={upsert_seconds:.1f}s "
        f"total={(encode_seconds + upsert_seconds):.1f}s"
    )


if __name__ == "__main__":
    main()
