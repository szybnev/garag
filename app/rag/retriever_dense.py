"""Dense retriever — configured DenseEmbedder + Qdrant `garag_v1`."""

from __future__ import annotations

from qdrant_client import QdrantClient

from app.rag import ScoredChunk
from app.rag.embedder import DenseEmbedder

DEFAULT_QDRANT_URL = "http://localhost:6380"
DEFAULT_COLLECTION = "garag_v1"


class DenseRetriever:
    def __init__(
        self,
        embedder: DenseEmbedder | None = None,
        *,
        qdrant_url: str = DEFAULT_QDRANT_URL,
        collection: str = DEFAULT_COLLECTION,
    ) -> None:
        self.embedder = embedder or DenseEmbedder()
        self.client = QdrantClient(url=qdrant_url, timeout=60)
        self.collection = collection

    def search(self, query: str, top_k: int = 20) -> list[ScoredChunk]:
        qv = self.embedder.encode([query], batch_size=1)[0].astype("float32").tolist()
        response = self.client.query_points(
            collection_name=self.collection,
            query=qv,
            limit=top_k,
            with_payload=True,
        )
        out: list[ScoredChunk] = []
        for hit in response.points:
            payload = hit.payload or {}
            out.append(
                ScoredChunk(
                    chunk_id=str(payload.get("chunk_id", hit.id)),
                    score=float(hit.score),
                    source=str(payload.get("source", "")),
                    title=str(payload.get("title", "")),
                    text=str(payload.get("text", "")),
                    url=payload.get("url"),
                    doc_id=str(payload.get("doc_id", "")),
                )
            )
        return out
