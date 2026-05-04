"""Dense embedder wrapper for GaRAG retrieval.

Runtime defaults to Ollama's OpenAI-compatible `/v1/embeddings`
endpoint with `andersc/qwen3-embedding:0.6b`. The previous
`BAAI/bge-m3` FlagEmbedding path remains available as an explicit
fallback for offline/local-model experiments.
"""

from __future__ import annotations

import importlib
from typing import Any, Self, cast

import httpx
import numpy as np

from app.config import settings

DEFAULT_MODEL = "andersc/qwen3-embedding:0.6b"
DEFAULT_FLAG_MODEL = "BAAI/bge-m3"
DEFAULT_DIM = 1024


class EmbeddingError(RuntimeError):
    """Raised when the embedding backend returns an unusable response."""


class DenseEmbedder:
    """Thin wrapper that returns numpy arrays of dense vectors."""

    def __init__(  # noqa: PLR0913
        self,
        model_name: str | None = None,
        *,
        provider: str | None = None,
        base_url: str | None = None,
        dim: int | None = None,
        use_fp16: bool = True,
        device: str = "cuda",
        timeout: float = 300.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.provider = provider or settings.embedding_provider
        self.dim = dim if dim is not None else settings.embedding_dim
        self.timeout = timeout
        self._client = client
        self._owns_client = client is None
        self._flag_model: Any | None = None

        if self.provider == "openai_compat":
            self.base_url = (base_url or settings.openai_embedding_base_url).rstrip("/")
            self.model_name = model_name or settings.openai_embedding_model
        elif self.provider == "flagembedding":
            self.base_url = None
            self.model_name = model_name or settings.bge_m3_model
            flag_embedding = importlib.import_module("FlagEmbedding")
            model_cls = flag_embedding.BGEM3FlagModel

            self._flag_model = model_cls(
                self.model_name,
                use_fp16=use_fp16,
                devices=[device],
            )
        else:
            msg = f"unsupported embedding provider: {self.provider}"
            raise ValueError(msg)

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> Self:
        if self.provider == "openai_compat":
            self._get_client()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        if self.provider == "openai_compat":
            return self._encode_openai_compat(texts)
        return self._encode_flagembedding(texts, batch_size=batch_size)

    def _encode_openai_compat(self, texts: list[str]) -> np.ndarray:
        client = self._get_client()
        try:
            response = client.post(
                f"{self.base_url}/embeddings",
                json={"model": self.model_name, "input": texts},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            msg = f"OpenAI-compatible /v1/embeddings failed: {exc}"
            raise EmbeddingError(msg) from exc

        try:
            payload = response.json()
            data = payload["data"]
        except (KeyError, TypeError, ValueError) as exc:
            msg = "OpenAI-compatible /v1/embeddings returned invalid JSON payload"
            raise EmbeddingError(msg) from exc

        if not isinstance(data, list):
            msg = "OpenAI-compatible /v1/embeddings payload has non-list data"
            raise EmbeddingError(msg)

        indexed: list[tuple[int, list[float]]] = []
        for position, item in enumerate(data):
            if not isinstance(item, dict):
                msg = "OpenAI-compatible /v1/embeddings data item is not an object"
                raise EmbeddingError(msg)
            item_dict = cast("dict[str, Any]", item)
            embedding = item_dict.get("embedding")
            if not isinstance(embedding, list):
                msg = "OpenAI-compatible /v1/embeddings data item has no embedding list"
                raise EmbeddingError(msg)
            index = item_dict.get("index", position)
            if not isinstance(index, int):
                msg = "OpenAI-compatible /v1/embeddings data item has non-integer index"
                raise EmbeddingError(msg)
            indexed.append((index, embedding))

        indexed.sort(key=lambda pair: pair[0])
        vecs = np.asarray([embedding for _, embedding in indexed], dtype=np.float32)
        self._validate_shape(vecs, expected_rows=len(texts))
        return vecs

    def _encode_flagembedding(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        if self._flag_model is None:
            msg = "FlagEmbedding model is not initialised"
            raise EmbeddingError(msg)
        out = self._flag_model.encode(
            texts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        # With return_dense=True and the other heads off, FlagEmbedding always
        # returns a numpy array under "dense_vecs" — but its declared return
        # type is a union, so we narrow it explicitly here.
        vecs = out["dense_vecs"]
        if not isinstance(vecs, np.ndarray):
            msg = f"expected ndarray from BGEM3FlagModel, got {type(vecs).__name__}"
            raise TypeError(msg)
        self._validate_shape(vecs, expected_rows=len(texts))
        return vecs.astype(np.float32, copy=False)

    def _validate_shape(self, vecs: np.ndarray, *, expected_rows: int) -> None:
        if vecs.ndim != 2:
            msg = f"unexpected embedding rank {vecs.ndim}, expected 2"
            raise EmbeddingError(msg)
        if vecs.shape[0] != expected_rows:
            msg = f"unexpected embedding row count {vecs.shape[0]}, expected {expected_rows}"
            raise EmbeddingError(msg)
        if vecs.shape[1] != self.dim:
            msg = f"unexpected embedding dim {vecs.shape[1]}, expected {self.dim}"
            raise EmbeddingError(msg)
