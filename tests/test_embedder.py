"""Mock-based tests for `DenseEmbedder`.

The unit suite exercises the OpenAI-compatible LM Studio embedding path
without requiring a running local model server.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import numpy as np
import pytest

from app.rag.embedder import DenseEmbedder, EmbeddingError


def test_openai_compat_embedding_payload_and_response() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(
            {
                "url": str(request.url),
                "payload": json.loads(request.content),
            }
        )
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 1, "embedding": [0.4, 0.5, 0.6]},
                    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
                ],
                "model": "text-embedding-qwen3-embedding-0.6b",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    embedder = DenseEmbedder(
        provider="openai_compat",
        base_url="http://lmstudio.test:1234/v1",
        model_name="text-embedding-qwen3-embedding-0.6b",
        dim=3,
        client=client,
    )

    vecs = embedder.encode(["first", "second"])

    assert requests == [
        {
            "url": "http://lmstudio.test:1234/v1/embeddings",
            "payload": {
                "model": "text-embedding-qwen3-embedding-0.6b",
                "input": ["first", "second"],
            },
        }
    ]
    assert vecs.dtype == np.float32
    np.testing.assert_allclose(vecs, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])


def test_openai_compat_embedding_empty_input_skips_backend() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("backend should not be called for empty input")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    embedder = DenseEmbedder(
        provider="openai_compat",
        base_url="http://lmstudio.test:1234/v1",
        model_name="text-embedding-qwen3-embedding-0.6b",
        dim=3,
        client=client,
    )

    vecs = embedder.encode([])

    assert vecs.shape == (0, 3)
    assert vecs.dtype == np.float32


def test_openai_compat_embedding_rejects_wrong_dimension() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    embedder = DenseEmbedder(
        provider="openai_compat",
        base_url="http://lmstudio.test:1234/v1",
        model_name="text-embedding-qwen3-embedding-0.6b",
        dim=3,
        client=client,
    )

    with pytest.raises(EmbeddingError, match="unexpected embedding dim 2, expected 3"):
        embedder.encode(["first"])


def test_embedder_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="unsupported embedding provider"):
        DenseEmbedder(provider="unknown", dim=3)
