"""Dense embedder wrapper around `BGEM3FlagModel`.

We expose only the `dense_vecs` head of bge-m3. Learned sparse and ColBERT
heads are explicitly disabled — they are PoxekBook E1 territory, not MVP.
"""

from __future__ import annotations

import numpy as np
from FlagEmbedding import BGEM3FlagModel

DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_DIM = 1024


class DenseEmbedder:
    """Thin wrapper that returns numpy arrays of normalised dense vectors."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        use_fp16: bool = True,
        device: str = "cuda",
    ) -> None:
        self.model_name = model_name
        self.dim = DEFAULT_DIM
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16, devices=[device])

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = self.model.encode(
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
        if vecs.shape[1] != self.dim:
            msg = f"unexpected embedding dim {vecs.shape[1]}, expected {self.dim}"
            raise RuntimeError(msg)
        return vecs
