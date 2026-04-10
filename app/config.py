"""Runtime configuration loaded from env / `.env` via `pydantic-settings`.

Default values match `.env.example`. Optimal `bm25_k1`, `bm25_b`, and
`fusion_alpha` are filled in on d6 by `scripts/tune_bm25.py` and
`scripts/tune_fusion.py` and committed to source so the runtime defaults
match the evaluated configuration.
"""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Qdrant
    qdrant_url: str = "http://localhost:6380"
    qdrant_collection: str = "garag_v1"

    # Ollama (external container on the host)
    ollama_url: str = "http://host.docker.internal:11434"
    ollama_model: str = "qwen3.5:35b"
    ollama_judge_model: str = "qwen3.5:35b"
    ollama_keep_alive: str = "30s"

    # Embedding / reranker
    bge_m3_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # BM25 — tuned via `scripts/tune_bm25.py` on golden_set_v1
    # (k1=0.8, b=0.5 → nDCG@10=0.7844, +2.7 п.п. над дефолтами 1.5/0.75)
    bm25_k1: float = 0.8
    bm25_b: float = 0.5

    # Hybrid fusion — tuned via `scripts/tune_fusion.py` after BM25 tuning
    # (alpha=0.3 → nDCG@10=0.7890, slightly above RRF k=60 = 0.7783)
    fusion_method: Literal["rrf", "alpha"] = "alpha"
    fusion_alpha: float = 0.3
    rrf_k: int = 60

    # Retrieval
    retrieve_top_k: int = 20
    rerank_top_k: int = 5

    # Generation — tuned via `scripts/tune_gen_params.py` on 20 golden queries
    # (36-config grid; winner = first fmt=1.0 config after rejecting num_predict=400
    #  due to tool_usage truncation; see docs/design.md §4.x for the full table)
    gen_temperature: float = 0.4
    gen_top_p: float = 1.0
    gen_num_predict: int = 800
    gen_seed: int = 42

    # Observability
    prometheus_enabled: bool = True
    log_level: str = "INFO"

    # Guardrails
    guardrails_enabled: bool = True


settings = Settings()
