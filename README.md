# GaRAG

**Hybrid RAG over a cybersecurity corpus** — MITRE ATT&CK + ATLAS, OWASP Top 10, public HackerOne reports, security tool man pages.

MVP for the **GigaSchool LLM-Engineer** final project (track A). Dedicated slice of the larger [PoxekBook](./docs/roadmap_to_poxekbook.md) vision.

> **Disclaimer.** GaRAG is an **academic project** built as the final assignment for the GigaSchool LLM-Engineer course. It is **not production-ready** and makes no claim of being so. Do not use it in real security operations, do not rely on its output for incident response or vulnerability assessment, and treat every generated answer as educational material that must be independently verified against the cited primary sources. Security scanning (`garak`) and guardrails (LLM Guard) are included as learning exercises, not as hardened defence in depth.

## Status

Work in progress — see `docs/plans/` (in parent `gigaschool/` repo) and the `bd` issue tracker for the current state.

Target release: **v0.1.0-garag** — 2026-04-24.

## Architecture (high level)

```
query
  │
  ▼
[input guardrails] → [dense (bge-m3) + sparse (BM25)] → [RRF fusion]
  │                                                           │
  ▼                                                           ▼
[reranker bge-reranker-v2-m3] ──► [generator qwen3.5:35b via Ollama, structured output]
                                                              │
                                                              ▼
                                                   [output guardrails]
                                                              │
                                                              ▼
                                             QueryResponse {answer, citations[], confidence, used_chunks[]}
```

Full design rationale — NFR table with targets, choice of `bge-m3`/`qwen3.5:35b`/`RecursiveChunker`, HackerOne metadata-only stance, and the explicit list of things GaRAG does not attempt — lives in [`docs/design.md`](./docs/design.md).

## Stack

- **Vector DB:** Qdrant (HNSW)
- **Embedding:** `BAAI/bge-m3` (dense)
- **Sparse:** `rank_bm25` (k1/b tuned per corpus)
- **Reranker:** `BAAI/bge-reranker-v2-m3` cross-encoder
- **LLM:** `qwen3.5:35b` via Ollama (`/api/chat`, `think=false`)
- **LLM-as-judge:** `qwen3.5:35b` (same model — self-bias caveat noted for d13)
- **API:** FastAPI + Pydantic structured output
- **UI:** Gradio
- **Orchestration:** Docker Compose (app + Qdrant + Prometheus + Grafana)
- **Security:** `garak` probes + LLM Guard guardrails
- **Python:** 3.12, uv + ruff + ty

## Quickstart

Prerequisites: Docker, Docker Compose, an existing `ollama` container on the host with `qwen3.5:35b` pulled.

```bash
# 1. Install dependencies
uv sync

# 2. Build the corpus and indices (one-off, ~20 min)
uv run python -m scripts.fetch_mitre_attack
uv run python -m scripts.fetch_mitre_atlas
uv run python -m scripts.fetch_owasp_top10
uv run python -m scripts.fetch_hackerone_reports --limit 500
uv run python -m scripts.fetch_man_pages
uv run python -m scripts.parse_sources
uv run python -m scripts.chunk_corpus
uv run python -m scripts.build_bm25

# 3. Start the stack
docker compose up -d qdrant
uv run python -m scripts.build_qdrant
docker compose up -d

# 4. Smoke test
curl -s http://localhost:8000/health | jq .
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is MITRE ATT&CK T1059.001?"}' | jq .
```

- **FastAPI:** `http://localhost:8000`
- **Gradio UI:** `http://localhost:7860`
- **Grafana:** `http://localhost:3001` (anonymous)
- **Prometheus:** `http://localhost:9091`

## Evaluation

```bash
uv run python -m scripts.eval_retrieval --golden data/golden/golden_set_v1.jsonl
uv run python -m scripts.eval_generation --golden data/golden/golden_set_v1.jsonl
uv run python -m scripts.nfr_benchmark --n 50 --concurrency 2
```

Reports land in `evaluation/reports/`. Current targets:

| Metric | Target |
|---|---|
| Recall@10 | ≥ 0.75 |
| nDCG@10 | ≥ 0.65 |
| LLM-judge faithfulness | ≥ 0.80 |
| LLM-judge correctness | ≥ 0.70 |
| Citation accuracy | ≥ 0.85 |
| p95 e2e latency (warm) | ≤ 8 s |
| Throughput | ≥ 2 RPS |
| Indexing time | ≤ 20 min |

## Security testing

```bash
bash security/garak/run_garak.sh    # ~15–30 min
```

Report summary in `security/garak/reports/summary.md`.

## Data disclaimer

The corpus is rebuilt from scratch by the scripts under `scripts/`. Raw scraped data is **not** shipped with this repository. HackerOne reports are pulled from open-source mirrors of public disclosures (e.g. `github.com/reddelexc/hackerone-reports`), capped at 500 top-disclosed reports.

## What's NOT in GaRAG (see `docs/roadmap_to_poxekbook.md`)

- Comparison of 3 embedding models (E1)
- 10-LLM benchmark (E6)
- GraphRAG, Agentic RAG
- Fine-tuning (QLoRA / DPO)
- Extended golden set (100–150 pairs)

These live in [PoxekBook](./docs/roadmap_to_poxekbook.md).

## License

[MIT](./LICENSE)
