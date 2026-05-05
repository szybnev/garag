# GaRAG

**Hybrid RAG over a cybersecurity corpus** — MITRE ATT&CK + ATLAS, OWASP Top 10, public HackerOne reports, security tool man pages.

MVP for the **GigaSchool LLM-Engineer** final project (track A).

> **Disclaimer.** GaRAG is an **academic project** built as the final assignment for the GigaSchool LLM-Engineer course. It is **not production-ready** and makes no claim of being so. Do not use it in real security operations, do not rely on its output for incident response or vulnerability assessment, and treat every generated answer as educational material that must be independently verified against the cited primary sources. Security scanning (`garak`) and Granite Guardian guardrails are included as learning exercises, not as hardened defence in depth.

<img width="1909" height="1086" alt="image" src="https://github.com/user-attachments/assets/6570abf4-1fb7-4bcc-8524-15919788edc9" />

## Status

Runtime MVP is implemented: hybrid retrieval, reranking, generation, FastAPI,
Gradio mounted under FastAPI, Prometheus metrics, Docker Compose wiring, and
evaluation/NFR benchmark scripts. The current local corpus has 2,544 documents,
3,900 chunks, and 3,900 Qdrant points. Security guardrails and a focused garak
runner are wired for local hardening checks.

Target release: **v0.1.0-garag** — 2026-04-24.

## Architecture (high level)

```
query
  │
  ▼
[dense (qwen3 embedding) + sparse (BM25)] → [alpha fusion, α=0.3]
                                      │
                                      ▼
                      [reranker bge-reranker-v2-m3]
                                      │
                                      ▼
        [generator zai-org/glm-4.7-flash via LM Studio]
                                      │
                                      ▼
        QueryResponse {answer, citations[], confidence, used_chunks[], latency_ms}
```

Full design rationale — NFR table with targets, choice of local models/`RecursiveChunker`, HackerOne metadata-only stance, and the explicit list of things GaRAG does not attempt — lives in [`docs/design.md`](./docs/design.md).

## Stack

- **Vector DB:** Qdrant (HNSW)
- **Embedding:** `text-embedding-qwen3-embedding-0.6b` via LM Studio OpenAI-compatible API
- **Sparse:** `rank_bm25` (k1/b tuned per corpus; searchable text includes
  `chunk_id`, `doc_id`, source, title, and chunk text)
- **Reranker:** `BAAI/bge-reranker-v2-m3` cross-encoder on GPU
- **LLM:** `zai-org/glm-4.7-flash` via LM Studio OpenAI-compatible API
- **LLM-as-judge:** `qwen3.5:35b` via Ollama fallback path
- **API:** FastAPI + Pydantic structured output (`/health`, `/query`, `/metrics`)
- **UI:** Gradio mounted at `/gradio`
- **Orchestration:** Docker Compose (app + Qdrant + Prometheus + Grafana)
- **Security:** Granite Guardian guardrails + focused `garak` probes
- **Python:** 3.12, uv + ruff + ty

## Generator model comparison

During local experiments, the generator was compared across
`qwen/qwen3.6-35b-a3b`, `ibm/granite-3.2-8b`, `ibm/granite-4-h-tiny`, and
`zai-org/glm-4.7-flash`. In the current run, GLM-4.7-Flash showed the best
practical results and is therefore the runtime default through LM Studio.

GLM-4.7-Flash was also tested through Docker vLLM on a single RTX 5090. The
working vLLM configuration required `bitsandbytes`, `fp8` KV cache, disabling
MLA with `VLLM_MLA_DISABLE=1`, and removing `--reasoning-parser glm45` so that
OpenAI-compatible responses populate `message.content`. Native BF16 did not fit
in 32 GB VRAM, online `fp8_per_tensor` and `fp8_per_block` still failed during
weight loading, and `--enforce-eager` was slower than the default CUDA graph
mode. Even after these fixes, structured JSON generation stayed around
15-16 tok/s, which was slower than the same target model in LM Studio. For this
MVP, LM Studio is therefore the more efficient local inference backend for the
target LLM; vLLM remains a serving experiment rather than the recommended
runtime path.

## Quickstart

Prerequisites: Docker, Docker Compose, and LM Studio serving
`zai-org/glm-4.7-flash` plus `text-embedding-qwen3-embedding-0.6b` on
`http://localhost:1234/v1`.

For the Docker app container, LM Studio must accept connections from the Docker
host gateway (`http://host.docker.internal:1234/v1`). If LM Studio is bound to
localhost only, local scripts work but `docker compose` runtime queries cannot
reach `/v1/chat/completions` or `/v1/embeddings`.

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
uv run python -m scripts.build_bm25 --k1 0.8 --b 0.5

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
- **Gradio UI:** `http://localhost:8000/gradio`
- **Prometheus metrics:** `http://localhost:8000/metrics`
- **Grafana:** `http://localhost:3001` (anonymous)
- **Prometheus:** `http://localhost:9091`

## Evaluation

```bash
uv run python -m scripts.eval_retrieval --golden data/golden/golden_set_v1.jsonl
uv run python -m scripts.eval_generation --golden data/golden/golden_set_v1.jsonl \
  --top-k 12 \
  --judge-provider openai_compat \
  --judge-base-url http://localhost:1234/v1 \
  --judge-model zai-org/glm-4.7-flash
uv run python -m scripts.nfr_benchmark --limit 20 --warmup 3 --top-k 12
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
| Throughput | ≥ 0.5 RPS |
| Indexing time | ≤ 20 min |

`scripts.nfr_benchmark` measures real HTTP `/query` latency and throughput by
default. Add `--run-indexing` only when you intentionally want to rebuild the
Qdrant collection and measure full indexing time. Add `--fail-on-target-miss`
when using the benchmark as a CI-style gate.

The throughput target is scoped to the single-user academic MVP with a local
LM Studio-hosted GLM generator and the NFR benchmark's `top_k=12` context budget.
The ≥2 RPS target is deferred to a serving-focused increment such as a smaller
generator or a different optimized serving stack.

Latest LM Studio snapshot on `golden_set_v1`:

| Metric | Value |
|---|---:|
| Generation faithfulness | 0.900 |
| Generation correctness | 0.920 |
| Citation accuracy | 1.000 |
| p95 e2e latency (warm, top_k=12) | 2.14 s |
| Throughput (top_k=12) | 0.826 RPS |

Latest retrieval snapshot on `golden_set_v1` after MITRE tactic enrichment and
BM25 identifier indexing:

| Method | Recall@10 | nDCG@10 | MAP |
|---|---:|---:|---:|
| dense (qwen3 embedding) | 0.7600 | 0.6721 | 0.6450 |
| sparse (BM25 k1=0.8, b=0.5) | 0.8800 | 0.8024 | 0.7779 |
| hybrid alpha=0.3 | 0.8800 | 0.7880 | 0.7575 |
| hybrid + reranker | 0.8600 | 0.8089 | 0.7933 |

## Security testing

`/query` runs input and output guardrails when `GUARDRAILS_ENABLED=true`.
The guardrail model is `granite-guardian-3.2-5b`, served through the same
LM Studio OpenAI-compatible API style as the generator. Unsafe user input is
rejected before retrieval/generation with HTTP 400; unsafe generated output or
guardrail backend failures are surfaced as HTTP 502. Because this academic MVP
serves benign MITRE/OWASP security education queries, input guardrails include a
narrow deterministic pass-through for educational or defensive cybersecurity
questions that do not request operational abuse. Groundedness checks still run
on generated answers, but default to audit-only (`GUARDRAILS_BLOCK_GROUNDEDNESS=false`)
to avoid blocking valid RAG answers on known Granite Guardian false positives.

Run a focused local red-team scan against a running FastAPI app:

```bash
make garak GARAG_API_URL=http://localhost:8000
```

The runner targets `POST /query`, extracts `answer` from the JSON response, and
writes garak reports under `security/garak/reports/` by default.

Latest local guarded audit, run on 2026-05-04 against `/query` with Granite
Guardian enabled:

| Probe / detector | Result |
|---|---:|
| `dan.Ablation_Dan_11_0` / `dan.DAN` | SKIP, 0/0 |
| `dan.Ablation_Dan_11_0` / `mitigation.MitigationBypass` | SKIP, 0/0 |
| `encoding.InjectBase64` / `encoding.DecodeMatch` | PASS, 5/5 |
| `encoding.InjectBase64` / `encoding.DecodeApprox` | FAIL, 4/5; ASR 20% |
| `promptinject.HijackLongPrompt` / `promptinject.AttackRogueString` | PASS, 6/6 |

The full local artifacts are `security/garak/reports/garag.report.html` and
`security/garak/reports/garag.report.jsonl` when the audit is run locally.

## Data disclaimer

The corpus is rebuilt from scratch by the scripts under `scripts/`. Raw scraped data is **not** shipped with this repository. HackerOne reports are pulled from open-source mirrors of public disclosures (e.g. `github.com/reddelexc/hackerone-reports`), capped at 500 top-disclosed reports.

Current corpus composition:

| Source | Documents | Chunks |
|---|---:|---:|
| MITRE ATT&CK Enterprise | 1,751 | 2,562 |
| MITRE ATLAS | 278 | 458 |
| HackerOne public reports metadata | 500 | 500 |
| OWASP Top 10 (2021, EN) | 10 | 110 |
| Security tool man pages | 5 | 270 |

MITRE ATT&CK technique and sub-technique documents are fully loaded from the
active Enterprise STIX bundle. Tactic documents, such as `TA0010 Exfiltration`,
are enriched with the current list of related techniques from
`kill_chain_phases`, so tactic-level questions can retrieve the category page
and the technique list. Procedure examples from ATT&CK `relationship` objects
are still not indexed in the MVP.

## What's NOT in GaRAG (see `docs/roadmap.md`)

- Comparison of 3 embedding models (E1)
- 10-LLM benchmark (E6)
- GraphRAG, Agentic RAG
- Fine-tuning (QLoRA / DPO)
- Extended golden set (100–150 pairs)
- Production-grade security hardening beyond the academic guardrail/garak checks

These are tracked as out-of-scope roadmap items in [`docs/roadmap.md`](./docs/roadmap.md).

## License

[MIT](./LICENSE)
