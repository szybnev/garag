# Roadmap: GaRAG → PoxekBook

**GaRAG** is the MVP slice of the larger **PoxekBook** vision. It closes the mandatory scoring blocks of the GigaSchool LLM-Engineer final project and ships as `v0.1.0-garag`. Everything that is *not* strictly required for that score lives in this roadmap.

What follows is the delta between GaRAG v0.1.0 and PoxekBook v1.0.

## What is already in GaRAG v0.1.0

- Single hybrid RAG pipeline: dense (`bge-m3`) + sparse (`rank_bm25`) → RRF → `bge-reranker-v2-m3` → `qwen3.5:14b`
- Single chunking strategy (`RecursiveChunker 256 gpt2`) with theoretical justification
- Single LLM for generation, single LLM (`qwen3.5:35b`) as judge
- 50-item golden set, categories: factual, tool usage, multi-hop
- FastAPI + Gradio + Docker Compose + Prometheus + Grafana
- `garak` security probes + LLM Guard guardrails
- Structured output `{answer, citations[], confidence, used_chunks[]}`

## PoxekBook increments

### Increment 2 — experimental depth (week 3)

- **E1: embedding comparison** — `bge-m3` vs `nomic-embed-text-v2-moe` vs `snowflake-arctic-embed2` on the golden set. Cross-lingual RU→EN retrieval eval.
- **E2: chunking comparison** — fixed vs recursive vs semantic vs entity-aware. Per-source winners.
- **E3: sparse retrieval** — compare `rank_bm25` (current) vs `bge-m3 learned sparse` vs tuned BM25 with cybersec stopwords.
- **E4: fusion methods** — RRF vs MMR vs alpha-weighted, grid over `alpha ∈ [0, 1]`.
- **E5: reranker on/off** — quantitative trade-off between quality and latency.
- **E6: LLM benchmark** — 4–5 candidate LLMs (3B–35B), 3-stage filter (censorship → TPS → quality). Pareto-optimal choice.
- **Extended golden set** — grow from 50 to 100–150 pairs.

### Increment 3 — advanced RAG (week 4)

- **GraphRAG** — NetworkX in-memory graph over MITRE ATT&CK (~2 000 nodes). Multi-hop queries: technique → group → mitigation.
- **Agentic RAG** — `smolagents` `ToolCallingAgent` with `retrieve_dense`, `retrieve_sparse`, `rerank`, `graph_lookup`, `answer` tools.
- **E7: RAG comparison** — Naive vs Hybrid (GaRAG) vs GraphRAG vs Agentic RAG on the extended golden set, per-category breakdown.
- **Optional fine-tuning** — QLoRA instruct-tune on domain Q&A pairs *or* contrastive encoder fine-tuning on `(Q, pos/neg)` pairs.
- **`vLLM` hosting** — replace Ollama with `vLLM` continuous batching for throughput if the benchmark shows it matters.

### Increment 4 — production hardening (post-course)

- Full `garak` probe set + custom cybersec-specific jailbreak probes
- Adversarial golden set with known prompt-injection payloads
- Request-level cost tracking (tokens in/out, per-stage GPU seconds)
- Multi-tenant auth + rate limiting (per-API-key throttling)
- Real-time CVE / MITRE refresh pipeline (indexing triggered by upstream updates)

## Things that will not change between GaRAG and PoxekBook

- Core stack (Qdrant, FastAPI, Gradio, Docker Compose, Prometheus, Grafana)
- Corpus sources (MITRE ATT&CK + ATLAS, OWASP, HackerOne, man pages)
- Pydantic response schema (`QueryResponse` contract stays stable across versions)
- Evaluation harness (`pytrec_eval` + LLM-as-judge)

## Repository relationship

- `garag` — **public** repo, frozen at `v0.1.0-garag` as the GigaSchool submission artefact. Bug-fix only.
- `poxekbook` — **private** repo, template-forked from `garag`, active development of increments 2–4.

When increments land in `poxekbook`, they are not backported to `garag`. The submission snapshot stays clean.
