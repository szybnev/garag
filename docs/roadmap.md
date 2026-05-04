# Roadmap: GaRAG Future Work

**GaRAG** closes the mandatory scoring blocks of the GigaSchool LLM-Engineer
final project and ships as `v0.1.0-garag`. Everything that is not strictly
required for that score lives in this roadmap.

## What is already in GaRAG v0.1.0

- Single hybrid RAG pipeline: dense (`text-embedding-qwen3-embedding-0.6b`) + sparse (`rank_bm25`) → alpha fusion (`alpha=0.3`) → `bge-reranker-v2-m3` → `zai-org/glm-4.7-flash`
- Single chunking strategy (`RecursiveChunker 256 gpt2`) with theoretical justification
- Single LLM for generation, single LLM (`qwen3.5:35b`) as judge
- 50-item golden set, categories: factual, tool usage, multi-hop
- FastAPI runtime (`/health`, `/query`, `/metrics`) + Gradio mounted at `/gradio`
- Docker Compose with Qdrant, app, Prometheus, and Grafana
- Structured output `{answer, citations[], confidence, used_chunks[]}`
- Sources block in the Gradio UI with a fallback to retrieved chunks when the generator omits explicit citations
- MITRE ATT&CK tactic documents enriched with related technique lists from `kill_chain_phases`

## Future Increments

### Increment 2 — Experimental Depth

- **E1: embedding comparison** — `bge-m3` vs `nomic-embed-text-v2-moe` vs `snowflake-arctic-embed2` on the golden set. Cross-lingual RU→EN retrieval eval.
- **E2: chunking comparison** — fixed vs recursive vs semantic vs entity-aware. Per-source winners.
- **E3: sparse retrieval** — compare `rank_bm25` (current) vs `bge-m3 learned sparse` vs tuned BM25 with cybersec stopwords.
- **E4: fusion methods** — RRF vs MMR vs alpha-weighted, grid over `alpha ∈ [0, 1]`.
- **E5: reranker on/off** — quantitative trade-off between quality and latency.
- **E6: LLM benchmark** — 4–5 candidate LLMs (3B–35B), 3-stage filter (censorship → TPS → quality). Pareto-optimal choice.
- **Extended golden set** — grow from 50 to 100–150 pairs.

### Increment 3 — Advanced RAG

- **GraphRAG** — NetworkX in-memory graph over MITRE ATT&CK (~2,000 nodes). Multi-hop queries: technique → group → mitigation.
- **Agentic RAG** — `smolagents` `ToolCallingAgent` with `retrieve_dense`, `retrieve_sparse`, `rerank`, `graph_lookup`, `answer` tools.
- **E7: RAG comparison** — Naive vs Hybrid (GaRAG) vs GraphRAG vs Agentic RAG on the extended golden set, per-category breakdown.
- **Optional fine-tuning** — QLoRA instruct-tune on domain Q&A pairs or contrastive encoder fine-tuning on `(Q, pos/neg)` pairs.
- **`vLLM` hosting** — replace LM Studio with `vLLM` continuous batching for throughput if the benchmark shows it matters.

### Increment 4 — Production Hardening

- Production-grade guardrail policies beyond the current Granite Guardian checks
- Full `garak` probe set + custom cybersec-specific jailbreak probes
- Extended NFR benchmark sweeps for latency, throughput, and indexing time
- Adversarial golden set with known prompt-injection payloads
- Request-level cost tracking (tokens in/out, per-stage GPU seconds)
- Multi-tenant auth + rate limiting (per-API-key throttling)
- Real-time CVE / MITRE refresh pipeline (indexing triggered by upstream updates)

## Stable Choices

- Core stack: Qdrant, FastAPI, Gradio, Docker Compose, Prometheus, Grafana
- Corpus sources: MITRE ATT&CK + ATLAS, OWASP, HackerOne metadata, man pages
- Pydantic response schema: `QueryResponse`
- Evaluation harness: `pytrec_eval` + LLM-as-judge
