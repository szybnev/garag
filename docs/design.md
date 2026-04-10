# GaRAG design document

> **Scope.** This file documents the architectural decisions behind **GaRAG**,
> the MVP slice of the larger PoxekBook vision. It captures non-functional
> requirements, the choice of models, the chunking strategy, and the legal
> stance on each data source. The full long-term vision (4 RAG pipelines,
> 7 experiments, fine-tuning) lives in the private `poxekbook` repo — see
> [`roadmap_to_poxekbook.md`](./roadmap_to_poxekbook.md) for the delta.

## 1. Context

GaRAG is a hybrid retrieval-augmented generation system for the cybersecurity
domain, built to help interns and junior engineers find authoritative answers
to questions like:

- *"What is MITRE ATT&CK T1059.001 and how is it typically detected?"*
- *"Which OWASP Top 10 category maps to CWE-89?"*
- *"How do I scan UDP ports with nmap?"*
- *"What are the common SSRF patterns in public HackerOne reports?"*

It is an **academic project** submitted as the final assignment for the
GigaSchool LLM-Engineer course. It is **not production-ready** and makes no
claim of being so. See `README.md` for the full disclaimer.

## 2. Corpus

Five sources, unified via `app.schemas.Document` into `data/raw/documents.parquet`
(2 495 documents, 791 KB compressed). Rebuilt from scratch by `scripts/fetch_*.py`
and `scripts/parse_sources.py` — **no raw data is shipped in this repo**.

| Source | Documents | Format | License |
|---|---|---|---|
| MITRE ATT&CK Enterprise | 1 705 | STIX 2.1 JSON | Apache 2.0 |
| MITRE ATLAS | 275 | YAML | Apache 2.0 |
| HackerOne public reports | 500 | CSV (**metadata only**) | see §2.1 |
| OWASP Top 10 (2021, EN) | 10 | Markdown | CC BY-SA 4.0 |
| man pages (5 security tools) | 5 | plain text | mixed |

### 2.1 HackerOne disclaimer

HackerOne reports are pulled from the open-source mirror
[`reddelexc/hackerone-reports`](https://github.com/reddelexc/hackerone-reports),
which aggregates metadata from the public disclosure page. To respect the
HackerOne Terms of Service, GaRAG indexes **only metadata** — program, title,
vulnerability type, bounty amount, upvote count, and the canonical URL. The
body of each report is **never fetched or stored**. The corresponding
`Document.text` is a short formatted summary that always points back to the
original report URL for the reader to review under HackerOne's own terms.

If HackerOne or any reporting researcher objects, the `garag-zqc.5` bid can be
closed as `deferred-legal` and the source excluded entirely — the remaining
four sources still cover the MVP scope.

## 3. Non-functional requirements

All thresholds are **targets for the MVP**, measured at the end of d13 via
`scripts/nfr_benchmark.py` and reported in `evaluation/reports/nfr_report.md`.

| Metric | Target | Rationale |
|---|---|---|
| e2e latency (p95, warm) | ≤ 8 s | Interactive Q&A on a single-user RTX 5090 |
| Throughput | ≥ 2 RPS | Single local instance, no concurrency hardening |
| Indexing time (full corpus) | ≤ 20 min | One-off on cold Qdrant, acceptable for a rebuild step |
| Retrieval Recall@10 | ≥ 0.75 | Measured on the 50-item golden set with hybrid + reranker |
| Retrieval nDCG@10 | ≥ 0.65 | Same |
| Generation faithfulness | ≥ 0.80 | LLM-as-judge (`qwen3.5:35b`), 0–1 scale |
| Generation correctness | ≥ 0.70 | LLM-as-judge vs reference answer |
| Citation accuracy | ≥ 0.85 | Fraction of citations pointing to returned chunks |
| Peak VRAM (without LLM) | ≤ 6 GB | bge-m3 + bge-reranker-v2-m3 together |
| Peak VRAM (with LLM) | ≤ 28 GB | Headroom on the RTX 5090 (32 GB) |

These targets are intentionally loose. The point of the MVP is to prove the
hybrid+rerank+structured-output pipeline runs end-to-end and clears the
GigaSchool grading criteria — not to out-benchmark open-source systems. PoxekBook
increments 2–3 tighten each threshold independently.

## 4. Architectural decisions

### 4.1 Embedding: `BAAI/bge-m3`

**Chosen because:**
- One model serves both dense and potential sparse retrieval (we use dense
  only for MVP; learned sparse is a PoxekBook E1 option).
- 8 192-token context accommodates our 256-token chunks with room to spare.
- Multilingual — lets us accept Russian-language queries against an English
  corpus without an additional translation step.
- Strong out-of-the-box retrieval scores on BEIR without fine-tuning.

Alternatives not picked for MVP: `nomic-embed-text-v2-moe`, `snowflake-arctic-embed2`.
Both will be benchmarked head-to-head in PoxekBook E1.

### 4.2 Sparse retrieval: `rank_bm25`, not bge-m3 learned sparse

**Chosen because:**
- Explicit `k1` / `b` knobs give us a tunable, which closes the 3-point
  "ranking tuning" requirement of `task.md` unambiguously.
- Fewer moving parts — no second Qdrant collection, no dual embedding pass.
- A grid search over `k1 ∈ [0.5, 2.0]` and `b ∈ [0.25, 1.0]` runs in seconds
  on the golden set.

bge-m3 learned sparse is a PoxekBook increment 2 item and is explicitly not
tried in MVP.

### 4.3 Reranker: `BAAI/bge-reranker-v2-m3`

Cross-encoder reranker on top-20 → top-5. Same family as the embedder, so
they share tokenisation behaviour and language coverage. The with/without
comparison lands in `experiments/03_retrieval_tuning.ipynb` on d7.

### 4.4 Generator: `qwen3.5:35b` via Ollama

Single LLM for MVP generation. Alternatives (4 B / 9 B / 27 B) are benchmarked
in PoxekBook E6, not here. The 14 B variant is not in the local Ollama cache
on the d9 host; we use 35 B MoE instead.

**Non-obvious wrinkles:**
- qwen3.5 uses a thinking mode. Through the OpenAI-compatible endpoint the
  `content` field can come back empty while the actual answer sits in
  `thinking`. We work around this by calling the native `/api/chat` endpoint
  with `{"think": false}` in the payload — verified on hw12-advanced-rag.
- Ollama's `format=<json_schema>` structured output reliably drops `required`
  fields on qwen3.5:35b MoE when nested objects have too many required keys
  (observed d9: 20/20 golden queries failed Pydantic validation on
  `citations[].source`). `app/rag/generator.py` works around this by asking
  the LLM only for `chunk_id` + `quote` per citation and hydrating
  `source` / `url` post-hoc from the retrieved chunks.

#### 4.4.1 Generation parameters — grid search

Tuned on d10 via `scripts/tune_gen_params.py`, a 36-config grid over
`(temperature × top_p × num_predict)` on the first 20 queries of
`golden_set_v1.jsonl`, with retrieval cached once per query so the only
moving variable is `Generator.generate()`. Grid:

- `temperature ∈ {0.0, 0.1, 0.2, 0.4}`
- `top_p       ∈ {0.8, 0.9, 1.0}`
- `num_predict ∈ {400, 800, 1200}`

Per-config score: `mean_confidence + 0.1·mean_citations - 0.02·mean_latency_gen_s`,
tiebreak on `grounded_rate desc → p95_latency asc`.

**Key finding:** every `num_predict=400` config clipped 1-2 of the 20 answers
(`format_rate ∈ [0.90, 0.95]`) — tool_usage and multi_hop answers blow past
400 tokens when the model explains a command or chains two sources. Configs
with `num_predict=400` are discarded even when their heuristic score ranks
highest (top-1 unconstrained was `temp=0.40 top_p=1.00 num_predict=400` with
`fmt=0.95` — unusable despite the best score).

Top-5 configs with `format_rate == 1.0`, sorted by heuristic score:

| rank | temp | top_p | num_predict | score | cites | p95 gen ms |
|---|---|---|---|---|---|---|
| **1** | **0.40** | **1.00** | **800** | **1.0909** | **2.05** | **2907** |
| 2 | 0.40 | 1.00 | 1200 | 1.0906 | 2.05 | 2967 |
| 3 | 0.10 | 1.00 | 800 | 1.0867 | 1.95 | 3425 |
| 4 | 0.10 | 0.90 | 1200 | 1.0864 | 1.95 | 3517 |
| 5 | 0.10 | 1.00 | 1200 | 1.0857 | 1.95 | 3643 |

**Chosen:** `temperature=0.4, top_p=1.0, num_predict=800`. Rationale:
- First `fmt=1.0` config by score.
- Fastest `p95 gen` of the fmt=1.0 tier (2907 ms vs 3425 ms for the next temp=0.1 block → 15 % headroom on the NFR p95 ≤ 8000 ms target).
- Highest mean citation count (2.05 per answer vs 1.95 for lower temperatures), meaning more evidence in each response.
- Stochasticity concern at `temp=0.4` is moot in practice because `seed=42` is pinned in `Settings.gen_seed`; reruns are bit-identical.
- `num_predict=1200` at the same `(temp, top_p)` adds 60 ms of p95 latency for the same score — wasteful.

Raw grid results and per-config breakdown live in
`evaluation/results/gen_params_grid.json` and
`experiments/04_generation_params.ipynb`.

### 4.5 LLM-as-judge: `qwen3.5:35b` (same model as generator)

Originally planned as a separate (larger) model than the generator to reduce
the "grading your own homework" bias. On the d9 host `qwen3.5:35b` is the
largest Ollama-available model, so generator and judge share the same
checkpoint. We accept this compromise for the MVP with two mitigations:

1. **Self-bias caveat** is printed verbatim in the header of every report
   produced by `scripts/eval_generation.py` — literature on LLM-as-judge
   (Zheng et al. 2023) reports 5-15 % upward bias on faithfulness when the
   generator and judge models agree on a checkpoint. Treat absolute numbers
   as upper bounds and rely on per-category deltas + manual 10-sample review
   for qualitative signal.
2. **Cross-model rerun** (GPT-4o or Claude as judge) is deferred to
   PoxekBook increment 2 alongside the 10-LLM benchmark.

Judge is called with `temperature=0.0`, `top_p=1.0`, `num_predict=300`,
`seed=42` — deterministic single-pass scoring on the 50-item golden set.

### 4.6 Chunking: `chonkie.RecursiveChunker(tokenizer="gpt2", chunk_size=256)`

Single strategy for MVP, full justification in
`experiments/01_chunking_choice.ipynb`. TL;DR:

- 3 779 chunks from 2 495 documents (average fan-out ×1.5)
- Median 148 tokens per chunk, target 256 — well-utilised budget
- Recursive hierarchy (paragraph → sentence → punctuation → whitespace)
  respects semantic boundaries better than fixed-size splitting
- gpt2 tokenizer is a close proxy for what both BM25 and the reranker will
  see downstream

Fixed-size, semantic, and per-entity alternatives are listed in the notebook
and scheduled for comparison in PoxekBook E2.

### 4.7 Vector DB: Qdrant in Docker Compose

HNSW with `m=16`, `ef_construct=200`, cosine distance, single collection
`garag_v1`. Sparse retrieval is kept **out of Qdrant** as a pickled
`BM25Okapi` object — this is the minimum-complexity path. Qdrant lives on
shifted ports (`6380`, `6381`) so it can coexist with any default 6333
installation on the host.

### 4.8 Orchestration: app in Docker, Ollama outside

The `ollama` container already runs on the host with `qwen3.5:14b` and
`qwen3.5:35b` pulled. Re-bundling it inside `garag`'s compose file would
duplicate ~30 GB of model weights. Instead, `app` reaches Ollama via
`http://host.docker.internal:11434` with an explicit
`extra_hosts: ["host.docker.internal:host-gateway"]` clause (required on
Linux).

## 5. What GaRAG intentionally does **not** have

- Multi-step GraphRAG (MITRE ATT&CK knowledge graph) — PoxekBook increment 3
- Agentic RAG with `smolagents` tools — PoxekBook increment 3
- Embedding benchmark (3 models) — PoxekBook E1
- LLM benchmark (4–5 models × 3-stage filter) — PoxekBook E6
- QLoRA / DPO fine-tuning — PoxekBook increment 3
- 100–150 item extended golden set — PoxekBook increment 2
- vLLM for throughput — only if the NFR benchmark shows Ollama is the bottleneck

## 6. References in this repository

- `README.md` — quickstart + academic disclaimer
- `docs/roadmap_to_poxekbook.md` — everything listed in §5, with targets
- `app/schemas.py` — `Document`, `Chunk`, (d9) `Citation`, `QueryRequest`,
  `QueryResponse`
- `scripts/parse_sources.py` — corpus unification entry point
- `scripts/chunk_corpus.py` — chunking with `RecursiveChunker 256 gpt2`
- `experiments/00_data_overview.ipynb` — per-source statistics
- `experiments/01_chunking_choice.ipynb` — chunking rationale + distribution
- `evaluation/reports/` — retrieval / generation / NFR reports (filled in d6,
  d10, d13 respectively)
