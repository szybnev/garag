# CLAUDE.md

Guidance for Claude Code (and any other coding agent) working in this repository.

## Project context

**GaRAG** is the academic MVP slice of a larger PoxekBook vision, submitted as the
final project for the **GigaSchool LLM-Engineer course (track A)**. It is a hybrid
retrieval-augmented generation system over a cybersecurity corpus (MITRE ATT&CK +
ATLAS, OWASP Top 10, public HackerOne reports as **metadata only**, security tool
man pages).

It is an **academic project**, **not production-ready**, and the README disclaimer
must stay intact in any rewrite. Do not remove it.

The full development plan lives at
`~/kurs/gigaschool/docs/plans/radiant-questing-sutton.md` — it is the source of
truth for the day-by-day breakdown (d1–d15), NFR thresholds, and the boundary
between GaRAG (this repo) and PoxekBook (private continuation).

## Current state

| | |
|---|---|
| **Plan day completed** | d10 runtime MVP |
| **Target release** | `v0.1.0-garag` on 2026-04-24 |
| **Open public repo** | https://github.com/szybnev/garag |
| **Private continuation** | https://github.com/szybnev/poxekbook (empty stub) |
| **bd issue tracker** | local `bd`; run `bd ready` / inspect `.beads/issues.jsonl` |
| **Test suite** | 90 tests passing, coverage 83% (60% threshold) |
| **Last latency snapshot** | retrieval p95 ~4.1 s with reranker (NFR target: e2e ≤ 8 s) |

### Latest retrieval metrics on 50 golden queries

| Method | Recall@10 | nDCG@10 | MAP |
|---|---|---|---|
| dense (bge-m3) | 0.8600 | 0.7261 | 0.6855 |
| sparse (BM25 tuned k1=0.8, b=0.5) | 0.8800 | 0.7844 | 0.7532 |
| hybrid alpha=0.3 | 0.8800 | 0.7890 | 0.7589 |
| **hybrid + reranker (current default)** | **0.8600** | **0.8089** | **0.7933** |

NFR thresholds (`docs/design.md §3`): Recall@10 ≥ 0.75, nDCG@10 ≥ 0.65 — both
cleared with 11–16 п.п. headroom. The weak category is **tool_usage** (Recall@10
≈ 0.53, mostly homogeneous nmap/nuclei chunks); the reranker brings tool_usage MAP
+12.45 п.п. and is therefore kept in the pipeline despite the 2.8 s latency cost.

## Stack and tuned parameters

| | |
|---|---|
| Python | 3.12, **uv** + ruff + ty (modern-python skill) |
| Vector DB | Qdrant `garag_v1`, HNSW `m=16, ef_construct=200`, dim 1024, cosine |
| Embedder | `text-embedding-qwen3-embedding-0.6b` via LM Studio OpenAI-compatible `/v1/embeddings`, dim 1024 |
| Sparse | `rank_bm25.BM25Okapi`, **tuned** `k1=0.8, b=0.5` + nltk english stopwords |
| Fusion | alpha-weighted min-max, **tuned** `alpha=0.3` (RRF k=60 also implemented) |
| Reranker | `BAAI/bge-reranker-v2-m3` cross-encoder, top-20 → top-5 |
| Generator (runtime) | `qwen/qwen3.6-35b-a3b` via LM Studio OpenAI-compatible `/v1/chat/completions` |
| LLM-as-judge | `qwen3.5:35b` (same model — MoE 36B is the largest available locally; self-bias caveat acknowledged for d13) |
| Structured output | `_GeneratedResponse` JSON schema; OpenAI-compatible `response_format` for LM Studio |
| Web layer (d10 runtime MVP) | FastAPI `/health` `/query` `/metrics`, Gradio mounted at `/gradio`, Docker Compose |
| Observability (d11) | Prometheus + Grafana, anonymous viewer |
| Security (d12) | `garak` probes + LLM Guard input/output guardrails planned |

All tuned values are persisted in `app/config.py` (`pydantic-settings`) and
`.env.example` so the runtime defaults match the evaluated configuration.

## Quick command reference

```bash
# Install / sync
uv sync                                          # all deps + dev group
make sync                                        # alias

# Lint / typecheck / test
make lint                                        # ruff check (ALL) + ty
make format                                      # ruff format
make test                                        # pytest with coverage

# Data pipeline (one-off, ~1 min total on a warm cache)
docker compose up -d qdrant                      # start vector DB only
uv run python -m scripts.fetch_mitre_attack      # ~50 MB
uv run python -m scripts.fetch_mitre_atlas
uv run python -m scripts.fetch_owasp_top10
uv run python -m scripts.fetch_hackerone_reports --limit 500
uv run python -m scripts.fetch_man_pages
uv run python -m scripts.parse_sources           # → data/raw/documents.parquet
uv run python -m scripts.chunk_corpus            # → data/processed/chunks.parquet
uv run python -m scripts.build_qdrant            # 3779 points, ~10 s
uv run python -m scripts.build_bm25              # → data/index/bm25.pkl, ~1 s

# Evaluation
uv run python -m scripts.eval_retrieval --alpha 0.3 --rerank
uv run python -m scripts.tune_bm25               # 28 configs in 3 s
uv run python -m scripts.tune_fusion             # alpha sweep

# bd issue tracker
bd ready                                         # what's next
bd update <id> --status in_progress
bd close <id> -r "reason"
bd sync                                          # MUST run before git push
```

## Architecture overview

```
                  ┌────────────────────────────┐
                  │   POST /query (FastAPI)    │  d10
                  │   QueryRequest             │
                  └──────────┬─────────────────┘
                             │
                  ┌──────────▼──────────────┐
                  │   HybridRetriever       │  app/rag/pipeline.py
                  ├─────────────────────────┤
                  │ DenseRetriever          │  bge-m3 + Qdrant query_points
                  │ SparseRetriever         │  rank_bm25 pickle
                  │ alpha-weighted fusion   │  alpha=0.3
                  │ Reranker (bge-rer-v2)   │  cross-encoder, top-20→top-5
                  └──────────┬──────────────┘
                             │
                  ┌──────────▼──────────────┐
                  │   Generator (d9)        │  app/rag/generator.py
                  │ qwen/qwen3.6-35b-a3b   │  LM Studio /v1/chat/completions
                  │   format=QueryResponse  │  structured output
                  └──────────┬──────────────┘
                             │
                  ┌──────────▼──────────────┐
                  │   QueryResponse         │
                  │   answer + citations[]  │
                  │   confidence ∈ [0,1]    │
                  └─────────────────────────┘
```

## Sharp edges

These are the non-obvious things that have already bitten us once. **Read before
touching the affected modules.**

- **qwen3.5 thinking mode.** The OpenAI-compatible endpoint can return an empty
  `content` field with the actual answer in `thinking`. Always use the **native
  Ollama `/api/chat`** with `{"think": false}` in the payload. Pattern verified
  in `hw12-advanced-rag` from the parent `gigaschool` repo.
- **Ollama structured output on MoE qwen3.5 is leaky.** Feeding the Ollama
  `format` field a JSON schema with many `required` fields per nested object
  (we tried the full `QueryResponse.model_json_schema()` including
  `Citation.source` / `Citation.url`) reliably produces JSON missing the
  `required` fields — 20/20 golden queries failed Pydantic validation on
  `citations[].source`. `app/rag/generator.py` sidesteps this by asking the
  LLM only for `chunk_id` + `quote` per citation and hydrating `source` /
  `url` post-hoc from the retrieved chunks. **Do not** expand the LLM-side
  schema without re-running `scripts/validate_generator.py`.
- **Ollama lives outside Docker Compose.** The `ollama` container is started
  separately on the host. The app inside Compose reaches it via
  `http://host.docker.internal:11434` with `extra_hosts: ["host.docker.internal:host-gateway"]`
  in `docker-compose.yml` (Linux requires the explicit `host-gateway` mapping).
  **Running scripts from the host** (e.g. `scripts/validate_generator.py`)
  must override `OLLAMA_URL=http://localhost:11434` — `host.docker.internal`
  only resolves inside the compose network.
- **`FlagEmbedding` / `FlagReranker` hang on HuggingFace ETag checks.** Even
  when the model is fully cached under `~/.cache/huggingface/hub/`, the init
  path opens a connection to HuggingFace Hub to check for newer revisions and
  can block for minutes (observed with `bge-reranker-v2-m3` on d9). Always
  set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` for any script that constructs
  a retriever or reranker from CLI, and keep those exports in the Makefile /
  Dockerfile once you wire them up.
- **Qdrant ports are shifted to 6380/6381.** `docker-compose.yml` maps
  `6380:6333` and `6381:6334` to avoid clashing with any default-port Qdrant the
  user might already have running. `app/config.py` and `.env.example` reflect this.
- **`qdrant_client.search` is gone.** The new API in `qdrant-client>=1.16` is
  `client.query_points(collection_name, query=vector, ...)`. Returns a
  `QueryResponse` with `.points`, not a flat list.
- **`FlagEmbedding.BGEM3FlagModel.encode()` returns a Union.** When you pass
  `return_dense=True, return_sparse=False, return_colbert_vecs=False`, you do get
  a numpy array under `out["dense_vecs"]`, but `ty` cannot infer that — use the
  `isinstance(vecs, np.ndarray)` narrow that's already in `app/rag/embedder.py`.
- **HackerOne is metadata-only.** Per ToS, we never fetch or persist the body
  of disclosed reports. `Document.text` for HackerOne is a short formatted
  summary (program / vuln_type / bounty / upvotes) plus the canonical link. If
  HackerOne or any reporting researcher objects, close `garag-zqc.5` as
  `deferred-legal` and exclude the source — the other four sources still cover
  the MVP scope (`docs/design.md §2.1`).
- **Golden set deterministic dedup.** `scripts/build_golden.py` excludes the
  `source_chunk_id`s of already-collected pairs before sampling, otherwise
  re-running with the same seed produces byte-identical duplicates because
  `qwen3.5:35b` is fixed-seed deterministic.
- **`pytrec-eval-terrier`, not `pytrec-eval`.** The original `pytrec-eval` 0.5
  fails to compile against modern GCC. Use the `pytrec-eval-terrier` fork (it's
  already in `pyproject.toml`).

## Conventions

- **Modern-python everywhere.** `[tool.ty.environment]`, **never** `[tool.ty]`.
  `ruff select = ["ALL"]` + explicit `ignore`. `[dependency-groups]` instead of
  `[project.optional-dependencies]`. Use `uv add <pkg>`, never edit
  `pyproject.toml` deps by hand.
- **Run things via `uv run`, never `source .venv/bin/activate`.**
- **Type-narrow at framework boundaries.** `FlagEmbedding`, `qdrant-client`, and
  `pickle.load` all return Union or Any types — narrow them with `isinstance` /
  `cast` once at the wrapper layer so downstream code stays clean.
- **Tests use mocks for retrievers.** See `tests/test_pipeline.py` —
  `_StubRetriever` and `_StubReranker` keep the suite GPU-free and Qdrant-free.
  Smoke / integration coverage happens in `experiments/02_indexing_and_hnsw.ipynb`
  and `03_retrieval_tuning.ipynb`, not in pytest.
- **Commit messages in Russian**, grouped by purpose. Match the existing
  `feat(area):` / `chore(bd):` / `docs:` style in `git log`.
- **Notebooks: minimum comments.** Reproduce final numbers; do not pad with
  prose. The plan file (`docs/plans/radiant-questing-sutton.md`) holds the why.
- **bd workflow.** Every day's work ends with `bd sync && git push` — see
  `AGENTS.md` for the "landing the plane" checklist.

## Things to **not** do

- **Do not commit `final-project/design-notebook.ipynb` or any reference to it.**
  It lives in the parent `gigaschool/` repo, not here. The git history of this
  repo was rewritten on d1 specifically to scrub all mentions before the first
  public push. Do not re-add them, even as relative-path links in markdown.
- **Do not bring up pre-commit hooks.** There is no existing pre-commit
  infrastructure. If we add one later, it should be `prek`, not `pre-commit`.
- **Do not switch to `src/` layout.** The package lives at `app/` directly to
  keep `Dockerfile`, scripts, and notebook imports simple. The PoxekBook fork
  may revisit this; GaRAG MVP will not.
- **Do not raise the test coverage threshold above 60%** without first adding
  more retrieval mock-tests. `app/rag/embedder.py`, `retriever_dense.py`, and
  `retriever_sparse.py` are exercised by the integration notebooks rather than
  pytest, so the unit-test coverage on them stays low by design.
- **Do not pull HackerOne report bodies.** Metadata only, see the sharp edge
  above.
- **Do not invent flags or env vars not in `app/config.py` / `.env.example`.**
  If something needs to be configurable, add it to `Settings` first.

## Reference paths

| Path | What it is |
|---|---|
| `app/schemas.py` | `Document`, `Chunk`, `Citation`, `QueryRequest`, `QueryResponse` |
| `app/config.py` | `pydantic-settings` runtime config with tuned defaults |
| `app/rag/pipeline.py` | `HybridRetriever` orchestrator |
| `app/rag/embedder.py` | LM Studio embedding wrapper with FlagEmbedding fallback |
| `app/rag/retriever_dense.py` | Qdrant `query_points` |
| `app/rag/retriever_sparse.py` | BM25Okapi pickle loader |
| `app/rag/fusion.py` | RRF + alpha-weighted, with unit tests |
| `app/rag/reranker.py` | bge-reranker-v2-m3 wrapper |
| `scripts/fetch_*.py` | per-source download (5 files) |
| `scripts/nfr_benchmark.py` | HTTP runtime NFR benchmark |
| `scripts/parsers/*.py` | per-source `Document` constructors (5 files) |
| `scripts/parse_sources.py` | unified pipeline orchestrator |
| `scripts/chunk_corpus.py` | chonkie RecursiveChunker 256 gpt2 |
| `scripts/build_qdrant.py`, `build_bm25.py` | index builders |
| `scripts/build_golden.py` | LLM-assisted golden set generator (qwen3.5:35b) |
| `scripts/eval_retrieval.py` | pytrec_eval harness with `--rerank` |
| `scripts/tune_bm25.py`, `tune_fusion.py` | grid search |
| `experiments/00_data_overview.ipynb` | corpus EDA |
| `experiments/01_chunking_choice.ipynb` | chunking justification |
| `experiments/02_indexing_and_hnsw.ipynb` | HNSW + BM25 verification |
| `experiments/03_retrieval_tuning.ipynb` | retrieval pipeline summary |
| `evaluation/reports/retrieval_report.md` | latest metrics table |
| `docs/design.md` | NFR + architectural decisions |
| `docs/roadmap_to_poxekbook.md` | what is intentionally left out of MVP |
| `tests/test_*.py` | 56 unit tests, 60% coverage gate |

## Useful external paths (read-only)

The parent `gigaschool/` repo contains the course homework notebooks that
inspired several patterns here. They are read-only references, do not import
from them at runtime.

- `~/kurs/gigaschool/docs/plans/radiant-questing-sutton.md` — full d1–d15 plan
- `~/kurs/gigaschool/hw07-rag-basics/` — golden set + LLM-as-judge prompt
- `~/kurs/gigaschool/hw09-vectors/` — Qdrant upsert pattern
- `~/kurs/gigaschool/hw10-information-retrieval/` — BM25 + pytrec_eval
- `~/kurs/gigaschool/hw11-ranking/` — CrossEncoder reranker pattern
- `~/kurs/gigaschool/hw12-advanced-rag/` — Ollama `/api/chat` `think=false` bypass
