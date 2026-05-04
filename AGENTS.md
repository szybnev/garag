# Agent Instructions

Guidance for Codex GPT-5.5 and other coding agents working in this repository.
Keep this file practical: it should help the next agent act correctly without
having to reconstruct project context from `CLAUDE.md`.

## Project Context

**GaRAG** is an academic MVP submitted as the
final project for the **GigaSchool LLM-Engineer course (track A)**.

It is a hybrid retrieval-augmented generation system over a cybersecurity corpus:
MITRE ATT&CK, MITRE ATLAS, OWASP Top 10, public HackerOne reports as metadata
only, and security tool man pages.

This is an **academic project**, **not production-ready**. The README disclaimer
must stay intact in any rewrite.

The full development plan lives at
`~/kurs/gigaschool/docs/plans/radiant-questing-sutton.md`. It is the source of
truth for the d1-d15 breakdown, NFR thresholds, and the current GaRAG scope.

## Current Snapshot

| Item | Value |
|---|---|
| Plan day completed | d10 runtime MVP |
| Target release | `v0.1.0-garag` on 2026-04-24 |
| Open public repo | https://github.com/szybnev/garag |
| bd issue tracker | local `bd`; run `bd ready` / inspect `.beads/issues.jsonl` |
| Test suite | 113 tests passing, coverage 82.30% (60% threshold) |
| Local corpus | 2,544 documents, 3,900 chunks, 3,900 Qdrant points |
| Last latency snapshot | retrieval p95 about 4.1 s with reranker |

Latest retrieval metrics on 50 golden queries:

| Method | Recall@10 | nDCG@10 | MAP |
|---|---:|---:|---:|
| dense (qwen3 embedding) | 0.7600 | 0.6721 | 0.6450 |
| sparse (BM25 tuned k1=0.8, b=0.5) | 0.8800 | 0.8024 | 0.7779 |
| hybrid alpha=0.3 | 0.8800 | 0.7880 | 0.7575 |
| hybrid + reranker (current default) | 0.8600 | 0.8089 | 0.7933 |

NFR thresholds from `docs/design.md` section 3: Recall@10 >= 0.75 and nDCG@10
>= 0.65. Both are cleared. The weak category is `tool_usage`; the reranker stays
enabled despite latency cost because it improves tool_usage MAP.

## Codex Operating Rules

- Answer the user in Russian.
- If the task is clear, act without asking for confirmation.
- If the task is ambiguous, ask 1-2 concrete questions.
- If you do not know, say "Не знаю" and ask for the missing context.
- If you are not sure, say "Не уверен в ответе" and explain what is uncertain.
- Change only files relevant to the current task.
- Prefer minimal, root-cause fixes over temporary workarounds.
- Do not refactor neighboring code unless explicitly requested.
- Use `rg` / `rg --files` for search.
- Use `apply_patch` for manual edits.
- Use Context7 for current library documentation when changing library usage.
- Use WebSearch only when internet research is needed.
- Before marking work done, verify it with tests, linters, builds, or an explicit
  docs-only check.
- For non-trivial changes, briefly consider whether there is a simpler design.

## Issue Tracking With bd

This project uses **bd** (beads) for issue tracking. Run `bd onboard` or
`bd prime` for workflow context.

Quick reference:

```bash
bd ready
bd show <id>
bd update <id> --status in_progress
bd create "Title" --type task --priority 2
bd close <id> -r "reason"
bd sync
```

Rules:

- Always use `bd` for task and issue tracking.
- On the first response in a new session, run `bd ready` and show the ready
  task list to the user before starting implementation work.
- File issues for unfinished follow-up work before ending a session.
- Close completed issues before syncing.
- Work is not complete until `bd sync` and `git push` both succeed.

## Landing the Plane

When ending a work session, complete all steps below.

1. File issues for remaining work.
2. Run quality gates if code changed.
3. Close finished `bd` issues or update their status.
4. Push all committed changes:

```bash
git pull --rebase
bd sync
git push
git status
```

5. Verify `git status` says the branch is up to date with origin.
6. Hand off concise context for the next session.

Never stop with local-only completed work. If push fails, resolve and retry.

## Git Conventions

- Commit messages must be in Russian.
- Group changes into separate commits by purpose.
- Match the existing style where useful: `feat(area):`, `chore(bd):`, `docs:`.
- Main branch for PRs is `main`.
- If `.gitlab-ci.yaml` exists, use `glab` for GitLab workflows.
- If `.github/` exists, use `gh` for GitHub workflows.

## Stack And Tuned Parameters

| Area | Current choice |
|---|---|
| Python | 3.12, `uv`, ruff, ty |
| Vector DB | Qdrant `garag_v1`, HNSW `m=16, ef_construct=200`, dim 1024, cosine |
| Embedder | `text-embedding-qwen3-embedding-0.6b` via LM Studio OpenAI-compatible `/v1/embeddings`, dim 1024 |
| Sparse | `rank_bm25.BM25Okapi`, tuned `k1=0.8, b=0.5` + NLTK english stopwords; searchable text includes chunk/doc IDs, source, title, and chunk text |
| Fusion | alpha-weighted min-max, tuned `alpha=0.3`; RRF k=60 also exists |
| Reranker | `BAAI/bge-reranker-v2-m3`, cross-encoder, top-20 to top-5 |
| Generator | `ibm/granite-3.2-8b` via LM Studio OpenAI-compatible `/v1/chat/completions` |
| LLM-as-judge | `qwen3.5:35b`; older same-checkpoint evals carry the d13 self-bias caveat |
| Structured output | `_GeneratedResponse` JSON schema; OpenAI-compatible `response_format` for LM Studio |
| Web layer | FastAPI `/health` `/query` `/metrics`, Gradio mounted at `/gradio`, Docker Compose |
| Observability | Prometheus + Grafana, anonymous viewer |
| Security | `garak` probes + LLM Guard input/output guardrails planned |

Runtime defaults must stay aligned across `app/config.py` and `.env.example`.

## Quick Commands

```bash
# Install / sync
uv sync
make sync

# Lint / typecheck / test
make lint
make format
make test

# Data pipeline
docker compose up -d qdrant
uv run python -m scripts.fetch_mitre_attack
uv run python -m scripts.fetch_mitre_atlas
uv run python -m scripts.fetch_owasp_top10
uv run python -m scripts.fetch_hackerone_reports --limit 500
uv run python -m scripts.fetch_man_pages
uv run python -m scripts.parse_sources
uv run python -m scripts.chunk_corpus
uv run python -m scripts.build_qdrant
uv run python -m scripts.build_bm25 --k1 0.8 --b 0.5

# Evaluation
uv run python -m scripts.eval_retrieval --alpha 0.3 --rerank
uv run python -m scripts.tune_bm25
uv run python -m scripts.tune_fusion
```

Run commands through `uv run`; do not activate the virtualenv with `source`.

## Architecture Map

```text
POST /query (FastAPI, QueryRequest)
  -> HybridRetriever in app/rag/pipeline.py
     -> DenseRetriever: qwen3 embedding + Qdrant query_points
     -> SparseRetriever: rank_bm25 pickle
     -> alpha-weighted fusion, alpha=0.3
     -> Reranker: bge-reranker-v2-m3, top-20 to top-5
  -> Generator in app/rag/generator.py
     -> ibm/granite-3.2-8b via LM Studio /v1/chat/completions
     -> QueryResponse with answer, citations, confidence
```

Reference paths:

| Path | Purpose |
|---|---|
| `app/schemas.py` | `Document`, `Chunk`, `Citation`, `QueryRequest`, `QueryResponse` |
| `app/config.py` | `pydantic-settings` runtime config |
| `app/rag/pipeline.py` | `HybridRetriever` orchestrator |
| `app/rag/embedder.py` | LM Studio embedding wrapper with FlagEmbedding fallback |
| `app/rag/retriever_dense.py` | Qdrant `query_points` |
| `app/rag/retriever_sparse.py` | BM25Okapi pickle loader |
| `app/rag/fusion.py` | RRF and alpha-weighted fusion |
| `scripts/nfr_benchmark.py` | HTTP runtime NFR benchmark |

## Sharp Edges

- **qwen3.5 thinking mode.** Native Ollama uses `/api/chat` with
  `{"think": false}`. Runtime now defaults to LM Studio's OpenAI-compatible
  `/v1/chat/completions`.
- **Ollama structured output on MoE qwen3.5 is leaky.** The full nested
  `QueryResponse` schema caused missing `citations[].source`. The generator asks
  only for `chunk_id` and `quote`, then hydrates `source` and `url` from chunks.
  Do not expand the LLM-side schema without re-running
  `scripts/validate_generator.py`.
- **Ollama lives outside Docker Compose.** Compose reaches it through
  `http://host.docker.internal:11434` with `extra_hosts`. Host scripts must use
  `OLLAMA_URL=http://localhost:11434`.
- **FlagEmbedding / FlagReranker can hang on Hugging Face ETag checks.** Use
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` for CLI scripts that construct a
  retriever or reranker.
- **Qdrant ports are shifted.** Compose maps `6380:6333` and `6381:6334`.
- **`qdrant_client.search` is gone.** Use
  `client.query_points(collection_name, query=vector, ...)` and read `.points`.
- **`BGEM3FlagModel.encode()` returns a Union.** Narrow with `isinstance` around
  `out["dense_vecs"]`, as in `app/rag/embedder.py`.
- **HackerOne is metadata-only.** Never fetch or persist disclosed report bodies.
- **MITRE ATT&CK tactics are enriched locally.** `scripts/parse_sources.py`
  builds tactic documents such as `TA0010 Exfiltration` from
  `kill_chain_phases`, including the related technique list. Procedure examples
  from ATT&CK `relationship` objects are not indexed in the MVP.
- **BM25 depends on searchable metadata.** `scripts/build_bm25.py` indexes
  `chunk_id`, `doc_id`, source, title, and chunk text. Rebuild it after parser
  or chunk schema changes.
- **Golden set deterministic dedup.** `scripts/build_golden.py` excludes already
  collected `source_chunk_id`s before sampling.
- **Use `pytrec-eval-terrier`, not `pytrec-eval`.** The original package fails to
  compile against modern GCC.

## Code Conventions

- Modern Python everywhere.
- `[tool.ty.environment]`, never `[tool.ty]`.
- `ruff select = ["ALL"]` with explicit ignores.
- `[dependency-groups]` instead of `[project.optional-dependencies]`.
- Use `uv add <pkg>` for dependencies; do not hand-edit dependency lists.
- Type-narrow at framework boundaries (`FlagEmbedding`, `qdrant-client`,
  `pickle.load`) so downstream code stays clean.
- Tests use mocks for retrievers. See `tests/test_pipeline.py`.
- Smoke and integration coverage live in notebooks, not pytest.
- Notebooks should have minimal comments and reproduce final numbers.

## Do Not Do

- Do not commit `final-project/design-notebook.ipynb` or any reference to it.
- Do not bring up `pre-commit`; if hooks are added later, use `prek`.
- Do not switch to a `src/` layout.
- Do not raise the test coverage threshold above 60% without adding retrieval
  mock tests first.
- Do not pull HackerOne report bodies.
- Do not invent flags or environment variables outside `app/config.py` and
  `.env.example`. Add new settings to `Settings` first.
