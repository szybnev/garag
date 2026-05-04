# Retrieval evaluation — `golden_set_v1`

Run on **50 queries** with `candidate_k=20`. Hybrid alpha-weighted uses `alpha=0.3`. Metrics computed via `pytrec_eval`.

**Reranker:** `BAAI/bge-reranker-v2-m3` cross-encoder applied on top of `hybrid alpha=0.3` candidates.

| Method | Recall@5 | nDCG@10 | Recall@10 | MAP | Latency (s) |
|---|---|---|---|---|---|
| dense (qwen3 embedding) | 0.7600 | 0.6721 | 0.7600 | 0.6450 | 0.6 |
| sparse (BM25) | 0.8600 | 0.8024 | 0.8800 | 0.7779 | 0.2 |
| hybrid RRF (k=60) | 0.8000 | 0.7519 | 0.8400 | 0.7267 | 0.9 |
| hybrid alpha=0.3 | 0.8600 | 0.7880 | 0.8800 | 0.7575 | 0.9 |
| hybrid + reranker | 0.8400 | 0.8089 | 0.8600 | 0.7933 | 4.1 |

## Per-category breakdown

### factual (20 queries)

| Method | Recall@5 | nDCG@10 | Recall@10 | MAP | Latency (s) |
|---|---|---|---|---|---|
| dense | 0.8000 | 0.7009 | 0.8000 | 0.6722 | 0.0 |
| sparse | 0.9500 | 0.9232 | 1.0000 | 0.8988 | 0.0 |
| hybrid RRF | 0.9000 | 0.8424 | 0.9500 | 0.8122 | 0.0 |
| hybrid alpha | 0.9500 | 0.8939 | 1.0000 | 0.8604 | 0.0 |
| hybrid + reranker | 0.9500 | 0.9275 | 1.0000 | 0.9050 | 0.0 |

### tool_usage (15 queries)

| Method | Recall@5 | nDCG@10 | Recall@10 | MAP | Latency (s) |
|---|---|---|---|---|---|
| dense | 0.4667 | 0.3795 | 0.4667 | 0.3539 | 0.0 |
| sparse | 0.6000 | 0.4682 | 0.6000 | 0.4278 | 0.0 |
| hybrid RRF | 0.4667 | 0.4079 | 0.5333 | 0.3727 | 0.0 |
| hybrid alpha | 0.6000 | 0.4595 | 0.6000 | 0.4111 | 0.0 |
| hybrid + reranker | 0.5333 | 0.4841 | 0.5333 | 0.4711 | 0.0 |

### multi_hop (15 queries)

| Method | Recall@5 | nDCG@10 | Recall@10 | MAP | Latency (s) |
|---|---|---|---|---|---|
| dense | 1.0000 | 0.9262 | 1.0000 | 0.9000 | 0.0 |
| sparse | 1.0000 | 0.9754 | 1.0000 | 0.9667 | 0.0 |
| hybrid RRF | 1.0000 | 0.9754 | 1.0000 | 0.9667 | 0.0 |
| hybrid alpha | 1.0000 | 0.9754 | 1.0000 | 0.9667 | 0.0 |
| hybrid + reranker | 1.0000 | 0.9754 | 1.0000 | 0.9667 | 0.0 |

