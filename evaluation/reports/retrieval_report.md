# Retrieval evaluation — `golden_set_v1`

Run on **50 queries** with `candidate_k=20`. Hybrid alpha-weighted uses `alpha=0.3`. Metrics computed via `pytrec_eval`.

**Reranker:** `BAAI/bge-reranker-v2-m3` cross-encoder applied on top of `hybrid alpha=0.3` candidates.

| Method | nDCG@10 | MAP | Recall@10 | Recall@5 | Latency (s) |
|---|---|---|---|---|---|
| dense (bge-m3) | 0.7261 | 0.6855 | 0.8600 | 0.7600 | 1.1 |
| sparse (BM25) | 0.7844 | 0.7532 | 0.8800 | 0.8600 | 0.2 |
| hybrid RRF (k=60) | 0.7783 | 0.7513 | 0.8600 | 0.8600 | 1.3 |
| hybrid alpha=0.3 | 0.7890 | 0.7589 | 0.8800 | 0.8600 | 1.3 |
| hybrid + reranker | 0.8089 | 0.7933 | 0.8600 | 0.8400 | 4.1 |

## Per-category breakdown

### factual (20 queries)

| Method | nDCG@10 | MAP | Recall@10 | Recall@5 | Latency (s) |
|---|---|---|---|---|---|
| dense | 0.8982 | 0.8662 | 1.0000 | 0.9500 | 0.0 |
| sparse | 0.9004 | 0.8688 | 1.0000 | 0.9500 | 0.0 |
| hybrid RRF | 0.9631 | 0.9500 | 1.0000 | 1.0000 | 0.0 |
| hybrid alpha | 0.9346 | 0.9125 | 1.0000 | 1.0000 | 0.0 |
| hybrid + reranker | 0.9275 | 0.9050 | 1.0000 | 0.9500 | 0.0 |

### tool_usage (15 queries)

| Method | nDCG@10 | MAP | Recall@10 | Recall@5 | Latency (s) |
|---|---|---|---|---|---|
| dense | 0.3650 | 0.3152 | 0.5333 | 0.4000 | 0.0 |
| sparse | 0.4386 | 0.3856 | 0.6000 | 0.6000 | 0.0 |
| hybrid RRF | 0.3929 | 0.3489 | 0.5333 | 0.5333 | 0.0 |
| hybrid alpha | 0.4083 | 0.3463 | 0.6000 | 0.5333 | 0.0 |
| hybrid + reranker | 0.4841 | 0.4708 | 0.5333 | 0.5333 | 0.0 |

### multi_hop (15 queries)

| Method | nDCG@10 | MAP | Recall@10 | Recall@5 | Latency (s) |
|---|---|---|---|---|---|
| dense | 0.8576 | 0.8148 | 1.0000 | 0.8667 | 0.0 |
| sparse | 0.9754 | 0.9667 | 1.0000 | 1.0000 | 0.0 |
| hybrid RRF | 0.9175 | 0.8889 | 1.0000 | 1.0000 | 0.0 |
| hybrid alpha | 0.9754 | 0.9667 | 1.0000 | 1.0000 | 0.0 |
| hybrid + reranker | 0.9754 | 0.9667 | 1.0000 | 1.0000 | 0.0 |

