# Retrieval evaluation — `golden_set_v1`

Run on **50 queries** with `candidate_k=20`. Hybrid alpha-weighted uses `alpha=0.3`. Metrics computed via `pytrec_eval`.

| Method | Recall@5 | nDCG@10 | MAP | Recall@10 | Latency (s) |
|---|---|---|---|---|---|
| dense (bge-m3) | 0.7600 | 0.7261 | 0.6855 | 0.8600 | 1.1 |
| sparse (BM25) | 0.8600 | 0.7844 | 0.7532 | 0.8800 | 0.2 |
| hybrid RRF (k=60) | 0.8600 | 0.7783 | 0.7513 | 0.8600 | 1.2 |
| hybrid alpha=0.3 | 0.8600 | 0.7890 | 0.7589 | 0.8800 | 1.2 |

## Per-category breakdown

### factual (20 queries)

| Method | Recall@5 | nDCG@10 | MAP | Recall@10 | Latency (s) |
|---|---|---|---|---|---|
| dense | 0.9500 | 0.8982 | 0.8662 | 1.0000 | 0.0 |
| sparse | 0.9500 | 0.9004 | 0.8688 | 1.0000 | 0.0 |
| hybrid RRF | 1.0000 | 0.9631 | 0.9500 | 1.0000 | 0.0 |
| hybrid alpha=0.3 | 1.0000 | 0.9346 | 0.9125 | 1.0000 | 0.0 |

### tool_usage (15 queries)

| Method | Recall@5 | nDCG@10 | MAP | Recall@10 | Latency (s) |
|---|---|---|---|---|---|
| dense | 0.4000 | 0.3650 | 0.3152 | 0.5333 | 0.0 |
| sparse | 0.6000 | 0.4386 | 0.3856 | 0.6000 | 0.0 |
| hybrid RRF | 0.5333 | 0.3929 | 0.3489 | 0.5333 | 0.0 |
| hybrid alpha=0.3 | 0.5333 | 0.4083 | 0.3463 | 0.6000 | 0.0 |

### multi_hop (15 queries)

| Method | Recall@5 | nDCG@10 | MAP | Recall@10 | Latency (s) |
|---|---|---|---|---|---|
| dense | 0.8667 | 0.8576 | 0.8148 | 1.0000 | 0.0 |
| sparse | 1.0000 | 0.9754 | 0.9667 | 1.0000 | 0.0 |
| hybrid RRF | 1.0000 | 0.9175 | 0.8889 | 1.0000 | 0.0 |
| hybrid alpha=0.3 | 1.0000 | 0.9754 | 0.9667 | 1.0000 | 0.0 |



## BM25 grid search (`scripts/tune_bm25.py`)

Search space: `k1 in [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]` x `b in [0.25, 0.5, 0.75, 1.0]` (28 configs, 3s on the 50-query golden set).

Top 5 by nDCG@10:

| k1 | b | Recall@5 | Recall@10 | nDCG@10 | MAP |
|---|---|---|---|---|---|
| 0.8 | 0.50 | 0.8600 | 0.8800 | 0.7844 | 0.7532 |
| 1.0 | 0.25 | 0.8600 | 0.8800 | 0.7801 | 0.7470 |
| 1.2 | 0.50 | 0.8800 | 0.8800 | 0.7781 | 0.7440 |
| 0.5 | 0.25 | 0.8400 | 0.8600 | 0.7770 | 0.7512 |
| 0.5 | 0.75 | 0.8400 | 0.8800 | 0.7746 | 0.7414 |

**Winner:** `k1=0.8, b=0.5`. Persisted in `app/config.py` and `.env.example`.


## Fusion grid search (`scripts/tune_fusion.py`)

Search space: `alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]` plus RRF baseline. Run on the 50-query golden set with `top_k=20` candidates after BM25 tuning (k1=0.8, b=0.5).

**RRF baseline (k=60):** nDCG@10=0.7783, MAP=0.7513, Recall@10=0.8600

Alpha sweep top 5 by nDCG@10:

| alpha | Recall@5 | Recall@10 | nDCG@10 | MAP |
|---|---|---|---|---|
| 0.3 | 0.8600 | 0.8800 | 0.7890 | 0.7589 |
| 0.0 | 0.8600 | 0.8800 | 0.7844 | 0.7532 |
| 0.5 | 0.8600 | 0.8600 | 0.7829 | 0.7583 |
| 0.2 | 0.8600 | 0.8800 | 0.7787 | 0.7452 |
| 0.1 | 0.8400 | 0.8800 | 0.7781 | 0.7445 |

**Winner:** alpha-weighted (alpha=0.3) → nDCG@10=0.7890, MAP=0.7589
