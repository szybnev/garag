# NFR benchmark — Runtime MVP

API URL: `http://localhost:8000`
Golden set: `/home/hehek/AI-testing/gigaschool/garag/data/golden/golden_set_v1.jsonl`
Top K: `3`
Health: `{"status": "ok", "pipeline_loaded": true, "version": "0.1.0"}`

## Summary

| Metric | Value | Target | Status |
|---|---:|---:|---|
| p95 e2e latency (warm) | 3071 ms | ≤ 8000 ms | PASS |
| Throughput | 0.585 RPS | ≥ 0.5 RPS | PASS |
| Indexing time | 42.7 s | ≤ 1200 s | PASS |

## Throughput note

The v0.1.0 throughput target is scoped to a single local FastAPI instance backed by a 35B LM Studio generator. Stage timings show generation dominates runtime: mean `gen` latency is 3241 ms in the throughput phase at `top_k=3`, while retrieval plus rerank stages are in the tens of milliseconds. The aspirational ≥2 RPS target is deferred to a serving-focused increment such as vLLM continuous batching or a smaller generator.

## Latency Phase

| count | success | failures | mean ms | p50 ms | p95 ms | min ms | max ms | rps |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 20 | 0 | 1964.5 | 2025.4 | 3071.3 | 1174.4 | 3071.3 | 0.000 |

## Throughput Phase

| count | success | failures | mean ms | p50 ms | p95 ms | min ms | max ms | rps |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 20 | 0 | 3333.5 | 3197.6 | 4932.3 | 2125.3 | 4932.3 | 0.585 |

## Indexing Phase

- Command: `uv run python -m scripts.build_qdrant`
- Elapsed: 42.7 s
- Status: PASS

## Per-query Latency

| qid | ok | status | latency ms | stage total ms | error |
|---|---|---:|---:|---:|---|
| `g001` | yes | 200 | 2177.7 | 2176.5 |  |
| `g002` | yes | 200 | 1409.5 | 1408.2 |  |
| `g003` | yes | 200 | 1701.3 | 1700.0 |  |
| `g004` | yes | 200 | 1179.4 | 1178.1 |  |
| `g005` | yes | 200 | 1245.4 | 1244.1 |  |
| `g006` | yes | 200 | 1268.9 | 1267.5 |  |
| `g007` | yes | 200 | 2896.8 | 2895.6 |  |
| `g008` | yes | 200 | 1174.4 | 1173.2 |  |
| `g009` | yes | 200 | 1188.7 | 1187.4 |  |
| `g010` | yes | 200 | 1226.6 | 1225.3 |  |
| `g011` | yes | 200 | 1935.0 | 1933.6 |  |
| `g012` | yes | 200 | 2025.4 | 2024.0 |  |
| `g013` | yes | 200 | 2349.2 | 2347.8 |  |
| `g014` | yes | 200 | 3071.3 | 3070.0 |  |
| `g015` | yes | 200 | 2926.4 | 2925.0 |  |
| `g016` | yes | 200 | 2983.5 | 2982.3 |  |
| `g017` | yes | 200 | 2208.6 | 2207.3 |  |
| `g018` | yes | 200 | 2191.6 | 2190.3 |  |
| `g019` | yes | 200 | 1764.3 | 1763.1 |  |
| `g020` | yes | 200 | 2367.0 | 2365.7 |  |

## Per-query Throughput

| qid | ok | status | latency ms | stage total ms | error |
|---|---|---:|---:|---:|---|
| `g001` | yes | 200 | 3572.5 | 3569.9 |  |
| `g002` | yes | 200 | 2618.1 | 2616.2 |  |
| `g003` | yes | 200 | 2389.7 | 2387.4 |  |
| `g004` | yes | 200 | 2140.6 | 2138.6 |  |
| `g005` | yes | 200 | 2269.9 | 2268.0 |  |
| `g006` | yes | 200 | 4111.0 | 4108.8 |  |
| `g007` | yes | 200 | 4253.2 | 4251.7 |  |
| `g008` | yes | 200 | 2899.6 | 2897.7 |  |
| `g009` | yes | 200 | 2125.3 | 2123.3 |  |
| `g010` | yes | 200 | 2284.0 | 2281.9 |  |
| `g011` | yes | 200 | 4932.3 | 4930.3 |  |
| `g012` | yes | 200 | 2302.0 | 2300.0 |  |
| `g013` | yes | 200 | 4641.6 | 4636.1 |  |
| `g014` | yes | 200 | 4268.5 | 4258.1 |  |
| `g015` | yes | 200 | 4669.2 | 4667.6 |  |
| `g016` | yes | 200 | 2322.8 | 2320.1 |  |
| `g017` | yes | 200 | 4166.1 | 4163.5 |  |
| `g018` | yes | 200 | 4568.9 | 4567.1 |  |
| `g019` | yes | 200 | 3197.6 | 3195.3 |  |
| `g020` | yes | 200 | 2936.4 | 2934.2 |  |
