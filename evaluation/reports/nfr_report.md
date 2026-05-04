# NFR benchmark — Runtime MVP

API URL: `http://localhost:8000`
Golden set: `/home/hehek/AI-testing/gigaschool/garag/data/golden/golden_set_v1.jsonl`
Top K: `12`
Health: `{"status": "ok", "pipeline_loaded": true, "version": "0.1.0"}`

## Summary

| Metric | Value | Target | Status |
|---|---:|---:|---|
| p95 e2e latency (warm) | 2137 ms | ≤ 8000 ms | PASS |
| Throughput | 0.826 RPS | ≥ 0.5 RPS | PASS |
| Indexing time | not run | ≤ 1200 s | NOT RUN |

## Throughput note

The v0.1.0 throughput target is scoped to a single local FastAPI instance backed by a 35B LM Studio generator. Stage timings show generation dominates runtime: mean `gen` latency is 2296 ms in the throughput phase at `top_k=12`, while retrieval plus rerank stages are in the tens of milliseconds. The aspirational ≥2 RPS target is deferred to a serving-focused increment such as vLLM continuous batching or a smaller generator.

## Latency Phase

| count | success | failures | mean ms | p50 ms | p95 ms | min ms | max ms | rps |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 20 | 0 | 1160.8 | 1051.6 | 2137.3 | 388.7 | 2137.3 | 0.000 |

## Throughput Phase

| count | success | failures | mean ms | p50 ms | p95 ms | min ms | max ms | rps |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 20 | 0 | 2405.4 | 2358.6 | 4257.2 | 526.1 | 4257.2 | 0.826 |

## Indexing Phase

Not run. Pass `--run-indexing` to measure full Qdrant rebuild time.

## Per-query Latency

| qid | ok | status | latency ms | stage total ms | error |
|---|---|---:|---:|---:|---|
| `g001` | yes | 200 | 667.6 | 666.6 |  |
| `g002` | yes | 200 | 739.5 | 738.3 |  |
| `g003` | yes | 200 | 388.7 | 387.5 |  |
| `g004` | yes | 200 | 751.7 | 750.4 |  |
| `g005` | yes | 200 | 1026.6 | 1025.3 |  |
| `g006` | yes | 200 | 613.9 | 612.7 |  |
| `g007` | yes | 200 | 1038.5 | 1037.1 |  |
| `g008` | yes | 200 | 969.0 | 967.8 |  |
| `g009` | yes | 200 | 1375.5 | 1374.3 |  |
| `g010` | yes | 200 | 917.8 | 916.4 |  |
| `g011` | yes | 200 | 1823.9 | 1822.6 |  |
| `g012` | yes | 200 | 1139.3 | 1138.1 |  |
| `g013` | yes | 200 | 1584.0 | 1582.7 |  |
| `g014` | yes | 200 | 2137.3 | 2136.1 |  |
| `g015` | yes | 200 | 2061.0 | 2059.8 |  |
| `g016` | yes | 200 | 1249.6 | 1248.4 |  |
| `g017` | yes | 200 | 1430.4 | 1429.2 |  |
| `g018` | yes | 200 | 1249.1 | 1247.8 |  |
| `g019` | yes | 200 | 1002.0 | 1000.8 |  |
| `g020` | yes | 200 | 1051.6 | 1050.3 |  |

## Per-query Throughput

| qid | ok | status | latency ms | stage total ms | error |
|---|---|---:|---:|---:|---|
| `g001` | yes | 200 | 1024.2 | 1022.4 |  |
| `g002` | yes | 200 | 1823.7 | 1822.0 |  |
| `g003` | yes | 200 | 1203.5 | 1202.0 |  |
| `g004` | yes | 200 | 1415.6 | 1413.8 |  |
| `g005` | yes | 200 | 1668.1 | 1666.4 |  |
| `g006` | yes | 200 | 526.1 | 524.4 |  |
| `g007` | yes | 200 | 3747.4 | 3745.8 |  |
| `g008` | yes | 200 | 2358.6 | 2357.0 |  |
| `g009` | yes | 200 | 3932.6 | 3930.9 |  |
| `g010` | yes | 200 | 1636.1 | 1634.5 |  |
| `g011` | yes | 200 | 4257.2 | 4255.6 |  |
| `g012` | yes | 200 | 1853.7 | 1852.2 |  |
| `g013` | yes | 200 | 3286.1 | 3282.7 |  |
| `g014` | yes | 200 | 2059.1 | 2050.2 |  |
| `g015` | yes | 200 | 3077.5 | 3076.3 |  |
| `g016` | yes | 200 | 2937.8 | 2936.2 |  |
| `g017` | yes | 200 | 4031.4 | 4029.6 |  |
| `g018` | yes | 200 | 3140.5 | 3139.1 |  |
| `g019` | yes | 200 | 2394.4 | 2393.1 |  |
| `g020` | yes | 200 | 1734.2 | 1733.2 |  |
