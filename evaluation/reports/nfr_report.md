# NFR benchmark — Runtime MVP

API URL: `http://localhost:8000`
Golden set: `/home/hehek/AI-testing/gigaschool/garag/data/golden/golden_set_v1.jsonl`
Health: `{"status": "ok", "pipeline_loaded": true, "version": "0.1.0"}`

## Summary

| Metric | Value | Target | Status |
|---|---:|---:|---|
| p95 e2e latency (warm) | 2753 ms | ≤ 8000 ms | PASS |
| Throughput | 0.615 RPS | ≥ 2.0 RPS | FAIL |
| Indexing time | 40.8 s | ≤ 1200 s | PASS |

## Latency Phase

| count | success | failures | mean ms | p50 ms | p95 ms | min ms | max ms | rps |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 20 | 0 | 1954.1 | 2113.4 | 2753.3 | 1120.1 | 2753.3 | 0.000 |

## Throughput Phase

| count | success | failures | mean ms | p50 ms | p95 ms | min ms | max ms | rps |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 20 | 0 | 3210.1 | 3298.7 | 5155.6 | 2062.1 | 5155.6 | 0.615 |

## Indexing Phase

- Command: `uv run python -m scripts.build_qdrant`
- Elapsed: 40.8 s
- Status: PASS

## Per-query Latency

| qid | ok | status | latency ms | stage total ms | error |
|---|---|---:|---:|---:|---|
| `g001` | yes | 200 | 2113.4 | 2112.1 |  |
| `g002` | yes | 200 | 1410.1 | 1408.8 |  |
| `g003` | yes | 200 | 1517.7 | 1516.4 |  |
| `g004` | yes | 200 | 1528.1 | 1526.9 |  |
| `g005` | yes | 200 | 1142.4 | 1141.1 |  |
| `g006` | yes | 200 | 1778.2 | 1777.0 |  |
| `g007` | yes | 200 | 2192.1 | 2190.8 |  |
| `g008` | yes | 200 | 1228.4 | 1227.1 |  |
| `g009` | yes | 200 | 1120.1 | 1118.9 |  |
| `g010` | yes | 200 | 1124.8 | 1123.5 |  |
| `g011` | yes | 200 | 2542.4 | 2541.0 |  |
| `g012` | yes | 200 | 2255.5 | 2254.2 |  |
| `g013` | yes | 200 | 2502.7 | 2501.1 |  |
| `g014` | yes | 200 | 2730.2 | 2728.8 |  |
| `g015` | yes | 200 | 2753.3 | 2752.0 |  |
| `g016` | yes | 200 | 2646.9 | 2645.6 |  |
| `g017` | yes | 200 | 1919.6 | 1918.2 |  |
| `g018` | yes | 200 | 2294.1 | 2292.8 |  |
| `g019` | yes | 200 | 1995.8 | 1994.4 |  |
| `g020` | yes | 200 | 2287.0 | 2285.7 |  |

## Per-query Throughput

| qid | ok | status | latency ms | stage total ms | error |
|---|---|---:|---:|---:|---|
| `g001` | yes | 200 | 3298.7 | 3296.3 |  |
| `g002` | yes | 200 | 2080.1 | 2078.2 |  |
| `g003` | yes | 200 | 2602.5 | 2600.7 |  |
| `g004` | yes | 200 | 2746.2 | 2744.1 |  |
| `g005` | yes | 200 | 2200.4 | 2198.3 |  |
| `g006` | yes | 200 | 2379.6 | 2377.7 |  |
| `g007` | yes | 200 | 3345.8 | 3343.7 |  |
| `g008` | yes | 200 | 2290.4 | 2288.5 |  |
| `g009` | yes | 200 | 2129.9 | 2127.8 |  |
| `g010` | yes | 200 | 2062.1 | 2060.0 |  |
| `g011` | yes | 200 | 3773.4 | 3771.6 |  |
| `g012` | yes | 200 | 3477.5 | 3475.6 |  |
| `g013` | yes | 200 | 4150.8 | 4145.4 |  |
| `g014` | yes | 200 | 4982.7 | 4973.9 |  |
| `g015` | yes | 200 | 5155.6 | 5154.1 |  |
| `g016` | yes | 200 | 3625.0 | 3623.6 |  |
| `g017` | yes | 200 | 3731.9 | 3730.5 |  |
| `g018` | yes | 200 | 4100.5 | 4098.5 |  |
| `g019` | yes | 200 | 3139.2 | 3137.7 |  |
| `g020` | yes | 200 | 2929.5 | 2928.1 |  |
