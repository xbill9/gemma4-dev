## Embedding alone (generator loaded, idle)

| requests | input tokens | req/s | input tok/s | median latency ms |
|---:|---:|---:|---:|---:|
| 1 | 128 | 27.4 | 3,506 | 36.2 |
| 16 | 128 | 229.4 | 29,361 | 66.5 |
| 128 | 128 | 828.7 | 106,074 | 130.8 |
| 1 | 1024 | 26.7 | 27,312 | 37.0 |
| 16 | 1024 | 178.7 | 182,973 | 84.4 |
| 128 | 1024 | 216.4 | 221,616 | 577.6 |
| 1 | 4096 | 21.5 | 88,127 | 46.7 |
| 16 | 4096 | 41.8 | 171,406 | 379.1 |
| 128 | 4096 | 41.6 | 170,401 | 3,068.3 |

## Generator (12B fp8) alone and under a steady 128-way, 1,024-token embedding load

| requests | alone out tok/s | under load | ratio | median TPOT alone / under load ms |
|---:|---:|---:|---:|---|
| 8 | 855 | 240 | 0.28 | 8.2 / 33.2 |
| 64 | 2,745 | 1,436 | 0.52 | 20.7 / 44.1 |

## Embedding alone and under a steady 64-way generation load (1,024-token inputs)

| requests | alone req/s | under load | ratio | median latency alone / under load ms |
|---:|---:|---:|---:|---|
| 1 | 26.7 | 0.7 | 0.03 | 37 / 1,267 |
| 16 | 178.7 | 65.2 | 0.36 | 84 / 245 |
| 128 | 216.4 | 114.2 | 0.53 | 578 / 1,228 |

Share of each server's alone throughput kept at the heaviest load: generator c64 0.52, embedder c128 0.53, sum 1.05

Overlap of background load and measurement (from result timestamps): {
 "gen-c8-under-embed": "True 2026-10-09 16:31:12.353032 2026-10-09 16:39:15 2026-10-09 16:31:26.741287 2026-10-09 16:32:35",
 "gen-c64-under-embed": "True 2026-10-09 16:31:12.353032 2026-10-09 16:39:15 2026-10-09 16:32:47.746036 2026-10-09 16:34:19"
}
