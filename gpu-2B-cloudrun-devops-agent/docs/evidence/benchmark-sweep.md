# cloudrun_run_benchmark (defaults) — MCP tool output, captured 2026-09-10 ~11:57 EDT via Claude Code, service gpu-2b-l4-devops-agent revision 00002-t5v (manual scaling, 1 instance)
### 📊 GPU Benchmark Results (Model: `/mnt/models/gemma-4-E2B-it`)

| Concurrency | Success Rate | Req/s | Tokens/s | Avg Latency | P95 Latency |
|---:|---:|---:|---:|---:|---:|
| 1 | 100.0% | 0.39 | 49.63 | 2.58s | 2.59s |
| 2 | 100.0% | 0.75 | 95.36 | 2.68s | 2.74s |
| 4 | 100.0% | 1.47 | 188.46 | 2.71s | 2.78s |
| 8 | 100.0% | 1.48 | 189.53 | 4.86s | 5.47s |

# arithmetic used in the article:
# 188.46 / 49.63 = 3.797  -> "3.8x"
# 189.53 / 188.46 = 1.0057 -> "+0.6%"
