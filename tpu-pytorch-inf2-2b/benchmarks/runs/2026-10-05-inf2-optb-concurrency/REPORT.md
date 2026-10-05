# inf2 Option B — concurrency sweep, 2026-10-05

`benchmark_results.csv` is the raw output of `run_benchmark` from the legacy
`~/gemma4-tips-aws/gpu-2B-inf-devops-agent` MCP server, run 2026-10-05 17:20–17:21 local time against
an inf2 instance in us-east-1.

| Concurrency | Requests | OK | Tok/s | Req/s | Avg latency | P95 latency |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 8 | 55.4 | 0.43 | 2.33 s | 2.33 s |
| 2 | 8 | 8 | 59.3 | 0.46 | 4.08 s | 4.49 s |
| 4 | 8 | 8 | 59.4 | 0.46 | 7.08 s | 8.78 s |

Method: one warm-up request, then 8 requests per level to `/v1/completions`, one fixed prompt,
`max_tokens` 128, temperature 0. Tokens are `usage.completion_tokens` summed over wall time.
P95 is the 8th of 8 sorted samples, so it is the slowest request.

What is known about the deployment:

- **Image:** the legacy server's deploy path pins `xbill9/gemma4-optb:tp2-slim` (TP=2, 2048 / 512,
  both NeuronCores). The run did not record the running container's tag.
- **Instance type: not recorded.** The instance was gone by the time this run was filed.
- Throughput stays at 59 tok/s from concurrency 2 to 4 while latency roughly doubles, so the server
  processes requests one at a time. The `tp2-slim` row in `nxd-gemma4-inf2-work/optb/DOCKER.md`
  quotes ~61 tok/s single-stream decode.

`run_benchmark` in this rig's `server.py` is the port of that tool. One difference: the legacy
tool counted a response with no `usage` block as `max_tokens` tokens; the port counts it as a
failed request.
