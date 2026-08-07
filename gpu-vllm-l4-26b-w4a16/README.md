# gpu-vllm-l4-26b-w4a16

Migrated benchmark artifacts for **`gemma-4-26B-A4B-it-qat-w4a16-ct`** served by **vLLM** on a single **NVIDIA L4** (24 GB).

**This rig serves nothing** — no MCP server, no skill, no deployment path. It is a home for
measurements, not a project. See [`CLAUDE.md`](CLAUDE.md), which carries the provenance warning these
artifacts require.

## Runs

| Run | Date | Endpoint | CSV |
| --- | --- | --- | --- |
| `2026-06-10-vllm-grid-cloudrun-l4` | 2026-06-10 | `https://gpu-26b-qat-l4-devops-agent-wgcq55zbfq-uk.a.run.app` | yes |
| `2026-07-12-vllm-grid-gce-l4` | 2026-07-12 | `http://35.243.250.174:8080` | yes |

Each `REPORT.md` is a 2D grid — context length x concurrent users — of average latency and throughput.

## Notes

**The encoding slot here contradicts `QUANTIZATION.md`**, which records that no `-w4a16-ct` release exists for this size. Both reports name it as a local mount path, not a Hub id. See `CLAUDE.md` — the short version is that a 26B cannot fit a 24 GB L4 at bf16, so the weights were 4-bit, most likely a local repack of `-q4_0-unquantized`.

## Before you cite a number

- These are **single-run** grids: no repeat count, no variance, on shared-tenancy hosts. One cell is
  one sample.
- **Ignore the `users=1` column.** Several grids show a multi-second latency at concurrency 1 that
  disappears at concurrency 2 — cold start, not a model property.
- The host label in parentheses after `Model:` is unreliable; the `Endpoint:` line is what the run
  directory names were resolved from.
- The source tree these came from duplicated artifacts across directories on a large scale. Only
  self-identifying reports were migrated. **Do not go back to `~/gemma4-tips` and read a model or a
  chip off a directory name.**
