# gpu-vllm-l4-4b-w4a16

Migrated benchmark artifacts for **`google/gemma-4-E4B-it-qat-w4a16-ct`** served by **vLLM** on a single **NVIDIA L4** (24 GB).

**This rig serves nothing** — no MCP server, no skill, no deployment path. It is a home for
measurements, not a project. See [`CLAUDE.md`](CLAUDE.md), which carries the provenance warning these
artifacts require.

## Runs

| Run | Date | Endpoint | CSV |
| --- | --- | --- | --- |
| `2026-07-12-vllm-grid-gce-l4` | 2026-07-12 | `http://34.31.68.246:8080` | yes |

Each `REPORT.md` is a 2D grid — context length x concurrent users — of average latency and throughput.

## Notes

`MODELS.md` puts E4B at 14.9 GiB of bf16 weights and 56 KiB/token of KV — 3.1x E2B. On a 24 GB L4 the 4-bit weights are what leave room for a usable KV pool.

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
