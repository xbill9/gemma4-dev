# gpu-vllm-l4-2b-w4a16

Migrated benchmark artifacts for **`google/gemma-4-E2B-it-qat-w4a16-ct`** served by **vLLM** on a single **NVIDIA L4** (24 GB).

**This rig serves nothing** — no MCP server, no skill, no deployment path. It is a home for
measurements, not a project. See [`CLAUDE.md`](CLAUDE.md), which carries the provenance warning these
artifacts require.

## Runs

| Run | Date | Endpoint | CSV |
| --- | --- | --- | --- |
| `2026-07-10-vllm-grid-gce-l4` | 2026-07-10 | `http://35.199.2.239:8080` | yes |

Each `REPORT.md` is a 2D grid — context length x concurrent users — of average latency and throughput.

## Notes

E2B is the only size here that also fits an L4 at bf16, so it is the one place a QAT-vs-bf16 comparison on L4 would mean something. It was not run.

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
