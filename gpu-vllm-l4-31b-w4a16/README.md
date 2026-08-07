# gpu-vllm-l4-31b-w4a16

Migrated benchmark artifacts for **`google/gemma-4-31B-it-qat-w4a16-ct`** served by **vLLM** on a single **NVIDIA L4** (24 GB).

**This rig serves nothing** — no MCP server, no skill, no deployment path. It is a home for
measurements, not a project. See [`CLAUDE.md`](CLAUDE.md), which carries the provenance warning these
artifacts require.

## Runs

| Run | Date | Endpoint | CSV |
| --- | --- | --- | --- |
| `2026-06-09-vllm-grid-cloudrun-l4` | 2026-06-09 | `https://gpu-31b-qat-l4-devops-agent-wgcq55zbfq-uk.a.run.app` | **none — see CLAUDE.md** |
| `2026-07-12-vllm-grid-gce-l4` | 2026-07-12 | `http://34.62.246.100:8080` | yes |

Each `REPORT.md` is a 2D grid — context length x concurrent users — of average latency and throughput.

## Notes

These reports state the 31B at bf16 leaves **0 GB** for KV on a single L4 and destabilizes above concurrency 8. Same conclusion `MODELS.md` and `HARDWARE.md` reach for the TPU rigs by another route: on a 24 GB accelerator this is a quantize-or-do-not-run model.

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
