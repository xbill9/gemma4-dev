# gpu-vllm-l4-12b-w4a16

Migrated benchmark artifacts for **`google/gemma-4-12B-it-qat-w4a16-ct`** served by **vLLM** on a single **NVIDIA L4** (24 GB).

**This rig serves nothing** — no MCP server, no skill, no deployment path. It is a home for
measurements, not a project. See [`CLAUDE.md`](CLAUDE.md), which carries the provenance warning these
artifacts require.

## Runs

| Run | Date | Endpoint | CSV |
| --- | --- | --- | --- |
| `2026-06-09-vllm-grid-cloudrun-l4` | 2026-06-09 | `https://gpu-12b-qat-l4-devops-agent-wgcq55zbfq-uk.a.run.app` | **none — see CLAUDE.md** |
| `2026-06-15-vllm-grid-ec2-l4` | 2026-06-15 | `http://44.204.128.2:8080` | yes |
| `2026-06-15-vllm-grid-gce-l4` | 2026-06-15 | `http://34.82.63.29:8080` | yes |
| `2026-06-21-vllm-grid-mtp-cloudrun-l4` | 2026-06-21 | `https://gpu-12b-qat-mtp-wgcq55zbfq-uk.a.run.app` | yes |

Each `REPORT.md` is a 2D grid — context length x concurrent users — of average latency and throughput.

## Notes

The four grids span three hosts, but they are **not** a clean host comparison — different dates, different endpoints, different grid sizes (21 rows for the MTP run against 145-157 for the rest). Nothing in the artifacts says what the MTP build was.

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
