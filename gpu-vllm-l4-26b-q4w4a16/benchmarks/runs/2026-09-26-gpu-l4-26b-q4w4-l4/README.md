# 2026-09-26-gpu-l4-26b-q4w4-l4

One arm of the `2026-09-26-gpu-l4` run of the sweep in `../../../../jev-tpu-31b/`, filed here because
this rig names its chip and checkpoint.

| | |
|---|---|
| Checkpoint | `/work/models/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct` |
| Hardware | `l4` |
| Arm | `26b-q4w4` (harness id `2026-09-26-gpu-l4-26b-q4w4`) |
| Read / latency / smoke / suite | `2026-09-26-gpu-l4-26b-q4w4{,-latency,-smoke,-suite}/` (those present) |
| This arm's logs | `logs/<run log dir>/` |
| Run-wide logs (run.log, patches, checksums) | `../../../../jev-tpu-31b/results/2026-09-26-gpu-l4-evidence/` |

Paired comparisons that include this run (they cover two cells, so they stay with the sweep):

- `../../../../jev-tpu-31b/results/2026-09-26-gpu-l4-VS-BF16.md`
- `../../../../jev-tpu-31b/results/2026-09-26-gpu-l4-VS-TPU.md`
