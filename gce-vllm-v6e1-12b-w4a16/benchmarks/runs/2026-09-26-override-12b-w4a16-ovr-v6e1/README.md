# 2026-09-26-override-12b-w4a16-ovr-v6e1

One arm of the `2026-09-26-override` run of the sweep in `../../../../jev-tpu-31b/`, filed here because
this rig names its chip and checkpoint.

| | |
|---|---|
| Checkpoint | `google/gemma-4-12B-it-qat-w4a16-ct` |
| Hardware | `v6e1` |
| Arm | `12b-w4a16-ovr` (harness id `2026-09-26-override-12b-w4a16-ovr`) |
| Read / latency / smoke / suite | `2026-09-26-override-12b-w4a16-ovr{,-latency,-smoke,-suite}/` (those present) |
| This arm's logs | `logs/<run log dir>/` |
| Run-wide logs (run.log, patches, checksums) | `../../../../jev-tpu-31b/results/2026-09-26-override-evidence/` |

Paired comparisons that include this run (they cover two cells, so they stay with the sweep):

- `../../../../jev-tpu-31b/results/2026-09-26-override-VS-PYTORCH.md`
- `../../../../jev-tpu-31b/results/2026-09-26-override-VS-UNIFIED.md`
