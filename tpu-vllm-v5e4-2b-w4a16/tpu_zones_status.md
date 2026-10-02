# GCP TPU Zones Quota & Startup Status

This file details the GCP zones with available quota for `TPUV5sLitepodPerProjectPerZoneForTPUAPI` and the status of TPU startup attempts for `v5litepod-4` (v5e-4).

This is mutable state, not documentation: `find_tpu` rewrites it in place to record which zones have failed,
and reads it back to skip known-bad zones. A zone is skipped only when its third column is exactly `No`.

Started empty on 2026-10-02 when this rig was forked from its v5e-1 sibling. The sibling's rows recorded
`v5litepod-1` attempts, and a zone that refused one accelerator type says nothing about another, so none
were carried over. Quota is not availability: a non-zero limit only means creation is permitted, and
Flex-start capacity still has to be granted.

- **Successful Zone:** none yet

| Zone | Quota Available | TPU v5e-4 Started Successfully | Details / Reason for Failure |
| :--- | :--- | :--- | :--- |
