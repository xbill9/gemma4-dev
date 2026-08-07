# GCP TPU Zones Quota & Startup Status

This file details the GCP zones with available quota for `TPUV6EPerProjectPerZoneForTPUAPI` and the status of TPU startup attempts for `v6e-1`.

This is mutable state, not documentation: `find_tpu` rewrites it in place to record which zones have failed,
and reads it back to skip known-bad zones. A zone is skipped only when its third column is exactly `No`, and
only when the `[model]` tag in the detail column matches the provisioning model being attempted.

Reseeded on 2026-08-07 from a live scan of the v6e quota when this rig was retargeted from v5e-1 to v6e-1.
Every row of the previous table was an observation about `v5litepod-1` — including the two europe-west4
FLEX_START rejections and the one us-west4-a success — and none of it is evidence about v6e, which is a
different accelerator type metered by a different quota. us-west4 does not appear below at all: it has no
v6e quota, no v6e-1 accelerator type, and no v6e SKU.

Quota is not availability. A non-zero limit only means creation is permitted; the zone still has to have
v6e-1 hardware and grant the capacity. Of the zones listed here, the billing catalog publishes a flex-start
(`DWS Defined Duration V6e`) rate in only eight regions — us-east5, us-central1, us-south1, europe-west4,
asia-northeast1, asia-southeast1, southamerica-east1, and southamerica-west1.

| Zone | Quota Available | TPU v6e-1 Started Successfully | Details / Reason for Failure |
| :--- | :--- | :--- | :--- |
| **asia-east1-a** | Yes | Not attempted | — |
| **asia-east1-b** | Yes | Not attempted | — |
| **asia-east1-c** | Yes | Not attempted | — |
| **asia-northeast1-a** | Yes | Not attempted | — |
| **asia-northeast1-b** | Yes | Not attempted | — |
| **asia-northeast1-c** | Yes | Not attempted | — |
| **asia-south1-a** | Yes | Not attempted | — |
| **asia-south1-b** | Yes | Not attempted | — |
| **asia-south1-c** | Yes | Not attempted | — |
| **asia-southeast1-a** | Yes | Not attempted | — |
| **asia-southeast1-b** | Yes | Not attempted | — |
| **asia-southeast1-c** | Yes | Not attempted | — |
| **europe-west4-a** | Yes | Not attempted | — |
| **europe-west4-b** | Yes | Not attempted | — |
| **europe-west4-c** | Yes | Not attempted | — |
| **southamerica-east1-a** | Yes | Not attempted | — |
| **southamerica-east1-b** | Yes | Not attempted | — |
| **southamerica-east1-c** | Yes | Not attempted | — |
| **southamerica-west1-a** | Yes | Not attempted | — |
| **southamerica-west1-b** | Yes | Not attempted | — |
| **southamerica-west1-c** | Yes | Not attempted | — |
| **us-central1-a** | Yes | Not attempted | — |
| **us-central1-b** | Yes | Not attempted | — |
| **us-central1-c** | Yes | Not attempted | — |
| **us-central1-f** | Yes | Not attempted | — |
| **us-east1-b** | Yes | Not attempted | — |
| **us-east1-c** | Yes | Not attempted | — |
| **us-east1-d** | Yes | Not attempted | — |
| **us-east4-c** | Yes | Not attempted | — |
| **us-east5-a** | Yes | Not attempted | — |
| **us-east5-b** | Yes | Not attempted | — |
| **us-south1-a** | Yes | Not attempted | — |
| **us-south1-b** | Yes | Not attempted | — |
| **us-south1-c** | Yes | Not attempted | — |
| **us-west1-a** | Yes | Not attempted | — |
| **us-west1-b** | Yes | Not attempted | — |
| **us-west1-c** | Yes | Not attempted | — |
