# GCP TPU Zones Quota & Startup Status

This file details the GCP zones with available quota for `TPUV5sLitepodPerProjectPerZoneForTPUAPI` and the status of TPU startup attempts for `v5litepod-1` (v5e-1).

This is mutable state, not documentation: `find_tpu` rewrites it in place to record which zones have failed,
and reads it back to skip known-bad zones. A zone is skipped only when its third column is exactly `No`.

Reset on 2026-08-04, reseeded from a live quota scan. The previous table was cleared because every creation
attempt behind it had been failing on a hardcoded `--network=vpc-glitnir`, a VPC that does not exist in this
project — those failures said nothing about zone capacity. Quota is not availability: a non-zero limit only
means creation is permitted, and Flex-start capacity still has to be granted.

| Zone | Quota Available | TPU v5e-1 Started Successfully | Details / Reason for Failure |
| :--- | :--- | :--- | :--- |
| **asia-east1-a** | Yes | Not attempted | — |
| **asia-east1-b** | Yes | Not attempted | — |
| **asia-east1-c** | Yes | Not attempted | — |
| **asia-northeast1-a** | Yes | Not attempted | — |
| **asia-northeast1-b** | Yes | Not attempted | — |
| **asia-northeast1-c** | Yes | Not attempted | — |
| **asia-southeast1-a** | Yes | Not attempted | — |
| **asia-southeast1-b** | Yes | Not attempted | — |
| **asia-southeast1-c** | Yes | Not attempted | — |
| **europe-north1-a** | Yes | Not attempted | — |
| **europe-north1-b** | Yes | Not attempted | — |
| **europe-north1-c** | Yes | Not attempted | — |
| **europe-west1-b** | Yes | Not attempted | — |
| **europe-west1-c** | Yes | Not attempted | — |
| **europe-west1-d** | Yes | Not attempted | — |
| **europe-west3-a** | Yes | Not attempted | — |
| **europe-west3-b** | Yes | Not attempted | — |
| **europe-west3-c** | Yes | Not attempted | — |
| **europe-west4-a** | Yes | No | FLEX_START not supported for v5litepod-1 in this location (API rejected at create). Quota is fine; the provisioning model is the blocker. |
| **europe-west4-b** | Yes | No | FLEX_START not supported for v5litepod-1 in this location (API rejected at create). |
| **europe-west4-c** | Yes | Not attempted | — |
| **northamerica-northeast1-a** | Yes | Not attempted | — |
| **northamerica-northeast1-b** | Yes | Not attempted | — |
| **northamerica-northeast1-c** | Yes | Not attempted | — |
| **southamerica-west1-a** | Yes | Not attempted | — |
| **southamerica-west1-b** | Yes | Not attempted | — |
| **southamerica-west1-c** | Yes | Not attempted | — |
| **us-central1-a** | Yes | Not attempted | — |
| **us-central1-b** | Yes | Not attempted | — |
| **us-central1-c** | Yes | Not attempted | — |
| **us-central1-f** | Yes | Not attempted | — |
| **us-east1-b** | Yes | Not attempted | — |
| **us-east1-d** | Yes | Not attempted | — |
| **us-east4-a** | Yes | Not attempted | — |
| **us-east4-b** | Yes | Not attempted | — |
| **us-east4-c** | Yes | Not attempted | — |
| **us-south1-a** | Yes | Not attempted | — |
| **us-south1-b** | Yes | Not attempted | — |
| **us-south1-c** | Yes | Not attempted | — |
| **us-west1-a** | Yes | Not attempted | — |
| **us-west1-b** | Yes | Not attempted | — |
| **us-west1-c** | Yes | Not attempted | — |
| **us-west4-a** | Yes | Provisioning | FLEX_START accepted for v5litepod-1 here. Created 2026-08-04, reached PROVISIONING. |
| **us-west4-c** | Yes | Not attempted | — |
