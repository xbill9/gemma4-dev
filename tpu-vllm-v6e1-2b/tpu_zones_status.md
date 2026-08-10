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

## Three separate gates, established 2026-08-10

A creation has to clear all three, and they fail differently. **Quota is the weakest signal of the three.**

**1. Does the zone have `v6e-1` hardware at all?** 37 zones report quota; only **18** offer the accelerator
type, per `gcloud compute tpus accelerator-types list --filter="type=v6e-1"`. This is model-independent —
it applies to spot and on-demand exactly as much as to flex-start, which is why it is recorded here in
prose and not as per-model `No` rows below.

> **Has `v6e-1`:** us-east5-a/b/c, us-central1-a/b/c, us-south1-a/c, us-west1-c, us-east1-d,
> europe-west4-a, asia-northeast1-b, asia-southeast1-b, asia-east1-c, asia-south1-b/c,
> southamerica-west1-a, southamerica-east1-c.
>
> **Quota but NO hardware:** us-central1-f, us-south1-b, us-west1-a/b, us-east1-b/c, us-east4-c,
> europe-west4-b/c, asia-northeast1-a/c, asia-east1-a/b, asia-south1-a, asia-southeast1-a/c,
> southamerica-east1-a/b, southamerica-west1-b/c.

**Google's own docs undercount this.** The regions-zones page names 8 zones for v6e; the API accepts 18.
Prefer the API — the docs table is a strict subset.

**2. Does the zone offer that provisioning model for that accelerator type?** Independent of both quota and
hardware, and this is where a published rate stops meaning anything. **us-central1-b and us-south1-a both
have v6e-1 hardware, quota, and a published `DWS Defined Duration V6e` rate for their region — and both
reject flex-start at the API**: `FLEX_START provisioning model is not supported for accelerator type
"v6e-1" in location "<zone>"`. This is the v6e analogue of the v5e result where flex-start `v5litepod-1`
was accepted in exactly one zone out of 44. It is recorded per-model in the table below.

**3. Is there free capacity right now?** Only reachable after the first two pass, and it is the one gate
that is not a property of the zone — it changes minute to minute and is never cached here. us-east5-b
produced no v6e-1 under **either** spot or flex-start across ~70 minutes on 2026-08-10, while accepting
both requests without error. `WAITING_FOR_RESOURCES` is this gate, and it is not a failure.

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
| **europe-west4-a** | Yes | Pending | [flex-start] Accepted; WAITING_FOR_RESOURCES as of 2026-08-10 — no capacity yet, not a rejection. |
| **europe-west4-b** | Yes | Not attempted | — |
| **europe-west4-c** | Yes | Not attempted | — |
| **southamerica-east1-a** | Yes | Not attempted | — |
| **southamerica-east1-b** | Yes | Not attempted | — |
| **southamerica-east1-c** | Yes | Not attempted | — |
| **southamerica-west1-a** | Yes | Not attempted | — |
| **southamerica-west1-b** | Yes | Not attempted | — |
| **southamerica-west1-c** | Yes | Not attempted | — |
| **us-central1-a** | Yes | Not attempted | — |
| **us-central1-b** | Yes | No | [flex-start] FLEX_START not supported for v6e-1 in this location (API, 2026-08-10). Hardware and quota both present; the provisioning model is the blocker. Says nothing about spot. |
| **us-central1-c** | Yes | Not attempted | — |
| **us-central1-f** | Yes | Not attempted | — |
| **us-east1-b** | Yes | Not attempted | — |
| **us-east1-c** | Yes | Not attempted | — |
| **us-east1-d** | Yes | Not attempted | — |
| **us-east4-c** | Yes | Not attempted | — |
| **us-east5-a** | Yes | Pending | [flex-start] Accepted; WAITING_FOR_RESOURCES as of 2026-08-10 — no capacity yet, not a rejection. |
| **us-east5-b** | Yes | Pending | [flex-start] Accepted; WAITING_FOR_RESOURCES as of 2026-08-10 — no capacity yet, not a rejection. |
| **us-south1-a** | Yes | No | [flex-start] FLEX_START not supported for v6e-1 in this location (API, 2026-08-10). Hardware and quota both present; the provisioning model is the blocker. Says nothing about spot. |
| **us-south1-b** | Yes | Not attempted | — |
| **us-south1-c** | Yes | Not attempted | — |
| **us-west1-a** | Yes | Not attempted | — |
| **us-west1-b** | Yes | Not attempted | — |
| **us-west1-c** | Yes | Not attempted | — |
