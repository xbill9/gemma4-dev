# GCP TPU Zones Quota & Startup Status

This file details the GCP zones with available quota for `TPU-V5P-per-project-zone` (on `compute.googleapis.com`) and the status
of TPU startup attempts for `ct5p-hightpu-1t-tpu` (one v5p chip, via Compute Engine).

This is mutable state, not documentation: `find_tpu` rewrites it in place to record which zones have failed,
and reads it back to skip known-bad zones. A zone is skipped only when its third column is exactly `No`.

**Reset on 2026-08-10** when this rig moved from v5e-1 to v5p. Every row behind the previous table described
`v5litepod-1`, a different accelerator on a different quota, so none of it was evidence about v5p.

Two independent limits are recorded below, and they are not the same thing:

- **Quota** — 10 zones carry a stated v5p limit (the TPU-API metric read 128 cores; this rig now reads the Compute Engine metric instead). Quota is
  a ceiling on what creation is *permitted*, not an offer of capacity. Those 10 are the only zones this table
  can ever contain, because `find_tpu` builds its sweep from the quota scan.
- **Hardware** — only **3** of those 10 publish the `v5p-8` accelerator type. The other 7 hold quota for a
  chip that is not installed there. Verified 2026-08-10 by
  `gcloud compute tpus accelerator-types list --zone=<zone> --filter="type=v5p-8"` in each of the 10.

**"3" is the intersection, not the size of the v5p fleet.** `v5p-8` is published in at least **nine** zones;
six of them (`europe-west1-b`, `-d`, `us-east1-d`, `us-east5-b`, `-c`, `us-south1-a`) carry no stated quota
for this project, so they never reach this table and `find_tpu` will never try them. They are a **quota
request** away, not a hardware limit — `us-east5-b` and `-c` sit in the same region as the rig's default zone
and are the obvious ask. `@HARDWARE.md` has the full breakdown.

A third limit sits on top and is not a column here: **flex-start is offered for v5p in `us-east5-a` only.**
`us-central1-a` and `europe-west4-b` have the hardware but must be reached with `spot` or `on-demand`.
Note a `DWS Defined Duration V5p` SKU *is* published for `us-central1` — a price existing is not capacity
being obtainable, which is the same trap the v5e rig hit in `europe-west4`.

Rows carry a `[model]` prefix in the detail column; `find_tpu` only skips a zone whose recorded failure was
under the *same* provisioning model.

| Zone | Quota Available | TPU v5p instance Started Successfully | Details / Reason for Failure |
| :--- | :--- | :--- | :--- |
| **us-east5-a** | Yes | Not attempted | Only v5p zone offering flex-start. Default zone for this rig. |
| **us-central1-a** | Yes | Not attempted | Hardware present. spot / on-demand only. |
| **europe-west4-b** | Yes | Not attempted | Hardware present. spot / on-demand only. |
| **europe-west4-a** | Yes | No | `v5p-8` not published in this zone (2026-08-10) — quota only |
| **europe-west4-c** | Yes | No | `v5p-8` not published in this zone (2026-08-10) — quota only |
| **us-central1-b** | Yes | No | `v5p-8` not published in this zone (2026-08-10) — quota only |
| **us-central1-c** | Yes | No | `v5p-8` not published in this zone (2026-08-10) — quota only |
| **us-central1-f** | Yes | No | `v5p-8` not published in this zone (2026-08-10) — quota only |
| **us-east1-b** | Yes | No | `v5p-8` not published in this zone (2026-08-10) — quota only |
| **us-east1-c** | Yes | No | `v5p-8` not published in this zone (2026-08-10) — quota only |
