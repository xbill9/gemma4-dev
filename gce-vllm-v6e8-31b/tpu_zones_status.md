# TPU zone status — Compute Engine path

**Mutable state, not documentation.** `find_tpu` rewrites this file in place to record which
zones failed, and reads it back to skip known-bad ones. Do not hand-edit it as if it were docs.

Four things about a row here are easy to get wrong:

- **A row here is evidence about Compute Engine, not about the Cloud TPU API.** The two
  control planes have separate quota pools and separate acceptance rules, so a zone that
  rejects a Queued Resource says nothing about an instance create, and vice versa. Never
  seed this file from the twin rig's.
- **A row is evidence about an EIGHT-CHIP request.** This rig asks for `ct6e-standard-8t`.
  A zone that granted one chip to `gce-vllm-v6e1-2b` is not a zone that will grant eight —
  capacity is per slice, not per region — so never seed this file from that rig's, in either
  direction.
- **A row is NOT evidence about the checkpoint, which is the one thing that would be safe to
  copy.** This rig was forked from `gce-vllm-v6e8-2b` and serves 31B where that one serves
  E2B; the capacity request is byte-identical. Its rows would in principle transfer — and the
  file is empty there too, so there is nothing to copy and the question has not arisen. If it
  ever does, the discriminator is the machine type and the provisioning model, not the model
  being served.
- **The `[model]` prefix in the detail column is this rig's lowercase label** (`flex-start`,
  `spot`, `on-demand`, `reservation-bound`), not gcloud's SCREAMING_CASE value. `find_tpu`
  only skips a zone whose recorded failure was under the *same* model.

Empty on purpose: **this rig has provisioned nothing and attempted nothing.** Every row below
must come from a real attempt. Capacity outcomes (`WAITING_FOR_RESOURCES`, a client-side
timeout with the request still live) are gate 3 and must never be recorded here as failures.

| Zone | Quota | Worked | Detail |
| --- | --- | --- | --- |
