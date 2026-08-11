# TPU zone status — Compute Engine path

**Mutable state, not documentation.** `find_tpu` rewrites this file in place to record which
zones failed, and reads it back to skip known-bad ones. Do not hand-edit it as if it were docs.

Two things differ from the sibling `tpu-vllm-v5e1-2b` file and are easy to get wrong:

- **A row here is evidence about Compute Engine, not about the Cloud TPU API.** The two
  control planes have separate quota pools and separate acceptance rules, so a zone that
  rejects a Queued Resource says nothing about an instance create, and vice versa. Never
  seed this file from the sibling rig's.
- **The `[model]` prefix in the detail column is this rig's lowercase label** (`flex-start`,
  `spot`, `on-demand`, `reservation-bound`), not gcloud's SCREAMING_CASE value. `find_tpu`
  only skips a zone whose recorded failure was under the *same* model.

Empty on purpose: **this rig has provisioned nothing and attempted nothing.** Every row below
must come from a real attempt. Capacity outcomes (`WAITING_FOR_RESOURCES`, a client-side
timeout with the request still live) are gate 3 and must never be recorded here as failures.

| Zone | Quota | Worked | Detail |
| --- | --- | --- | --- |
