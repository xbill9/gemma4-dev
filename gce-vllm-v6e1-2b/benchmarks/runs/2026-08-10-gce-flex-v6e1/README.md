# 2026-08-10 — first provision and sweep on the Compute Engine path

**The first time this rig has provisioned anything or measured anything.** Until this run every
claim in its `CLAUDE.md` about how Compute Engine behaves was read off gcloud, the billing
catalog, and Google's docs, with nothing confirmed by a successful create.

## Provenance

`google/gemma-4-E2B-it` on TPU **v6e-1**, **`europe-west4-a`**, **flex-start**
(`--provisioning-model=FLEX_START`), machine type `ct6e-standard-1t`, image family
`ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e`, `vllm/vllm-tpu:nightly`
(`0.26.1rc1.dev256+gf5bb701fa`), TP=1, `max_model_len` **32768**, `max_num_batched_tokens` 4096,
`kv_cache_dtype=auto`, `enable_prefix_caching` default.

Instance `gce-vllm-v6e1-2b`, created 02:08 UTC, `--max-run-duration=4h`
`--instance-termination-action=DELETE`.

## Why europe-west4-a and not the configured zone

`tpu.env` defaults to `us-east5-b`. **That zone cannot run this rig**, and the reason is the exact
trap the rig was built to document — but the specifics recorded in `CLAUDE.md` were half right.

`TPUS-PER-TPU-FAMILY-per-project-region` on `compute.googleapis.com`, read 2026-08-10:

| region | CT6E |
| :--- | ---: |
| europe-west4 | **32** |
| asia-east1, asia-northeast1, asia-south1, asia-southeast1 | 32 |
| southamerica-east1, southamerica-west1, us-south1 | 32 |
| **us-east5** | **no stated value** |
| us-east1, us-west1 | no stated value |

So `CLAUDE.md` was right that us-east5 has no CT6E value, and **wrong** in the implication that
the project holds no Compute Engine v6e quota at all — it holds 32 chips in eight regions, one of
which (`europe-west4-a`) also publishes `ct6e-standard-1t`. The TPU-API quota this project holds
in us-east5 (512 chips) is genuinely unusable from this path, which is the finding that survives.

`europe-west4-a` is also the zone two of the sibling baseline's three result files were measured
in, which makes the comparison tighter rather than looser. See "What the baseline actually is".

## The flex-start create was granted immediately

No queueing. `gcloud compute instances create --provisioning-model=FLEX_START` returned inside
its 590s client cap and the instance was `RUNNING` within ~2 minutes. `--request-valid-for-duration=2h`
was never exercised, so this run says nothing about how the DWS queue behaves under contention.

Cost, from the live billing catalog rather than a table: flex-start v6e is **$1.35/chip-hr** in
both europe-west4 and us-east5. Note spot is *dearer* in both — $1.782 in europe-west4, $1.4033
in us-east5 — so `CLAUDE.md`'s "don't assume spot is cheapest" holds in a second region.

## The boot failed, and the cause is a real control-plane difference

**The Compute Engine accelerator image ships no Docker.** `ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e`
has no `docker` on PATH at first boot. The Cloud TPU API's runtime versions (`v2-alpha-tpuv6e`)
do, which is why the sibling rig's startup script — inherited here verbatim — pulls straight away.

The first boot therefore died in the pull loop's five retries, 100 seconds in:

```
+ sudo docker pull vllm/vllm-tpu:nightly
sudo: docker: command not found
...
ERROR: Failed to pull vLLM Docker image after multiple retries. Exiting.
```

**How this presents is the dangerous part.** The instance still reported `status: RUNNING`, and
kept reporting it, because on this path `RUNNING` means the VM booted — not that the startup
script succeeded, and certainly not that vLLM is serving. `CLAUDE.md` already warns that `RUNNING`
is a weaker claim than the Queued Resource's `ACTIVE`; this is that warning arriving as a concrete
failure. Nothing short of reading `/var/log/vllm-startup.log` or curling `:8000` distinguishes a
dead boot from a healthy one, and `check_tpu_availability` cannot.

Fixed in `../../startup_script_template.sh`: install `docker.io` when `command -v docker` finds
nothing, with the same five-retry shape as the pull loop. The added block contains no literal
braces, so it survives the `str.format()` render — verified.

This instance was repaired in place (`apt-get install docker.io`, then
`google_metadata_script_runner startup`) rather than recreated, to keep the granted flex-start
capacity. The sweep below therefore ran on a VM whose Docker was installed by hand; the template
fix reproduces it from scratch but **is not itself exercised by this run**.

## The baseline's recorded provenance is not evidence of anything

The result JSONs carry `zone` and `provisioning_model` fields. **They are hardcoded string
literals in the harness, not measurements** — `run_cells.py:239-241` writes
`"zone": "us-east5-b", "provisioning_model": "spot"` unconditionally, and `run_knee.py:86-87` and
`run_overflow.py:143` write `europe-west4-a` / `flex-start` the same way. Nothing reads the node.

This was caught here only because the field is *load-bearing for this rig*: the same harness,
run on a Compute Engine instance in europe-west4-a, dutifully stamped its output `us-east5-b` /
`spot`. Corrected in `results/cells_gce_v6e1.json`, with the reason recorded in the file.

**A first reading of this run concluded the sibling baseline was "mixed-provenance" — that its
main sweep ran in us-east5-b on spot while its knee and overflow ran in europe-west4-a on
flex-start. That conclusion was wrong** and is retracted. It rested entirely on these fields, and
comparing two hardcoded literals establishes nothing about two nodes. The sibling `REPORT.md`'s
`europe-west4-a` / flex-start claim has nothing measured contradicting it; the sibling
`README.md` line 83's conflicting `us-east5-b` / spot claim is most likely the same stale literal
read back as fact.

What can honestly be said: **the baseline's zone and provisioning model are unrecorded.** Both
runs are v6e-1, same checkpoint, same engine build, same serving flags, same harness. Whether
they shared a zone is unknown, so a residual difference cannot be cleanly attributed to the
control plane — but there is no positive evidence of a zone difference either.

**Fix the harness before the next run:** take zone and provisioning model as arguments, or read
them off the instance. A provenance field that is written without being measured is worse than an
absent one, because it reads as evidence.

## Harness

`run_cells.py` is copied byte-identical from the sibling run
(`sha256 dc00c839a6ad3607b340fef8168f85a758d93b679c19e18aa1bf1fb43919683d`) and driven the same
way: on the VM host, `vllm bench serve` inside the `vllm-gemma4` container, so the load generator
sits next to the server and network latency does not pollute TTFT.
