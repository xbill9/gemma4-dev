---
name: demo-check
description: Pre-demo readiness check for this rig's Gemma 4 vLLM-on-TPU stack — finds the RUNNING Compute Engine TPU instance, resolves its endpoint, health-checks vLLM on :8000, runs a smoke query, and reports go/no-go. Use before a live demo, or when asked whether the stack is up.
---

Verify the serving stack is demo-ready. This is **read-only** — never create, restart, or destroy an
instance here, even if a step fails. Report the failure and suggest the fix instead.

> **Rewritten 2026-08-19 for this rig.** It previously walked the Cloud TPU API's Queued Resource path,
> inherited from the fork. That is the wrong API here and fails silently rather than loudly: a `ct6e-*`
> instance **does not appear in `gcloud compute tpus tpu-vm list` or `queued-resources list` at all**, so
> the old step 2 reported "stack is down" against a perfectly healthy VM. This rig provisions eight v6e
> chips as one ordinary Compute Engine instance (`ct6e-standard-8t`).

Work through the steps in order. Stop at the first hard failure and report; don't run later steps against a
stack that isn't there.

## 1. Resolve project, zone and instance name

`tpu.env` is the source of truth, and a real environment variable overrides it. Read the current values:

```
grep -E '^(GOOGLE_CLOUD_PROJECT|GOOGLE_CLOUD_ZONE|GOOGLE_CLOUD_REGION|MACHINE_TYPE|MODEL_NAME)=' tpu.env
```

The default instance name is the rig directory name, `gce-vllm-v6e8-31b`, unless `INSTANCE_NAME` is set.
Use `GOOGLE_CLOUD_ZONE` (`europe-west4-a`) as the primary. If step 2 finds nothing there, say so explicitly
rather than sweeping other zones — a demo runs against one instance, and `find_tpu` is the tool for
searching, not this skill.

## 2. Find the RUNNING instance

```
gcloud compute instances list --project=<PROJECT_ID> --zones=<ZONE> \
  --filter="machineType~'ct6e|ct5p'" --format=json
```

Look for `status == "RUNNING"`. Report its name, `machineType`, and the `rig=` label.

- **`RUNNING` is weaker than the old `ACTIVE`.** A Queued Resource reached ACTIVE once its node was up; an
  instance is RUNNING the moment the VM boots — long before the startup script has installed Docker, pulled
  the vLLM image, or loaded weights. **Never report READY on step 2.** Steps 4 and 5 are the readiness test.
- `PROVISIONING` / `STAGING` — still coming up. Not a bug; report the wait.
- `TERMINATED` / `STOPPING` — report it verbatim. Flex-start and spot instances self-delete at
  `MAX_RUN_DURATION` (4h), so a vanished instance is usually expected, not broken.
- Nothing at all — the stack is down. Say so; do not deploy.
- An instance whose `rig=` label is a **sibling rig's** — report it and leave it alone. Several rigs
  provision `ct6e-*` into this project.

## 3. Resolve the endpoint

The instance **is** the node — there is no Queued Resource indirection and no derived `<id>-node`.

```
gcloud compute instances describe <name> --project=<PROJECT_ID> --zone=<ZONE> --format=json
```

Take `networkInterfaces[0].accessConfigs[0].natIP`, falling back to `networkInterfaces[0].networkIP`. The
endpoint is `http://<ip>:8000`. Ignore every IP hardcoded in the repo's markdown — they're stale.

## 4. Health-check vLLM

```
curl -sS -m 10 http://<ip>:8000/v1/models
```

A JSON body listing the served model id is a pass. Connection refused or a timeout usually means the
container is still pulling or loading weights. Check the boot log first — it is the one place a dead boot
is distinguishable from a slow one:

```
gcloud compute ssh <name> --project=<PROJECT_ID> --zone=<ZONE> \
  --command="sudo tail -50 /var/log/vllm-startup.log; sudo docker logs --tail 50 vllm-gemma4 2>&1 | tail -50"
```

Note `gcloud compute ssh`, **not** `gcloud compute tpus tpu-vm ssh` — the latter cannot see this instance.

Look for `Application startup complete.` **Budget up to 90 minutes from a cold VM on this rig, not the ~20
the E2B rigs need** — the 31B checkpoint is 62 GB to pull from Hugging Face before vLLM starts compiling.
`startup_script_template.sh` waits that long before declaring failure, and a boot that is merely slow is
indistinguishable from a dead one until it does. Do not call a stack down inside the first hour; read the
log. Two failure signatures worth naming in the report:

- `sudo: docker: command not found` — the image ships no Docker and the startup script installs it; if this
  appears the install line was lost. The instance will sit at `RUNNING` forever.
- Secret Manager retries for 30 minutes — the instance is missing `--scopes=cloud-platform`, or its service
  account lacks `roles/secretmanager.secretAccessor` on the `hf-token` secret.

## 5. Smoke query

Use **`/v1/chat/completions`**. Raw `/v1/completions` returns an empty completion on `-it` models, so an
empty result there proves nothing:

```
curl -sS -m 60 http://<ip>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<model_id_from_step_4>","messages":[{"role":"user","content":"What is Site Reliability Engineering?"}],"max_tokens":64}'
```

Confirm non-empty generated text. Note the wall-clock latency.

## 6. Report

Give a short go/no-go:

- **Verdict** — READY, or NOT READY with the blocking reason in one line.
- **Zone** actually used, and whether it matched `tpu.env`.
- **Instance** name, `machineType`, `status`, provisioning model, and `rig=` label.
- **Endpoint** URL and the served model id.
- **Smoke query** — latency, and the first line of the completion.
- **Warnings** — anything that will bite mid-demo: a model id that differs from `MODEL_NAME`, a slow first
  token, an instance close to its `MAX_RUN_DURATION` deletion, or a `--tensor-parallel-size` in the running
  container that differs from `tpu.env`.

Keep it tight. Someone is about to present.
