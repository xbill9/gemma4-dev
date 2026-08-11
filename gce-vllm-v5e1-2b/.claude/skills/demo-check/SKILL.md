---
name: demo-check
description: Readiness check for the Gemma 4 vLLM-on-TPU stack on this rig's Compute Engine path — finds the RUNNING instance, resolves its endpoint, health-checks vLLM on :8000, runs a smoke query, and reports go/no-go. Use when asked whether the stack is up.
---

Verify the serving stack is ready. This is **read-only** — never create, restart, or destroy an instance
here, even if a step fails. Report the failure and suggest the fix instead.

**This rig provisions through Compute Engine, not the Cloud TPU API.** Every command below is
`gcloud compute instances` / `gcloud compute ssh`. The twin rig `tpu-vllm-v5e1-2b` has its own version of
this skill built on `queued-resources` and `tpu-vm`; those commands return **not-found** against a
`ct5lp-*` instance that is plainly RUNNING, so don't reach for them when a step here comes up empty.

Bear in mind this rig may never have provisioned anything — whether v5e is reachable on the Compute Engine
path at all is an open question (see `CLAUDE.md`). "Nothing found" is the expected result, not a fault.

Work through the steps in order. Stop at the first hard failure and report; don't run later steps against a
stack that isn't there.

## 1. Resolve project and zone

`tpu.env` is the single source of truth, and a real environment variable beats it. Read the current values:

```
grep -n '^GOOGLE_CLOUD_PROJECT\|^GOOGLE_CLOUD_ZONE\|^GOOGLE_CLOUD_REGION\|^INSTANCE_NAME' tpu.env
```

Defaults are `us-west4-a` / `us-west4`. Say explicitly which zone you ended up in.

## 2. Find the instance

```
gcloud compute instances list --project=<PROJECT_ID> --zones=<ZONE> \
  --filter="machineType~'ct5lp|ct5p|ct6e'" --format=json
```

Look for one with `status == "RUNNING"`, preferring the one named `gce-vllm-v5e1-2b` or labelled
`rig=gce-vllm-v5e1-2b`.

- `PROVISIONING` / `STAGING` — the VM is still coming up. Not a bug; report the wait.
- `TERMINATED` / `SUSPENDED` — report the state verbatim. On flex-start and spot this is the *expected*
  end state: both delete the VM at `--max-run-duration`.
- Nothing at all — the stack is down. Say so; do not deploy.

**`RUNNING` is a weaker claim than the twin's `ACTIVE`.** A Queued Resource reached ACTIVE once its node was
up; an instance reports RUNNING the moment the VM boots, long before the startup script has installed
Docker, pulled the vLLM image or loaded the model. Never report READY on step 2 alone.

Also worth a look, because four TPU-API v5e rigs share this zone and compete for the same physical chips:

```
gcloud alpha compute tpus queued-resources list --project=<PROJECT_ID> --zone=<ZONE> --format=json
```

Anything there belongs to a sibling rig. Report it as context for a capacity failure; never touch it.

## 3. Resolve the endpoint

Take the IP from the instance JSON at `networkInterfaces[0].accessConfigs[0].natIP`, falling back to
`networkInterfaces[0].networkIP`. (This is **not** where the TPU API puts it — that path uses
`networkEndpoints[0]`, which is absent here.) The endpoint is `http://<ip>:8000`. Ignore every IP hardcoded
in the repo's markdown — they're stale.

## 4. Health-check vLLM

```
curl -sS -m 10 http://<ip>:8000/v1/models
```

A JSON body listing the served model id is a pass. Connection refused or a timeout usually means the
container is still pulling or loading weights — check with:

```
gcloud compute ssh <instance_name> --project=<PROJECT_ID> --zone=<ZONE> \
  --command="sudo docker logs --tail 50 \$(sudo docker ps -q | head -1)"
```

Look for `Application startup complete.` Startup can take ~20 minutes from a cold VM, most of it XLA
compilation, and the compile cache is container-local so a restart repays all of it.

If `docker ps` itself fails with `docker: command not found`, that is the known image gotcha, not a fluke:
the Compute Engine image ships no Docker and the startup script installs it. Read the startup log instead —
this is the failure mode that leaves an instance reporting RUNNING forever:

```
gcloud compute ssh <instance_name> --project=<PROJECT_ID> --zone=<ZONE> \
  --command="sudo tail -50 /var/log/vllm-startup.log"
```

## 5. Smoke query

Use the **chat** endpoint. Raw `/v1/completions` returns an empty completion on `-it` models, so an empty
result there proves nothing:

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
- **Instance** name, `status`, machine type, and provisioning model (from `scheduling.provisioningModel`).
- **Time left**, if it carries a `--max-run-duration` — flex-start and spot both self-delete.
- **Endpoint** URL and the served model id.
- **Smoke query** — latency, and the first line of the completion.
- **Warnings** — anything that will bite: a zone mismatch between `tpu.env` and what you found, a model id
  that differs from `MODEL_NAME`, a sibling rig holding the zone's chips, or a slow first token.

Keep it tight.
