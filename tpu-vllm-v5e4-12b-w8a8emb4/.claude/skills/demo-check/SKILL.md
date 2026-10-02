---
name: demo-check
description: Pre-demo readiness check for the Gemma 4 vLLM-on-TPU stack — finds the ACTIVE queued resource, resolves its endpoint, health-checks vLLM on :8000, runs a smoke query, and reports go/no-go. Use before a live demo, or when asked whether the stack is up.
---

Verify the serving stack is demo-ready. This is **read-only** — never create, restart, or destroy a queued
resource here, even if a step fails. Report the failure and suggest the fix instead.

Work through the steps in order. Stop at the first hard failure and report; don't run later steps against a
stack that isn't there.

## 1. Resolve project and zone

Zone varies per demo and `server.py:26-27` hardcodes it, so don't trust any single source. Read the current
values:

```
grep -n '^ZONE\|^REGION\|^PROJECT_ID' server.py
```

Use that `ZONE` as the primary. If step 2 finds nothing there, retry once in the `Makefile` default zone
(`us-central1-a`) before declaring the stack down — and say explicitly which zone you ended up in.

## 2. Find the ACTIVE queued resource

```
gcloud alpha compute tpus queued-resources list --project=<PROJECT_ID> --zone=<ZONE> --format=json
```

Look for one with `state.state == "ACTIVE"`. Report its name and state.

- `WAITING_FOR_RESOURCES` / `PROVISIONING` — capacity hasn't landed yet. Not a bug; report the wait.
- `SUSPENDED` / `FAILED` — report the state and the `state` sub-message verbatim.
- Nothing at all in either zone — the stack is down. Say so; do not deploy.

## 3. Resolve the endpoint

Get the node id from the queued resource's `tpu.nodeSpec[0].nodeId`, then:

```
gcloud compute tpus tpu-vm describe <node_id> --project=<PROJECT_ID> --zone=<ZONE> --format=json
```

Take `networkEndpoints[0].accessConfig.externalIp`, falling back to `networkEndpoints[0].ipAddress`. The
endpoint is `http://<ip>:8000`. Ignore every IP hardcoded in the repo's markdown — they're stale.

## 4. Health-check vLLM

```
curl -sS -m 10 http://<ip>:8000/v1/models
```

A JSON body listing the served model id is a pass. Connection refused or a timeout usually means the
container is still pulling or loading weights — check with:

```
gcloud compute tpus tpu-vm ssh <node_id> --project=<PROJECT_ID> --zone=<ZONE> \
  --command="sudo docker logs --tail 50 \$(sudo docker ps -q | head -1)"
```

Look for `Application startup complete.` Startup can take ~20 minutes from a cold VM.

## 5. Smoke query

```
curl -sS -m 60 http://<ip>:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<model_id_from_step_4>","prompt":"What is Site Reliability Engineering?","max_tokens":64}'
```

Confirm non-empty generated text. Note the wall-clock latency.

## 6. Report

Give a short go/no-go:

- **Verdict** — READY, or NOT READY with the blocking reason in one line.
- **Zone** actually used, and whether it matched `server.py`.
- **Queued resource** name and state.
- **Endpoint** URL and the served model id.
- **Smoke query** — latency, and the first line of the completion.
- **Warnings** — anything that will bite mid-demo: a zone mismatch between `server.py` and the Makefile, a
  model id that differs from `MODEL_NAME` in `server.py`, or a slow first token.

Keep it tight. Someone is about to present.
