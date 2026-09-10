---
name: vllm-health
description: End-to-end health check of the deployed Cloud Run vLLM service — auth, service state, endpoint, model listing, and a latency probe. Use when asked whether the vLLM/Gemma endpoint is up or when a tool call to it fails.
---

Diagnose the live vLLM deployment layer by layer, stopping at the first one that fails and reporting
which layer broke. Run the steps in order.

## 1. Auth

```bash
gcloud auth application-default print-access-token >/dev/null 2>&1 && echo "ADC ok" || echo "ADC MISSING"
gcloud config get-value project
```

If ADC is missing, tell the user to run `source ./set_adc.sh` (it is interactive — don't try to drive
the browser login yourself) and stop.

## 2. Service state

```bash
make status
```

Read the `status.conditions` and `status.url` from the output. A service that exists but has
`Ready: False`, or has scaled to zero and is failing its startup probe, is the answer — report it and
stop. Startup on cold start is slow by design (`initialDelaySeconds=180`, GCS FUSE model load), so a
first request after idle may legitimately take minutes.

## 3. Endpoint reachability

```bash
ENDPOINT=$(make -s endpoint)
TOKEN=$(gcloud auth print-identity-token)
curl -s -o /dev/null -w '%{http_code} %{time_total}s\n' -H "Authorization: Bearer $TOKEN" "$ENDPOINT/health"
```

The service is deployed `--no-allow-unauthenticated`, so a `403` means the identity token is missing or
the caller lacks `run.invoker` — not that the model is down.

## 4. Model listing

```bash
curl -s -H "Authorization: Bearer $TOKEN" "$ENDPOINT/v1/models" | python3 -m json.tool
```

Note the returned model `id` — it is the path vLLM actually loaded (e.g. `/mnt/models/gemma-4-E2B-it`).
If it differs from `MODEL_NAME` in the environment or `.agents/mcp_config.json`, that mismatch is a real
finding: the code prefers the live value via `get_active_model_name()`, but any hand-written curl or
`make query` using `MODEL_PATH` will 404.

## 5. Latency probe

```bash
make query PROMPT="Reply with the single word: ok"
```

Report time-to-response. A cold start is expected to be slow; a warm request that is slow points at
`--max-num-seqs`/batching or GPU memory pressure.

## Report

Give a short verdict: which layer failed (or all green), the observed endpoint, the live model id, and
the probe latency. If something failed, give the one concrete next action — don't list every possibility.
