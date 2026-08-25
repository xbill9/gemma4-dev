# Deployment Guide: Gemma 4 31B on v6e-8 via Compute Engine

What this rig deploys, and the exact commands. **Every value below is also in `tpu.env`, which is
the source of truth** — if the two disagree, `tpu.env` wins and this file is stale.

## 📦 Model Artifacts

**Gemma 4 31B**, the reference instruction-tuned release, pulled from Hugging Face at boot.

*   **Model ID:** `google/gemma-4-31B-it`
*   **Format:** Hugging Face Transformers, standard bf16
*   **Precision:** bfloat16 — no weight-quantization route boots on this stack (`@../QUANTIZATION.md`)
*   **Size:** 31.0B parameters, **62 GB on disk**, 57.7 GiB resident, ~7.2 GiB per chip at TP=8

> The 62 GB download is why `startup_script_template.sh` waits **90 minutes** for
> `Application startup complete.` rather than the 20 it waited on E2B. A boot that is merely slow
> and a boot that has failed look identical until that loop gives up.

## 🚀 Inference Stack

**Compute Engine, not the Cloud TPU API.** The accelerator is a property of the machine type;
there is no queued resource, no `--runtime-version`, and no derived `<id>-node`. A `ct6e-*`
instance created this way is **invisible to `gcloud compute tpus tpu-vm list`** — use
`gcloud compute instances list`.

*   **Hardware**
    *   **TPU:** v6e (Trillium), 8 chips, one host
    *   **Machine type:** `ct6e-standard-8t` (360 vCPU / 1440 GB)
    *   **Topology:** `2x4`
    *   **Image:** family `ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e`, project `ubuntu-os-accelerator-images`
*   **Software**
    *   **Container:** `vllm/vllm-tpu:nightly`
    *   **Max model length:** `32768` (per-request ceiling; see `SERVING-PARAMS.md`)
    *   **Tensor parallel size:** `8` — **not optional on this checkpoint.** 57.7 GiB of weights
        do not fit one 31.24 GiB chip, so TP=1 and TP=2 cannot boot. TP=4 is the only alternative.
    *   **Max num batched tokens:** `4096`

## 🛠 Usage

To point the MCP agent at a running service:

```bash
export VLLM_BASE_URL="http://<INSTANCE_IP>:8000"
export MODEL_NAME="google/gemma-4-31B-it"
export GOOGLE_CLOUD_PROJECT="aisprint-491218"
```

Then `make run`. In practice you do not need any of this — `mcp-run.sh` exports `tpu.env` for you,
and the endpoint is discovered rather than configured (first RUNNING instance → external IP → `:8000`).

## 📜 Deployment Commands

### 1. Create the instance

**This is a Compute Engine create.** The `gcloud compute tpus tpu-vm create` form that this file
carried while the rig was forked from the TPU-API lineage is the *wrong control plane* and would
meter against a different quota pool.

```bash
gcloud compute instances create gce-vllm-v6e8-31b \
  --machine-type=ct6e-standard-8t \
  --image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e \
  --image-project=ubuntu-os-accelerator-images \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=200GB \
  --scopes=cloud-platform \
  --provisioning-model=FLEX_START \
  --instance-termination-action=DELETE \
  --max-run-duration=4h \
  --zone=europe-west4-a \
  --project=aisprint-491218 \
  --metadata-from-file=startup-script=startup_script_template.sh
```

Prefer the `create_tpu_vm_instance` / `find_tpu_vm` tools, which render the startup script with this
rig's parameters, apply the `rig=gce-vllm-v6e8-31b` label, and record zone failures in
`tpu_zones_status.md`. `get_vllm_deployment_config` prints the copy-pasteable form of the same call.

> **Eight chips bill as eight chips.** At the us-east5 v6e flex-start list rate of $1.35/chip-hr that
> is $10.80/hr, and a 4h `MAX_RUN_DURATION` is a ~$43 instance. Run `estimate_deployment_cost` first.

### 2. The vLLM container the startup script runs

```bash
sudo docker run -t --rm --name vllm-gemma4 --privileged --net=host \
    -v /dev/shm:/dev/shm --shm-size 10gb \
    -e HF_HOME=/dev/shm \
    -e HF_TOKEN=$HF_TOKEN \
    vllm/vllm-tpu:nightly \
    vllm serve google/gemma-4-31B-it \
    --max-model-len 32768 \
    --tensor-parallel-size 8 \
    --disable_chunked_mm_input \
    --max_num_batched_tokens 4096 \
    --limit-mm-per-prompt '{"image":4,"audio":1}' \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4
```

`--privileged --net=host` and the `/dev/shm` bind mount are what let the eight per-chip workers see
the chips and each other; all collectives stay on the on-board ICI. The bind mount makes `--shm-size`
a no-op — the host's `/dev/shm` is what is used, which is also where the 62 GB checkpoint lands
(`HF_HOME=/dev/shm`, half of 1440 GB available).

This mirrors `startup_script_template.sh`. **If you change the flags here, change them there too** —
the template is what an actual deploy runs. Both paths read `_vllm_serve_flags()` in `server.py`, so
the honest fix for a flag change is `tpu.env`.

### 3. Verification

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "google/gemma-4-31B-it",
        "messages": [{"role": "user", "content": "Hello Gemma 4!"}]
    }'
```

Use `/v1/chat/completions`, not `/v1/completions` — raw completions return empty output on `-it`
checkpoints, which reads as a broken deploy and is not one.
