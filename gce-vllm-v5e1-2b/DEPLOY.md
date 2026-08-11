# Deployment Guide: vLLM on TPUs (Gemma 4)

This document summarizes the deployment state and configuration for the vLLM inference server running on Google Cloud TPUs.

## 📦 Model Artifacts
The model used is **Gemma 4 2B**, served directly from Hugging Face.

*   **Model ID:** `google/gemma-4-E2B-it`
*   **Format:** Hugging Face Transformers (standard BF16)
*   **Precision:** bfloat16

## 🚀 Inference Stack (vLLM on TPU)
The inference server is deployed on **one Cloud TPU v5e chip** using the `vllm-tpu` specialized container,
provisioned through **Compute Engine**.

*   **Hardware:**
    *   **TPU Version:** v5e, 16 GB HBM
    *   **Machine type:** `ct5lp-hightpu-1t` (1 chip, 24 vCPU / 48 GB host)
    *   **Image:** `ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e`, 200 GB boot disk. Ships no Docker; the startup
        script installs it.
*   **Software:**
    *   **Image:** `vllm/vllm-tpu:nightly`
    *   **Max Model Length:** `16384`
    *   **Tensor Parallel Size:** `1`

## 🛠 Usage
To connect the MCP Agent to the TPU service, export the following environment variables:

```bash
export VLLM_BASE_URL="http://<TPU_VM_IP>:8000"
export MODEL_NAME="google/gemma-4-E2B-it"
export GOOGLE_CLOUD_PROJECT="aisprint-491218"
```

Then run the agent:
```bash
make run
```

## 📜 Deployment Commands

### 1. Create the v5e Compute Engine instance

**This rig uses Compute Engine, not the Cloud TPU API.** There is no `tpu-vm create` and no Queued Resource
here. Note that whether v5e is reachable this way at all is unsettled — see `CLAUDE.md` before running this.

```bash
gcloud compute instances create gce-vllm-v5e1-2b \
    --project $PROJECT_ID --zone $ZONE \
    --machine-type ct5lp-hightpu-1t \
    --image-family ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e \
    --image-project ubuntu-os-accelerator-images \
    --maintenance-policy TERMINATE \
    --boot-disk-size 200GB \
    --scopes cloud-platform \
    --provisioning-model FLEX_START \
    --request-valid-for-duration 2h \
    --max-run-duration 4h \
    --instance-termination-action DELETE
```

`--maintenance-policy TERMINATE` is required (a TPU instance cannot live-migrate) and `--scopes
cloud-platform` is what lets the startup script read the HF token from Secret Manager. Swap `FLEX_START` for
`SPOT`, `STANDARD` or `RESERVATION_BOUND` as needed; `--request-valid-for-duration` applies to `FLEX_START`
only. Or just use `make deploy-tpu-flex`, which builds the same command from `tpu.env`.

### 2. Launch vLLM Container (on the instance)
```bash
sudo docker run -t --rm --name vllm-gemma4 --privileged --net=host \
    -v /dev/shm:/dev/shm --shm-size 10gb \
    -e HF_HOME=/dev/shm \
    -e HF_TOKEN=$HF_TOKEN \
    vllm/vllm-tpu:nightly \
    vllm serve google/gemma-4-E2B-it \
    --max-model-len 16384 \
    --tensor-parallel-size 1 \
    --disable_chunked_mm_input \
    --max_num_batched_tokens 4096 \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4
```

This mirrors what `startup_script_template.sh` runs on the VM. If you change the
flags here, change them there too — the template is what an actual deploy uses.

### 3. Verification
```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "google/gemma-4-E2B-it",
        "messages": [{"role": "user", "content": "Hello Gemma 4!"}]
    }'
```
