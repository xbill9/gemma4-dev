# Deployment Guide: vLLM on TPUs (Gemma 4)

This document summarizes the deployment state and configuration for the vLLM inference server running on Google Cloud TPUs.

## 📦 Model Artifacts
The model used is **Gemma 4 2B**, served directly from Hugging Face.

*   **Model ID:** `google/gemma-4-E2B-it-qat-w4a16-ct`
*   **Format:** Hugging Face Transformers (standard BF16)
*   **Precision:** bfloat16

## 🚀 Inference Stack (vLLM on TPU)
The inference server is deployed on **Cloud TPU v5e (v5litepod)** using the `vllm-tpu` specialized container.

*   **Hardware:** 
    *   **TPU Version:** v5e (v5litepod)
    *   **Topology:** `2x2` (4 chips on one host, v5litepod-4)
*   **Software:**
    *   **Image:** `vllm/vllm-tpu@sha256:19a1a052…` with `patches/` applied (applied in `patches/ORDER`), built on the VM at boot as `vllm-tpu-w4a16:patched`
    *   **Max Model Length:** `16384`
    *   **Tensor Parallel Size:** `4`

## 🛠 Usage
To connect the MCP Agent to the TPU service, export the following environment variables:

```bash
export VLLM_BASE_URL="http://<TPU_VM_IP>:8000"
export MODEL_NAME="google/gemma-4-E2B-it-qat-w4a16-ct"
export GOOGLE_CLOUD_PROJECT="aisprint-491218"
```

Then run the agent:
```bash
make run
```

## 📜 Deployment Commands

### 1. Create TPU v5e Instance
```bash
gcloud alpha compute tpus tpu-vm create vllm-gemma4-tpu \
    --type v5litepod --topology 2x2 \
    --project $PROJECT_ID --zone $ZONE --version v2-alpha-tpuv5-lite
```

### 2. Launch vLLM Container (on TPU VM)
```bash
sudo docker run -t --rm --name vllm-gemma4 --privileged --net=host \
    -v /dev/shm:/dev/shm --shm-size 10gb \
    -e HF_HOME=/dev/shm \
    -e HF_TOKEN=$HF_TOKEN \
    vllm-tpu-w4a16:patched \
    vllm serve google/gemma-4-E2B-it-qat-w4a16-ct \
    --max-model-len 16384 \
    --tensor-parallel-size 4 \
    --disable_chunked_mm_input \
    --max_num_batched_tokens 4096 \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4
```

`vllm-tpu-w4a16:patched` exists once the boot script has built it from the pinned digest and `patches/`. This mirrors what `startup_script_template.sh` runs on the VM. If you change the
flags here, change them there too — the template is what an actual deploy uses.

### 3. Verification
```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "google/gemma-4-E2B-it-qat-w4a16-ct",
        "messages": [{"role": "user", "content": "Hello Gemma 4!"}]
    }'
```
