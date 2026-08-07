# Deployment Guide: vLLM on TPUs (Gemma 4)

This document summarizes the deployment state and configuration for the vLLM inference server running on Google Cloud TPUs.

## 📦 Model Artifacts
The model used is **Gemma 4 2B**, served directly from Hugging Face.

*   **Model ID:** `google/gemma-4-E2B-it-qat-q4_0-unquantized`
*   **Format:** Hugging Face Transformers (standard BF16)
*   **Precision:** bfloat16

## 🚀 Inference Stack (vLLM on TPU)
The inference server is deployed on **Cloud TPU v5e (v5litepod)** using the `vllm-tpu` specialized container.

*   **Hardware:** 
    *   **TPU Version:** v5e (v5litepod)
    *   **Topology:** `1x1` (1 chip, v5litepod-1)
*   **Software:**
    *   **Image:** `vllm/vllm-tpu:nightly`
    *   **Max Model Length:** `16384`
    *   **Tensor Parallel Size:** `1`

## 🛠 Usage
To connect the MCP Agent to the TPU service, export the following environment variables:

```bash
export VLLM_BASE_URL="http://<TPU_VM_IP>:8000"
export MODEL_NAME="google/gemma-4-E2B-it-qat-q4_0-unquantized"
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
    --type v5litepod --topology 1x1 \
    --project $PROJECT_ID --zone $ZONE --version v2-alpha-tpuv5-lite
```

### 2. Launch vLLM Container (on TPU VM)
```bash
sudo docker run -t --rm --name vllm-gemma4 --privileged --net=host \
    -v /dev/shm:/dev/shm --shm-size 10gb \
    -e HF_HOME=/dev/shm \
    -e HF_TOKEN=$HF_TOKEN \
    vllm/vllm-tpu:nightly \
    vllm serve google/gemma-4-E2B-it-qat-q4_0-unquantized \
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
        "model": "google/gemma-4-E2B-it-qat-q4_0-unquantized",
        "messages": [{"role": "user", "content": "Hello Gemma 4!"}]
    }'
```
