# Deployment Guide: vLLM on TPUs (Gemma 4)

This document summarizes the deployment state and configuration for the vLLM inference server running on Google Cloud TPUs.

## 📦 Model Artifacts
The model used is **Gemma 4 12B**, served directly from Hugging Face.

*   **Model ID:** `xbill9/gemma-4-12B-it-qat-w8a8-int8-emb4`
*   **Format:** compressed-tensors, text only
*   **Precision:** int8 W8A8 linears, int4 `embed_tokens` and untied `lm_head`

## 🚀 Inference Stack (vLLM on TPU)
The inference server is deployed on **Cloud TPU v5e (v5litepod)** using the `vllm-tpu` specialized container.

*   **Hardware:** 
    *   **TPU Version:** v5e (v5litepod)
    *   **Topology:** `2x2` (4 chips on one host, v5litepod-4)
*   **Software:**
    *   **Image:** `vllm-tpu-w8a8emb4:patched`, built on the VM at boot from the pinned `vllm/vllm-tpu@sha256:19a1a052…` and `patches/` (see README.md)
    *   **Max Model Length:** `8192`
    *   **Tensor Parallel Size:** `4`
    *   Every serving setting but the TP size matches the v5e-1 sibling, `../tpu-vllm-v5e1-12b-w8a8emb4`.

## 🛠 Usage
To connect the MCP Agent to the TPU service, export the following environment variables:

```bash
export VLLM_BASE_URL="http://<TPU_VM_IP>:8000"
export MODEL_NAME="xbill9/gemma-4-12B-it-qat-w8a8-int8-emb4"
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
    vllm-tpu-w8a8emb4:patched \
    vllm serve xbill9/gemma-4-12B-it-qat-w8a8-int8-emb4 \
    --max-model-len 8192 \
    --tensor-parallel-size 4 \
    --disable_chunked_mm_input \
    --max_num_batched_tokens 512 \
    --gpu-memory-utilization 0.92 \
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
        "model": "xbill9/gemma-4-12B-it-qat-w8a8-int8-emb4",
        "messages": [{"role": "user", "content": "Hello Gemma 4!"}]
    }'
```
