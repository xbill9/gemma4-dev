#!/usr/bin/env bash
# Cloud-init for the jev evaluation instance (EC2 G6, one NVIDIA L4).
# Pulls a vLLM nightly that contains PR #57250 and serves one model at a time
# on :8000. Switch models with: /opt/jev/serve.sh <ar|diffusion|e4b|e2b>
set -euxo pipefail
systemctl enable --now docker
# Hosts under 30 GiB of RAM get a 16 GB swapfile for the 17 GB checkpoint load.
if [ "$(awk '/MemTotal/{print int($2/1048576)}' /proc/meminfo)" -lt 30 ] && ! swapon --show --noheadings | grep -q /swapfile; then
  fallocate -l 16G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
fi
mkdir -p /opt/jev /opt/hf

IMAGE=vllm/vllm-openai:nightly-e9757321527ca1ecd514c07c1418dd2c53da3d19
echo "[stage] pull-start $(date -Is)"
docker pull "$IMAGE"
echo "[stage] pull-done $(date -Is)"

cat >/opt/jev/serve.sh <<'SCRIPT'
#!/usr/bin/env bash
# One flag set for both arms, taken from the DiffusionGemma-on-L4 recipe
# (Triton attention, eager, prefix caching, 2048 context, 16 sequences,
# 32 logprobs). Only the model and the diffusion canvas differ.
set -euo pipefail
IMAGE=vllm/vllm-openai:nightly-e9757321527ca1ecd514c07c1418dd2c53da3d19
case "${1:?ar|diffusion|e4b|e2b}" in
  ar)        MODEL=cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit;          EXTRA=() ;;
  diffusion) MODEL=cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4;   EXTRA=(--diffusion-config '{"canvas_length": 64}') ;;
  e4b)       MODEL=google/gemma-4-E4B-it;                         EXTRA=(--dtype bfloat16) ;;
  e2b)       MODEL=google/gemma-4-E2B-it;                         EXTRA=(--dtype bfloat16) ;;
  *) echo "unknown arm $1" >&2; exit 2 ;;
esac
set +x
HF_TOKEN=$(aws secretsmanager get-secret-value --region us-east-1 --secret-id vllm/hf-token --query SecretString --output text 2>/dev/null || true)
docker rm -f vllm 2>/dev/null || true
docker run -d --name vllm --ipc=host --gpus all -e HF_TOKEN="$HF_TOKEN" \
  -v /opt/hf:/root/.cache/huggingface -p 8000:8000 "$IMAGE" \
  --model "$MODEL" --served-model-name "$MODEL" --host 0.0.0.0 --port 8000 \
  --attention-backend TRITON_ATTN --gpu-memory-utilization 0.92 \
  --max-model-len 2048 --max-num-seqs 16 --max-logprobs 32 \
  --enforce-eager --enable-prefix-caching "${EXTRA[@]}"
echo "serving $MODEL"
SCRIPT
chmod 700 /opt/jev/serve.sh
/opt/jev/serve.sh ar
echo "[stage] serving-started $(date -Is)"
touch /opt/jev/INSTALL_DONE
