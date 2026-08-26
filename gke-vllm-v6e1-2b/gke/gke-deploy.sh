#!/usr/bin/env bash
# Apply the vLLM Deployment + Service to the cluster gke-up.sh created.
#
# Three steps, none of which has an analogue on the GCE twin: fetch cluster
# credentials, materialise the HF token as a Kubernetes Secret, apply the manifest.
# On the twin all three are one startup script running on a VM.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
[ -f tpu.env ] && set -a && . ./tpu.env && set +a

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:?set GOOGLE_CLOUD_PROJECT in tpu.env}"
LOCATION="${GKE_LOCATION:?set GKE_LOCATION in tpu.env}"
CLUSTER="${GKE_CLUSTER_NAME:?set GKE_CLUSTER_NAME in tpu.env}"

export MODEL_NAME="${MODEL_NAME:?}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
export TPU_TOPOLOGY="${TPU_TOPOLOGY:-1x1}"
export GKE_TPU_ACCELERATOR="${GKE_TPU_ACCELERATOR:-tpu-v6e-slice}"
export GKE_SERVICE_TYPE="${GKE_SERVICE_TYPE:-LoadBalancer}"
export VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-tpu:nightly}"
export VLLM_SHM_SIZE="${VLLM_SHM_SIZE:-16Gi}"
export HF_K8S_SECRET="${HF_K8S_SECRET:-hf-token}"
HF_SECRET_ID="${HF_SECRET_ID:-hf-token}"

command -v kubectl >/dev/null || { echo "❌ kubectl not found — run: make gke-preflight" >&2; exit 2; }

echo "🔑 fetching credentials for $CLUSTER..."
gcloud container clusters get-credentials "$CLUSTER" --project="$PROJECT_ID" --location="$LOCATION"

# The token lives in Secret Manager, same source the GCE twin's startup script reads.
# It is copied into a Kubernetes Secret rather than baked into the manifest, so the
# rendered YAML stays safe to commit and to print.
echo "🔐 syncing $HF_K8S_SECRET from Secret Manager secret '$HF_SECRET_ID'..."
HF_TOKEN="$(gcloud secrets versions access latest --secret="$HF_SECRET_ID" --project="$PROJECT_ID")"
[ -n "$HF_TOKEN" ] || { echo "❌ empty HF token from Secret Manager" >&2; exit 1; }
kubectl create secret generic "$HF_K8S_SECRET" \
  --from-literal=token="$HF_TOKEN" \
  --dry-run=client -o yaml | kubectl apply -f -
unset HF_TOKEN

echo "📦 applying vllm-gemma4 ($MODEL_NAME, max-model-len=$MAX_MODEL_LEN, tp=$TENSOR_PARALLEL_SIZE)..."
RENDERED="$(mktemp -t vllm-gemma4-XXXX.yaml)"
envsubst < gke/vllm-gemma4.yaml.tmpl > "$RENDERED"
# An unset variable renders as an empty flag value (--max_num_batched_tokens=) that vLLM
# rejects minutes later, inside the pod. Catch it here instead.
if grep -qE '^\s*-\s+--[a-z0-9_-]+=\s*$' "$RENDERED"; then
  echo "❌ rendered manifest has an empty flag value — an env var is unset:" >&2
  grep -nE '^\s*-\s+--[a-z0-9_-]+=\s*$' "$RENDERED" >&2
  exit 1
fi
kubectl apply -f "$RENDERED"
echo "   rendered manifest: $RENDERED"

echo
echo "⏳ waiting for the pod to be scheduled onto the TPU node..."
kubectl wait --for=condition=PodScheduled pod -l app=vllm-gemma4 --timeout=300s || true
kubectl get pods -l app=vllm-gemma4 -o wide

echo
echo "The model pull plus TPU compile takes several minutes. Watch it with:"
echo "  make gke-logs"
echo "  make gke-status"
echo "  make gke-endpoint     # external IP, once the LoadBalancer is assigned"
