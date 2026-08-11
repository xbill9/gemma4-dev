#!/bin/bash
exec > /var/log/vllm-startup.log 2>&1
set -ex # Enable command tracing and exit on error

echo "Starting Queued vLLM Bootloader..."
echo "-----------------------------------"
echo "Project ID: {project_id}"
echo "Zone: {zone}"
echo "Model Name: {model_name}"
echo "HF_SECRET_ID: {hf_secret_id}"
echo "-----------------------------------"

# Ensure internet connectivity
echo "Checking internet connectivity..."
set +e # Allow ping to fail without exiting immediately
for i in $(seq 1 30); do
  echo "Attempt $i/30: Pinging 8.8.8.8..."
  ping -c 1 8.8.8.8
  if [ $? -eq 0 ]; then
    echo "Internet connected."
    break
  fi
  echo "Ping failed, retrying in 5 seconds..."
  sleep 5
  if [ $i -eq 30 ]; then
    echo "ERROR: Internet connectivity failed after multiple retries. Exiting."
    exit 1
  fi
done
set -e # Re-enable exit on error

# Install Docker if the image does not already carry it.
#
# THIS IS A CONTROL-PLANE DIFFERENCE, NOT A PRECAUTION. The Cloud TPU API's runtime
# versions (v2-alpha-tpuv5-lite and friends) ship with Docker preinstalled, so the sibling
# rig's copy of this script could pull straight away. The Compute Engine image family
# ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e does NOT: `docker` is absent from PATH on a
# fresh boot. Verified 2026-08-10 on the first instance the v6e twin ever created, where
# its absence burned the pull loop's five retries and failed the whole script in 100s.
# That rig boots the same image family as this one, so it applies here unchanged.
#
# The failure is loud in /var/log/vllm-startup.log ("sudo: docker: command not found")
# and completely silent everywhere else: the instance still reports RUNNING, so nothing
# short of reading this log or curling :8000 distinguishes it from a healthy boot.
echo "Checking for Docker..."
if ! command -v docker > /dev/null 2>&1; then
  echo "Docker not present on this image — installing docker.io..."
  set +e
  for i in $(seq 1 5); do
    echo "Attempt $i/5: apt-get install docker.io"
    sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker.io
    if command -v docker > /dev/null 2>&1; then
      echo "Docker installed."
      break
    fi
    echo "Docker install failed, retrying in 20 seconds..."
    sleep 20
    if [ $i -eq 5 ]; then
      echo "ERROR: Could not install Docker after multiple retries. Exiting."
      exit 1
    fi
  done
  set -e
  sudo systemctl enable --now docker
else
  echo "Docker already present."
fi

# Docker pull vLLM image
echo "Pulling vLLM Docker image: vllm/vllm-tpu:nightly"
set +e # Allow docker pull to fail without exiting immediately
for i in $(seq 1 5); do
  echo "Attempt $i/5: sudo docker pull vllm/vllm-tpu:nightly"
  sudo docker pull vllm/vllm-tpu:nightly
  if [ $? -eq 0 ]; then
    echo "Docker image pulled successfully."
    break
  fi
  echo "Docker pull failed, retrying in 20 seconds..."
  sleep 20
  if [ $i -eq 5 ]; then
    echo "ERROR: Failed to pull vLLM Docker image after multiple retries. Exiting."
    exit 1
  fi
done
set -e # Re-enable exit on error

# Set vLLM environment variables
echo "Setting vLLM environment variables..."
VLLM_MODEL="{model_name}"
VLLM_MAX_MODEL_LEN="{max_model_len}"
VLLM_TP_SIZE="{tensor_parallel_size}"
VLLM_MAX_BATCHED_TOKENS="{max_num_batched_tokens}"
VLLM_LIMIT_MM_PER_PROMPT='{limit_mm_per_prompt}'
HF_HOME="/dev/shm"

echo "VLLM_MODEL set to: $VLLM_MODEL"
echo "VLLM_MAX_MODEL_LEN set to: $VLLM_MAX_MODEL_LEN"
echo "VLLM_TP_SIZE set to: $VLLM_TP_SIZE"
echo "VLLM_MAX_BATCHED_TOKENS set to: $VLLM_MAX_BATCHED_TOKENS"
echo "VLLM_LIMIT_MM_PER_PROMPT set to: $VLLM_LIMIT_MM_PER_PROMPT"
echo "HF_HOME set to: $HF_HOME"

# Fetch the Hugging Face token from Secret Manager at boot.
# The token is NEVER written into instance metadata or this script — it is read at
# runtime with the VM's own service-account credentials from the metadata server.
# Tracing stays off from here on: every line below touches the token.
set +x
echo "Fetching '{hf_secret_id}' from Secret Manager (retrying for up to 30 minutes)..."
METADATA_TOKEN_URL="http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
SECRET_URL="https://secretmanager.googleapis.com/v1/projects/{project_id}/secrets/{hf_secret_id}/versions/latest:access"

HF_TOKEN=""
set +e
for i in $(seq 1 90); do
  ACCESS_TOKEN=$(curl -s -H "Metadata-Flavor: Google" "$METADATA_TOKEN_URL" |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])' 2>/dev/null)
  if [ -n "$ACCESS_TOKEN" ]; then
    HF_TOKEN=$(curl -s -H "Authorization: Bearer $ACCESS_TOKEN" "$SECRET_URL" |
      python3 -c 'import base64,json,sys; print(base64.b64decode(json.load(sys.stdin)["payload"]["data"]).decode())' 2>/dev/null)
  fi
  if [ -n "$HF_TOKEN" ]; then
    echo "Secret retrieved successfully on attempt $i (value masked)."
    break
  fi
  echo "Attempt $i/90: could not read the secret yet. Retrying in 20 seconds..."
  echo "  (if this never succeeds, grant the VM service account roles/secretmanager.secretAccessor on '{hf_secret_id}')"
  sleep 20
done
set -e

if [ -z "$HF_TOKEN" ]; then
  echo "ERROR: Could not read '{hf_secret_id}' from Secret Manager after 30 minutes."
  echo "Grant the VM's service account roles/secretmanager.secretAccessor on the secret, then reset the VM."
  exit 1
fi

echo "Attempting to start vLLM container..."
# Stop and remove any existing container with the same name to ensure a clean start
sudo docker stop vllm-gemma4 > /dev/null 2>&1 || true
sudo docker rm vllm-gemma4 > /dev/null 2>&1 || true

# Log the full docker run command before executing it. HF_TOKEN is deliberately shown
# as a literal placeholder here — do not interpolate it into a logged string.
echo 'Executing command: sudo docker run --name vllm-gemma4 --privileged --net=host -d \
  -v /dev/shm:/dev/shm --shm-size 10gb \
  -e HF_HOME="$HF_HOME" \
  -e HF_TOKEN=<masked> \
  vllm/vllm-tpu:nightly vllm serve "$VLLM_MODEL" \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  --tensor-parallel-size "$VLLM_TP_SIZE" \
  --disable_chunked_mm_input \
  --max_num_batched_tokens "$VLLM_MAX_BATCHED_TOKENS" \
  --limit-mm-per-prompt "$VLLM_LIMIT_MM_PER_PROMPT" \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4'

sudo docker run --name vllm-gemma4 --privileged --net=host -d \
  -v /dev/shm:/dev/shm --shm-size 10gb \
  -e HF_HOME="$HF_HOME" \
  -e HF_TOKEN="$HF_TOKEN" \
  vllm/vllm-tpu:nightly vllm serve "$VLLM_MODEL" \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  --tensor-parallel-size "$VLLM_TP_SIZE" \
  --disable_chunked_mm_input \
  --max_num_batched_tokens "$VLLM_MAX_BATCHED_TOKENS" \
  --limit-mm-per-prompt "$VLLM_LIMIT_MM_PER_PROMPT" \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4

if [ $? -ne 0 ]; then
  echo "ERROR: Docker run command failed. Check parameters and image."
  sudo docker logs vllm-gemma4 || echo "Could not fetch logs for failed container."
  exit 1
fi

# The token is no longer needed in this shell; drop it and resume tracing.
unset HF_TOKEN ACCESS_TOKEN
set -x

echo "Docker container started. Waiting for 'Application startup complete.' in logs (up to 20 minutes)..."
HEALTHY=0
for i in $(seq 1 120); do
  if sudo docker logs vllm-gemma4 2>&1 | grep -q "Application startup complete."; then
    echo "vLLM 'Application startup complete.' message found in logs."
    HEALTHY=1
    break
  fi
  echo "vLLM not yet fully started (attempt $i/120). Retrying in 10 seconds..."
  sleep 10
done

if [ "$HEALTHY" -eq 0 ]; then
  echo "ERROR: vLLM did not report 'Application startup complete.' within the timeout."
  echo "Attempting to retrieve Docker logs for 'vllm-gemma4':"
  sudo docker logs vllm-gemma4 || echo "Could not retrieve Docker logs."
  exit 1
fi

echo "vLLM application startup complete. The server should now be ready."
