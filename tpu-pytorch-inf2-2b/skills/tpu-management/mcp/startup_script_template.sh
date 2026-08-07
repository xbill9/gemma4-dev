#!/bin/bash
# Mirror all output to the serial console: SSH to TPU VMs is often blocked by
# firewall policy, and the serial log is then the only way to watch boot progress
# (gcloud compute instances get-serial-port-output).
exec > >(tee /var/log/vllm-startup.log > /dev/console) 2>&1
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

# The TPU runtime images ship Docker, but the GCE ubuntu-accel images
# (ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e) do NOT — install it if missing.
if ! command -v docker >/dev/null 2>&1; then
  echo "Docker not found; installing docker.io..."
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y
  apt-get install -y docker.io
  systemctl enable --now docker
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

# Fetch the Hugging Face token from Secret Manager at boot. The token is never
# baked into instance metadata (readable by anyone with compute.instances.get).
# Retries for up to ~30 minutes so an IAM grant applied after creation still
# lands; a fetch that 403s forever means the VM's service account is missing
# roles/secretmanager.secretAccessor or the cloud-platform scope.
echo "Fetching HF token from Secret Manager secret '{hf_secret_id}'..."
# Tracing (set -x) stays off from here on: it would print the expanded token in
# the HF_TOKEN assignment and the docker run command. The echo statements below
# narrate progress instead.
set +x
HF_TOKEN=""
set +e
for i in $(seq 1 180); do
  ACCESS_TOKEN=$(curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])' 2>/dev/null)
  if [ -n "$ACCESS_TOKEN" ]; then
    HF_TOKEN=$(curl -sf -H "Authorization: Bearer $ACCESS_TOKEN" \
      "https://secretmanager.googleapis.com/v1/projects/{project_id}/secrets/{hf_secret_id}/versions/latest:access" \
      | python3 -c 'import json,sys,base64; print(base64.b64decode(json.load(sys.stdin)["payload"]["data"]).decode())' 2>/dev/null)
  fi
  if [ -n "$HF_TOKEN" ]; then
    echo "HF token fetched from Secret Manager."
    break
  fi
  echo "Attempt $i/180: could not fetch secret (missing IAM grant or scope?). Retrying in 10 seconds..."
  sleep 10
done
set -e
if [ -z "$HF_TOKEN" ]; then
  echo "ERROR: could not fetch Secret Manager secret '{hf_secret_id}' after 30 minutes."
  echo "Grant roles/secretmanager.secretAccessor to the VM's service account and reset the VM."
  exit 1
fi

# Set vLLM environment variables
echo "Setting vLLM environment variables..."
VLLM_MODEL="{model_name}"
VLLM_MAX_MODEL_LEN="65536"
VLLM_TP_SIZE="{tp_size}"
VLLM_MAX_BATCHED_TOKENS="4096"
{limit_mm_per_prompt_env}
HF_HOME="/dev/shm"

echo "VLLM_MODEL set to: $VLLM_MODEL"
echo "VLLM_MAX_MODEL_LEN set to: $VLLM_MAX_MODEL_LEN"
echo "VLLM_TP_SIZE set to: $VLLM_TP_SIZE"
echo "VLLM_MAX_BATCHED_TOKENS set to: $VLLM_MAX_BATCHED_TOKENS"
if [ -n "{limit_mm_per_prompt_env}" ]; then
  echo "VLLM_LIMIT_MM_PER_PROMPT set." # Don't echo actual value for sensitive info
fi
echo "HF_HOME set to: $HF_HOME"
echo "HF_TOKEN set (value masked for security)."

echo "Attempting to start vLLM container..."
# Stop and remove any existing container with the same name to ensure a clean start
sudo docker stop vllm-gemma4 > /dev/null 2>&1 || true
sudo docker rm vllm-gemma4 > /dev/null 2>&1 || true

# Log the full docker run command before executing it (token masked — this goes
# to the serial console, which is readable by anyone with instance access)
echo "Executing command: sudo docker run --name vllm-gemma4 --privileged --net=host -d \\
  -v /dev/shm:/dev/shm --shm-size 10gb \\
  -e HF_HOME=\"$HF_HOME\" \\
  -e HF_TOKEN=\"***masked***\" \\
  vllm/vllm-tpu:nightly vllm serve \"$VLLM_MODEL\" \\
  --max-model-len \"$VLLM_MAX_MODEL_LEN\" \\
  --tensor-parallel-size \"$VLLM_TP_SIZE\" \\
  --disable_chunked_mm_input \\
  --max_num_batched_tokens 4096 \\
  --enable-auto-tool-choice \\
  --tool-call-parser gemma4 \\
  --reasoning-parser gemma4"

sudo docker run --name vllm-gemma4 --privileged --net=host -d \
  -v /dev/shm:/dev/shm --shm-size 10gb \
  -e HF_HOME="$HF_HOME" \
  -e HF_TOKEN="$HF_TOKEN" \
  vllm/vllm-tpu:nightly vllm serve "$VLLM_MODEL" \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  --tensor-parallel-size "$VLLM_TP_SIZE" \
  --disable_chunked_mm_input \
  --max_num_batched_tokens 4096 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4

if [ $? -ne 0 ]; then
  echo "ERROR: Docker run command failed. Check parameters and image."
  sudo docker logs vllm-gemma4 || echo "Could not fetch logs for failed container."
  exit 1
fi

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
