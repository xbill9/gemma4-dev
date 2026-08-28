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
# versions (v2-alpha-tpuv6e and friends) ship with Docker preinstalled, so the sibling
# rig's copy of this script could pull straight away. The Compute Engine image family
# ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e does NOT: `docker` is absent from PATH on a
# fresh boot. Verified 2026-08-10 on the first instance this rig ever created, where its
# absence burned the pull loop's five retries and failed the whole script in 100s.
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
VLLM_IMAGE="{vllm_image}"
echo "Pulling vLLM Docker image: $VLLM_IMAGE"
set +e # Allow docker pull to fail without exiting immediately
for i in $(seq 1 5); do
  echo "Attempt $i/5: sudo docker pull $VLLM_IMAGE"
  sudo docker pull "$VLLM_IMAGE"
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
# Tensor parallel degree, rendered from TENSOR_PARALLEL_SIZE. At 8 this launches one vLLM
# worker per chip on this single host; all collectives stay on the on-board ICI, so nothing
# below needs a multi-host rank or coordinator address. --privileged --net=host and the
# /dev/shm bind mount are what let those workers see the chips and each other. Note the bind
# mount makes --shm-size below a no-op — the host's /dev/shm (half of 1440 GB) is what is
# actually used, which is why an 8-way run does not need it raised.
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

# --- Warm the Hugging Face cache from GCS, if it has been staged ---------------------------
# The 62 GB checkpoint pull from Hugging Face IS the boot-readiness budget below: it is the
# reason this script waits 90 minutes rather than the 20 it waited while the rig served E2B,
# and at $10.80/hr for eight chips it is roughly $16 of the cost of every single boot.
#
# An in-region GCS read is one or two orders of magnitude faster than the public internet, so
# staging the cache once and restoring it here turns that into a few minutes. `stage_model_to_gcs.sh`
# produces the object; MODEL_GCS_URI in tpu.env points at it.
#
# The cache is stored as a single uncompressed tar and streamed straight into place. Three
# reasons, each learned the boring way:
#   - tar PRESERVES THE SYMLINKS. The HF cache keeps one real copy per blob under blobs/ and
#     symlinks snapshots/ at it. `gcloud storage cp -r` follows symlinks, which would upload
#     124 GB instead of 62 and restore two real copies.
#   - ONE object, not ~100 shards, so the restore is a single sequential read rather than a
#     per-object round trip each time.
#   - NOT gzipped. bf16 weights do not compress, so gzip would trade GCS bandwidth for a CPU
#     bottleneck and come out slower.
#
# THIS IS AN OPTIMIZATION AND MUST NEVER BECOME A NEW FAILURE MODE. Every branch below falls
# through to the normal online pull, which is why the tar is verified to exist before the
# stream starts and why HF_HUB_OFFLINE is set ONLY after a restore that actually succeeded.
MODEL_GCS_URI="{model_gcs_uri}"
HF_OFFLINE_FLAG=""
if [ -n "$MODEL_GCS_URI" ]; then
  echo "Staged cache configured at $MODEL_GCS_URI — checking that it exists..."
  if sudo gcloud storage ls "$MODEL_GCS_URI" > /dev/null 2>&1; then
    echo "Restoring Hugging Face cache from $MODEL_GCS_URI into $HF_HOME (streaming, no temp disk)..."
    RESTORE_START=$(date +%s)
    set +e
    sudo gcloud storage cat "$MODEL_GCS_URI" | sudo tar -C "$HF_HOME" -xf -
    RESTORE_RC=$?
    set -e
    if [ $RESTORE_RC -eq 0 ] && [ -d "$HF_HOME/hub" ]; then
      echo "Cache restored in $(($(date +%s) - RESTORE_START))s. Serving offline; no Hugging Face download needed."
      # Only safe because the restore succeeded and the tar is a COMPLETE hub/ tree. This also
      # makes the boot deterministic: no revision check against a moving upstream.
      HF_OFFLINE_FLAG="-e HF_HUB_OFFLINE=1"
    else
      echo "WARNING: restore failed (rc=$RESTORE_RC). Falling back to the normal Hugging Face pull."
      HF_OFFLINE_FLAG=""
    fi
  else
    echo "WARNING: $MODEL_GCS_URI is not readable (missing object, or the VM service account"
    echo "  lacks roles/storage.objectViewer). Falling back to the normal Hugging Face pull."
  fi
else
  echo "No MODEL_GCS_URI set — pulling the checkpoint from Hugging Face (expect 60-90 minutes)."
  echo "  Stage it with stage_model_to_gcs.sh to cut this to minutes on every future boot."
fi

# --- Optional xprof sidecar -----------------------------------------------------------------
# Arming the profiler at BOOT rather than afterwards saves a full engine restart, which on
# this checkpoint is ~7 minutes of recompilation (measured: init engine 427.9 s, compilation
# 399.6 s). vLLM's own VLLM_TORCH_PROFILER_DIR does not exist on these images and
# /start_profile 404s, so the trigger has to be injected in-process — see profiling/.
# Empty XPROF_SRC leaves the container byte-identical to an unprofiled boot.
XPROF_SRC="{xprof_gcs_uri}"
XPROF_MOUNT=""
if [ -n "$XPROF_SRC" ]; then
  echo "Fetching xprof sidecar from $XPROF_SRC..."
  sudo mkdir -p /opt/xprof /dev/shm/xprof
  if sudo gcloud storage cp "$XPROF_SRC/*" /opt/xprof/ 2>/dev/null; then
    XPROF_MOUNT="-v /opt/xprof:/opt/xprof -e PYTHONPATH=/opt/xprof -e VLLM_XPROF_DIR=/dev/shm/xprof -e VLLM_XPROF_PORT=9012"
    echo "xprof sidecar armed; trace control will listen on :9012."
  else
    echo "WARNING: could not fetch the sidecar. Booting WITHOUT profiling."
  fi
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
  '"$HF_OFFLINE_FLAG $XPROF_MOUNT"' \
  "$VLLM_IMAGE" vllm serve "$VLLM_MODEL" \
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
  $HF_OFFLINE_FLAG $XPROF_MOUNT \
  "$VLLM_IMAGE" vllm serve "$VLLM_MODEL" \
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

# 90 minutes, not the 20 this script carried while the rig served E2B. THE CHECKPOINT IS WHY:
# 31B is 62 GB of safetensors to pull from Hugging Face before vLLM starts compiling, against
# E2B's ~10 GB, and the TPU compile that followed took 685 s for E2B on a single v5e chip.
# Neither number is measured here and both grow with this model, so the old ceiling would have
# expired mid-boot on a healthy instance — and this loop's timeout is what the script reports
# as failure. Raise it again rather than trimming it if a real boot runs long; the instance is
# already paid for by then, and MAX_RUN_DURATION is the thing that actually stops the billing.
echo "Docker container started. Waiting for 'Application startup complete.' in logs (up to 90 minutes)..."
HEALTHY=0
for i in $(seq 1 540); do
  if sudo docker logs vllm-gemma4 2>&1 | grep -q "Application startup complete."; then
    echo "vLLM 'Application startup complete.' message found in logs."
    HEALTHY=1
    break
  fi
  echo "vLLM not yet fully started (attempt $i/540). Retrying in 10 seconds..."
  sleep 10
done

if [ "$HEALTHY" -eq 0 ]; then
  echo "ERROR: vLLM did not report 'Application startup complete.' within the timeout."
  echo "Attempting to retrieve Docker logs for 'vllm-gemma4':"
  sudo docker logs vllm-gemma4 || echo "Could not retrieve Docker logs."
  exit 1
fi

echo "vLLM application startup complete. The server should now be ready."
