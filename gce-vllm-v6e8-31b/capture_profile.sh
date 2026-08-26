#!/bin/bash
#
# Capture an xprof/TensorBoard trace from the live vLLM TPU deployment and pull it local.
#
# THE OBVIOUS WAY DOES NOT WORK. An earlier version of this script set
# VLLM_TORCH_PROFILER_DIR and POSTed /start_profile, which is the documented vLLM recipe.
# Measured on hardware 2026-08-25 against vllm/vllm-tpu:nightly: the variable is not known to
# the build ("Unknown vLLM environment variable detected"), /start_profile returns 404, and
# the OpenAPI document contains no profile route at all. See profiler_sidecar.py.
#
# So this script injects the trigger into the engine process instead:
#   1. copies profiling/{profiler_sidecar,sitecustomize}.py to the VM
#      (they live in a subdirectory because a sitecustomize.py at the rig root would be
#       auto-imported by every Python process started from there, hook or no hook)
#   2. recreates the container with that directory bind-mounted and on PYTHONPATH, so Python
#      auto-imports sitecustomize in every process it starts (the sidecar keeps itself to the
#      one process that owns the TPU)
#   3. waits for the engine to serve, POSTs /start, drives load, POSTs /stop
#   4. tars the trace off the VM
#
# Step 2 means a FULL ENGINE RESTART, which on this checkpoint is ~7 minutes of recompilation
# (measured: init engine took 427.9 s, of which 399.6 s was compilation). The checkpoint
# itself is already in /dev/shm, so nothing is re-downloaded.
#
# Usage: ./capture_profile.sh [out_dir]
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$HERE/tpu.env" ]; then set -a; . "$HERE/tpu.env"; set +a; fi

PROJECT="${GOOGLE_CLOUD_PROJECT:-aisprint-491218}"
ZONE="${GOOGLE_CLOUD_ZONE:-europe-west4-a}"
INSTANCE="${INSTANCE_NAME:-gce-vllm-v6e8-31b}"
MODEL="${MODEL_NAME:-google/gemma-4-31B-it}"
TP="${TENSOR_PARALLEL_SIZE:-8}"
MML="${MAX_MODEL_LEN:-32768}"
MNBT="${MAX_NUM_BATCHED_TOKENS:-4096}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-tpu:nightly}"
XPROF_DIR="${VLLM_XPROF_DIR:-/dev/shm/xprof}"
XPROF_PORT="${VLLM_XPROF_PORT:-9012}"
DURATION="${PROFILE_SECONDS:-20}"
OUT="${1:-$HERE/benchmarks/runs/$(date +%Y-%m-%d)-xprof-tp${TP}}"

ssh_() { gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT" --command="$1"; }

echo "▶ instance $INSTANCE ($ZONE)   TP=$TP   trace -> $XPROF_DIR"

echo "== 1. shipping the sidecar =="
gcloud compute scp --zone="$ZONE" --project="$PROJECT" \
  "$HERE/profiling/profiler_sidecar.py" "$HERE/profiling/sitecustomize.py" "$INSTANCE:/tmp/"
ssh_ "sudo mkdir -p /opt/xprof $XPROF_DIR && sudo cp /tmp/profiler_sidecar.py /tmp/sitecustomize.py /opt/xprof/"

echo "== 2. recreating the container with the sidecar on PYTHONPATH =="
# PYTHONPATH puts /opt/xprof on sys.path, which is what makes Python auto-import
# sitecustomize. --net=host is already set, so the control port is reachable on localhost.
ssh_ "sudo docker rm -f vllm-gemma4 >/dev/null 2>&1 || true
sudo docker run --name vllm-gemma4 --privileged --net=host -d \
  -v /dev/shm:/dev/shm -v /opt/xprof:/opt/xprof --shm-size 10gb \
  -e HF_HOME=/dev/shm -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH=/opt/xprof \
  -e VLLM_XPROF_DIR=$XPROF_DIR -e VLLM_XPROF_PORT=$XPROF_PORT \
  $IMAGE vllm serve $MODEL \
  --max-model-len $MML --tensor-parallel-size $TP --disable_chunked_mm_input \
  --max_num_batched_tokens $MNBT --limit-mm-per-prompt '{\"image\":4,\"audio\":1}' \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4"

echo "== 3. waiting for the engine (expect ~7 min of recompilation) =="
for i in $(seq 1 25); do
  if ssh_ "sudo docker logs vllm-gemma4 2>&1 | grep -q 'Application startup complete'" 2>/dev/null; then
    echo "   serving after ${i} min"; break
  fi
  [ "$i" = "25" ] && { echo "❌ never came up; sudo docker logs vllm-gemma4"; exit 1; }
  sleep 60
done

# The sidecar prints one line when it installs. If it is absent the container came up without
# a trace trigger, and driving load would produce an empty directory rather than an error.
echo "== 4. confirming the sidecar installed =="
if ! ssh_ "sudo docker logs vllm-gemma4 2>&1 | grep -q 'xprof-sidecar. trace control'"; then
  echo "❌ sidecar did not install. Its own reason (if any):"
  ssh_ "sudo docker logs vllm-gemma4 2>&1 | grep -i 'xprof-sidecar' || echo '  (no sidecar output at all — PYTHONPATH not applied?)'"
  exit 1
fi
ssh_ "curl -s http://localhost:$XPROF_PORT/ ; echo"

echo "== 5. tracing $DURATION s under load =="
ssh_ "curl -s -X POST http://localhost:$XPROF_PORT/start; echo"
ssh_ "timeout $DURATION sudo docker exec vllm-gemma4 vllm bench serve \
  --host localhost --port 8000 --model $MODEL --dataset-name random \
  --num-prompts 128 --random-input-len 32 --random-output-len 128 \
  --max-concurrency 64 --ignore-eos >/dev/null 2>&1 || true; echo load-done"
ssh_ "curl -s -X POST http://localhost:$XPROF_PORT/stop; echo"

echo "== 6. pulling the trace =="
mkdir -p "$OUT"
ssh_ "sudo tar -C $(dirname "$XPROF_DIR") -czf /tmp/xprof.tgz $(basename "$XPROF_DIR")"
gcloud compute scp --zone="$ZONE" --project="$PROJECT" "$INSTANCE:/tmp/xprof.tgz" "$OUT/xprof.tgz"
tar -C "$OUT" -xzf "$OUT/xprof.tgz"

echo
if find "$OUT" -name "*.xplane.pb" | grep -q .; then
  echo "✅ trace in $OUT"
  find "$OUT" -name "*.xplane.pb" -exec ls -lh {} \;
  echo "   analyse:    python3 analyze_trace.py $OUT"
  echo "   xprof:      xprof --logdir $OUT"
  echo "   tensorboard: tensorboard --logdir $OUT"
else
  echo "⚠️  no .xplane.pb found — the trace started but wrote nothing."
  echo "   Usually means the load never reached the engine. Check step 5 output."
  exit 1
fi
