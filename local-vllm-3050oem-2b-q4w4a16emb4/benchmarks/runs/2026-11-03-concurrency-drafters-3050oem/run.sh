#!/usr/bin/env bash
# Concurrency sweep with drafters, vLLM 0.33.0 on the RTX 3050 OEM: the same sweep.py arguments as
# ../2026-11-03-concurrency-3050oem (no drafter, run the same morning), with the W4A16 drafter
# (/home/xbill/models/gemma-4-E2B-it-qat-q4_0-w4a16-ct-assistant, SHA-256 in ../2026-11-03-drafters-3050oem)
# at 1-4 drafted tokens, one server per arm. Acceptance from the spec-decode counters in /metrics
# before and after each shape. The card is cooled to <= 50 C before each arm. Run from the rig root.
set -euo pipefail
set -a; . ./tpu.env; set +a
R=benchmarks/runs/2026-11-03-concurrency-drafters-3050oem
export SPECULATIVE_MODEL=/home/xbill/models/gemma-4-E2B-it-qat-q4_0-w4a16-ct-assistant
for N in ${ARMS:-1 2 3 4}; do
  A=w4a16-n$N; mkdir -p "$R/$A"
  until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" -le 50 ]; do sleep 5; done
  NUM_SPECULATIVE_TOKENS=$N python3 serve.py --print > "$R/$A/command.txt"
  NUM_SPECULATIVE_TOKENS=$N setsid python3 serve.py > "$R/$A/vllm-server.log" 2>&1 &
  SRV=$!
  until curl -fsS "$ENDPOINT/health" >/dev/null 2>&1; do kill -0 $SRV 2>/dev/null || { echo "server died $A"; exit 1; }; sleep 2; done
  nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,memory.used,utilization.gpu \
    --format=csv,noheader -l 1 > "$R/$A/gpu.csv" &
  SMI=$!
  for shape in "512 128 in512-out128" "128 512 in128-out512"; do
    set -- $shape
    curl -fsS "$ENDPOINT/metrics" > "$R/$A/metrics-before-$3.prom"
    python3 sweep.py --base "$ENDPOINT/v1" --model "$MODEL_NAME" --out "$R/$A/$3" \
      --concurrency 1,2,4,8,16,32,64 --conc-input "$1" --conc-output "$2" \
      --repeats 5 --prompt-mode unique --decode-source auto --expect-device gpu
    curl -fsS "$ENDPOINT/metrics" > "$R/$A/metrics-after-$3.prom"
  done
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv > "$R/$A/vram.txt"
  kill $SMI; kill -TERM -- -$SRV; wait $SRV 2>/dev/null || true
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 100 ]; do sleep 2; done
  echo "$(date +%T) END $A"
done
echo DONE
