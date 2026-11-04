#!/usr/bin/env bash
# Concurrency sweep with the drafter, llama.cpp 7182ef71e on the RTX 3050 OEM: the same server and sweep.py
# arguments as ../2026-11-03-concurrency-3050oem (no drafter, run the same morning) plus the exact Q4_0 drafter
# (SPEC_DRAFT_MODEL in tpu.env) at 1 and 2 drafted tokens, one server per arm. The card is cooled to <= 50 C
# before each arm. Run from the rig root.
set -euo pipefail
set -a; . ./tpu.env; set +a
R=benchmarks/runs/2026-11-03-concurrency-drafters-3050oem
"$LLAMA_SERVER_BIN" --version > "$R/version.txt" 2>&1 || true
for N in ${ARMS:-1 2}; do
  A=q4x-n$N; mkdir -p "$R/$A"
  until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" -le 50 ]; do sleep 5; done
  "$LLAMA_SERVER_BIN" -m "$MODEL_PATH" --host "$HOST" --port "$PORT" -ngl 99 \
    -c 65536 --parallel 64 -ctk f16 -ctv f16 -fa 1 -t 6 -tb 12 \
    --reasoning off --metrics \
    -md "$SPEC_DRAFT_MODEL" --spec-type draft-mtp -ngld all --spec-draft-n-max $N > "$R/$A/llama-server.log" 2>&1 &
  SRV=$!
  until curl -fsS "$ENDPOINT/health" >/dev/null 2>&1; do kill -0 $SRV 2>/dev/null || { echo "server died $A"; exit 1; }; sleep 1; done
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
  kill $SMI $SRV; wait $SRV 2>/dev/null || true
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 100 ]; do sleep 2; done
  echo "$(date +%T) END $A"
done
echo DONE
