#!/usr/bin/env bash
# Concurrency sweep, RTX 3050 OEM. One server sized for 64 slots x 1024 tokens;
# both shapes fit a slot (512+128 and 128+512 plus chat template). GPU telemetry
# is sampled every second into gpu.csv for the whole run. Run from the rig root.
set -euo pipefail
set -a; . ./tpu.env; set +a
R=benchmarks/runs/2026-10-13-concurrency-3050oem
"$LLAMA_SERVER_BIN" -m "$MODEL_PATH" --host "$HOST" --port "$PORT" -ngl 99 \
  -c 65536 --parallel 64 -ctk f16 -ctv f16 -fa 1 -t 6 -tb 12 \
  --reasoning off --metrics > "$R/llama-server.log" 2>&1 &
SRV=$!
trap 'kill $SRV $SMI 2>/dev/null || true' EXIT
until curl -fsS "$ENDPOINT/health" >/dev/null 2>&1; do sleep 1; done
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,memory.used,utilization.gpu \
  --format=csv,noheader -l 1 > "$R/gpu.csv" &
SMI=$!
for shape in "512 128 in512-out128" "128 512 in128-out512"; do
  set -- $shape
  curl -fsS "$ENDPOINT/metrics" > "$R/metrics-before-$3.prom"
  python3 sweep.py --base "$ENDPOINT/v1" --model "$MODEL_NAME" --out "$R/$3" \
    --concurrency 1,2,4,8,16,32,64 --conc-input "$1" --conc-output "$2" \
    --repeats 5 --prompt-mode unique --decode-source auto --expect-device gpu
  curl -fsS "$ENDPOINT/metrics" > "$R/metrics-after-$3.prom"
done
nvidia-smi --query-compute-apps=pid,used_memory --format=csv > "$R/vram.txt"
