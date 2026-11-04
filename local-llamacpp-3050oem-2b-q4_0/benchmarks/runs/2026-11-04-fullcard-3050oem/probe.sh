#!/usr/bin/env bash
# Can llama.cpp turn the rest of the 8 GB card into speed? The server of ../2026-11-03-concurrency-3050oem
# (7182ef71e, 64 slots x 1,024, f16 cache, flash attention, no drafter; the per-layer table in CPU RAM because
# that run passed no -ot) against variants that use more of the card: the per-layer table on the card (-ot,
# as the one-request runs serve), larger prompt batches (-ub/-b 2048), a shared cache (-kvu). Same sweep.py
# arguments at 8 and 64 requests, 3 repeats. Run from the rig root.
set -euo pipefail
set -a; . ./tpu.env; set +a
R=benchmarks/runs/2026-11-04-fullcard-3050oem
BASE=(-m "$MODEL_PATH" --host 127.0.0.1 --port "$PORT" -ngl 99 -c 65536 --parallel 64 -ctk f16 -ctv f16
      -fa 1 -t 6 -tb 12 --reasoning off --metrics)
OT=(-ot "$OVERRIDE_TENSORS")
declare -A ARM=(
  [a-base]=""
  [b-ot]="OT"
  [c-ot-ub2048]="OT -ub 2048 -b 2048"
  [d-ot-ub2048-kvu]="OT -ub 2048 -b 2048 -kvu"
)
for A in ${ARMS:-a-base b-ot c-ot-ub2048 d-ot-ub2048-kvu}; do
  mkdir -p "$R/$A"
  X=()
  for w in ${ARM[$A]}; do if [ "$w" = OT ]; then X+=("${OT[@]}"); else X+=("$w"); fi; done
  until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" -le 50 ]; do sleep 5; done
  echo "$LLAMA_SERVER_BIN ${BASE[*]} ${X[*]}" > "$R/$A/command.txt"
  "$LLAMA_SERVER_BIN" "${BASE[@]}" "${X[@]}" > "$R/$A/llama-server.log" 2>&1 &
  SRV=$!
  until curl -fsS "$ENDPOINT/health" >/dev/null 2>&1; do kill -0 $SRV 2>/dev/null || { echo "server died $A"; exit 1; }; sleep 1; done
  nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,memory.used,utilization.gpu \
    --format=csv,noheader -l 1 > "$R/$A/gpu.csv" &
  SMI=$!
  for shape in "512 128 in512-out128" "128 512 in128-out512"; do
    set -- $shape
    python3 sweep.py --base "$ENDPOINT/v1" --model "$MODEL_NAME" --out "$R/$A/$3" \
      --concurrency 8,64 --conc-input "$1" --conc-output "$2" \
      --repeats 3 --prompt-mode unique --decode-source auto --expect-device gpu
  done
  kill $SMI $SRV; wait $SRV 2>/dev/null || true
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 100 ]; do sleep 2; done
  echo "$(date +%T) END $A"
done
echo DONE
