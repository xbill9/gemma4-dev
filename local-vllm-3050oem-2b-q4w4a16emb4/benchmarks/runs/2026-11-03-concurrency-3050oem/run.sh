#!/usr/bin/env bash
# Concurrency sweep, vLLM 0.33.0 on the RTX 3050 OEM. The same sweep.py arguments as
# local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-13-concurrency-3050oem: two shapes
# (512 in / 128 out, 128 in / 512 out), levels 1-64, 5 repeats, unique prompts (leading
# nonce, so vLLM's prefix cache cannot hit), decode from the stream. Server from tpu.env
# through serve.py (MAX_NUM_SEQS 64, no drafter). GPU telemetry every second into gpu.csv.
# Run from the rig root.
set -euo pipefail
set -a; . ./tpu.env; set +a
R=benchmarks/runs/2026-11-03-concurrency-3050oem
python3 serve.py --print > "$R/command.txt"
python3 serve.py > "$R/vllm-server.log" 2>&1 &
SRV=$!
trap 'kill $SRV $SMI 2>/dev/null || true' EXIT
until curl -fsS "$ENDPOINT/health" >/dev/null 2>&1; do kill -0 $SRV 2>/dev/null || { echo "server died"; exit 1; }; sleep 2; done
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,memory.used,utilization.gpu \
  --format=csv,noheader -l 1 > "$R/gpu.csv" &
SMI=$!
python3 -c "import importlib.metadata as m; print({p: m.version(p) for p in ('vllm','torch','transformers','flashinfer-python')})" > "$R/versions.txt"
for shape in "512 128 in512-out128" "128 512 in128-out512"; do
  set -- $shape
  curl -fsS "$ENDPOINT/metrics" > "$R/metrics-before-$3.prom"
  python3 sweep.py --base "$ENDPOINT/v1" --model "$MODEL_NAME" --out "$R/$3" \
    --concurrency 1,2,4,8,16,32,64 --conc-input "$1" --conc-output "$2" \
    --repeats 5 --prompt-mode unique --decode-source auto --expect-device gpu
  curl -fsS "$ENDPOINT/metrics" > "$R/metrics-after-$3.prom"
done
nvidia-smi --query-compute-apps=pid,used_memory --format=csv > "$R/vram.txt"
echo DONE
