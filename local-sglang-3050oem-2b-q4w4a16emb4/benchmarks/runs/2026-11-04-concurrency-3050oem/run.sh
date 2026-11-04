#!/usr/bin/env bash
# Concurrency sweep, SGLang 0.5.23 on the RTX 3050 OEM: the sweep.py arguments of
# local-vllm-3050oem-2b-q4w4a16emb4/benchmarks/runs/2026-11-03-concurrency-3050oem (512 in / 128 out and
# 128 / 512, levels 1-64, 5 repeats, unique prompts, decode from the stream, thinking on). Server from
# tpu.env through serve.py, no drafter. Run from the rig root.
set -euo pipefail
set -a; . ./tpu.env; set +a
R=benchmarks/runs/2026-11-04-concurrency-3050oem
SPECULATIVE_MODEL= python3 serve.py --print > "$R/command.txt"
SPECULATIVE_MODEL= python3 serve.py > "$R/server.log" 2>&1 &
SRV=$!
trap 'kill $SRV $SMI 2>/dev/null || true' EXIT
until curl -fsS "$ENDPOINT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":2}" >/dev/null 2>&1; do
  kill -0 $SRV 2>/dev/null || { echo "server died"; exit 1; }; sleep 2; done
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,memory.used,utilization.gpu \
  --format=csv,noheader -l 1 > "$R/gpu.csv" &
SMI=$!
python3 -c "import sglang, torch, flashinfer; print('sglang', sglang.__version__, 'torch', torch.__version__, 'flashinfer', flashinfer.__version__)" > "$R/versions.txt" 2>&1 || true
for shape in "512 128 in512-out128" "128 512 in128-out512"; do
  set -- $shape
  python3 sweep.py --base "$ENDPOINT/v1" --model "$MODEL_NAME" --out "$R/$3" \
    --concurrency 1,2,4,8,16,32,64 --conc-input "$1" --conc-output "$2" \
    --repeats 5 --prompt-mode unique --decode-source auto --expect-device gpu
done
nvidia-smi --query-compute-apps=pid,used_memory --format=csv > "$R/vram.txt"
echo DONE
