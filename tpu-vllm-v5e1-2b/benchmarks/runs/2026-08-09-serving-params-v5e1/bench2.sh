#!/usr/bin/env bash
# bench2.sh <container-tag> <reps> [outdir]
# Full 4x3 matrix with repeats, one JSON line per (cell,rep) to stdout.
TAG="$1"; REPS="${2:-3}"; OUT="${3:-/tmp/bench-$TAG}"
mkdir -p "$OUT"
cell(){ # ctx conc nprompts rep
  local ctx=$1 conc=$2 np=$3 rep=$4
  local f="$OUT/c${ctx}-u${conc}-r${rep}.txt"
  sudo docker exec vllm-$TAG vllm bench serve \
    --model google/gemma-4-E2B-it --base-url http://localhost:8000 \
    --dataset-name random --random-input-len $ctx --random-output-len 128 \
    --max-concurrency $conc --num-prompts $np --ignore-eos \
    --percentile-metrics ttft,tpot,itl --metric-percentiles 50,99 > "$f" 2>&1
  python3 - "$f" "$ctx" "$conc" "$rep" <<'PY'
import re,sys,json
t=open(sys.argv[1]).read()
def g(p):
    m=re.search(p+r"[^0-9\-]*([0-9]+\.?[0-9]*)",t)
    return float(m.group(1)) if m else None
print(json.dumps({"ctx":int(sys.argv[2]),"conc":int(sys.argv[3]),"rep":int(sys.argv[4]),
 "out_tps":g("Output token throughput"),"tot_tps":g("Total Token throughput"),
 "ttft_med":g("Median TTFT"),"ttft_p99":g("P99 TTFT"),
 "tpot_med":g("Median TPOT"),"itl_med":g("Median ITL"),"itl_p99":g("P99 ITL"),
 "ok":g("Successful requests")}))
PY
}
for rep in $(seq 1 $REPS); do
  for ctx in 128 1024 8192; do
    for conc in 1 4 16 64; do
      case $conc in 1) np=8;; 4) np=16;; 16) np=48;; 64) np=128;; esac
      [ $ctx -ge 8192 ] && [ $conc -eq 1 ] && np=6
      cell $ctx $conc $np $rep
    done
  done
done
