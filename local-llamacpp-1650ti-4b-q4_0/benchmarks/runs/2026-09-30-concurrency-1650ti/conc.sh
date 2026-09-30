#!/bin/bash
# E4B concurrency sweep on the 1650 Ti: two shapes, one server, every level cooled first.
#   long:  512 in / 128 out (the 2026-09-03 E2B full-sweep shape)
#   short: 128 in / 512 out (the 2026-09-08 E2B short-prompt shape)
# -c 16384 --parallel 16 = 1024 tokens/slot. 32 slots does NOT fit on E4B: llama.cpp gives each slot
# its own 1024-cell sliding-window KV (20 layers x 2 KiB x 1024 = 40 MiB/slot), so 32 slots asks for
# 1280 MiB sliding + 512 MiB full on top of 2493 MiB of weights (probe/alloc-c32768-p32.log).
set -u
ROOT=/home/xbill/gemma4-dev
RIG=$ROOT/local-llamacpp-1650ti-4b-q4_0
OUT=$RIG/benchmarks/runs/2026-09-30-concurrency-1650ti
LOG=$OUT/driver.log
LEVELS=1,2,4,8,16
cd $RIG; set -a; . ./tpu.env; set +a
pkgt() { sensors 2>/dev/null | awk '/Package id 0/{gsub(/[+°C]/,"",$4); print int($4)}'; }
gput() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits; }
gpuc() { nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader; }
thr() { cat /sys/devices/system/cpu/cpu0/thermal_throttle/package_throttle_count 2>/dev/null || echo na; }
state() { echo "$(date +%T) pkg=$(pkgt)C gpu=$(gput)C throttle=$(thr) $*" | tee -a "$LOG"; }
cooldown() {
  local t0=$SECONDS; sleep 60
  while :; do
    if [ "$(pkgt)" -le 50 ] && [ "$(gput)" -le 45 ]; then break; fi
    if [ $((SECONDS - t0)) -ge 900 ]; then state "cooldown TIMEOUT"; break; fi
    sleep 10
  done
  state "cooldown done after $((SECONDS - t0))s"
}
state "BEGIN"
setsid nohup $LLAMA_SERVER_BIN -lv 4 -m $MODEL_PATH --host $HOST --port $PORT -ngl $N_GPU_LAYERS -c 16384 \
  -ctk $KV_CACHE_TYPE -ctv $KV_CACHE_TYPE -fa $FLASH_ATTENTION -t $THREADS -tb $THREADS_BATCH \
  --parallel 16 --reasoning $REASONING --metrics > $OUT/server.log 2>&1 < /dev/null &
for i in $(seq 90); do curl -fsS $ENDPOINT/health >/dev/null 2>&1 && break; sleep 1; done
state "healthy vram=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader)"
for shape in long:512:128 short:128:512; do
  IFS=: read name cin cout <<< "$shape"
  for c in ${LEVELS//,/ }; do
    cooldown
    d=$OUT/$name/c$c; mkdir -p $d
    state "START $name c=$c gpu_clock/power=$(gpuc)"
    python3 sweep.py --base $ENDPOINT/v1 --out $d --rig local-llamacpp-1650ti-4b-q4_0 --expect-device gpu \
      --concurrency $c --conc-input $cin --conc-output $cout --repeats 3 --prompt-mode unique \
      --decode-source auto > $d/sweep.log 2>&1
    state "END $name c=$c rc=$? gpu_clock/power=$(gpuc)"
  done
done
curl -fsS $ENDPOINT/metrics > $OUT/metrics-final.prom 2>/dev/null
kill $(pgrep -x llama-server); while pgrep -x llama-server >/dev/null; do sleep 1; done
state "DONE"
# Restore the E2B demo server with its original argv.
(cd $ROOT/local-llamacpp-1650ti-2b-q4_0 && setsid nohup $(cat $RIG/benchmarks/runs/2026-09-30-exact-gguf-e4b-1650ti/serve/e2b-server-cmdline-before.txt) > run/llama-server.log 2>&1 < /dev/null &)
for i in $(seq 60); do curl -fsS $ENDPOINT/health >/dev/null 2>&1 && break; sleep 2; done
state "E2B server restored: $(curl -fsS $ENDPOINT/health)"
