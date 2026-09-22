#!/bin/bash
# ABBA paired sweep: CPU, GPU, GPU, CPU. Flags identical except -ngl.
set -u
ROOT=/home/xbill/gemma4-dev
CPU=$ROOT/local-llamacpp-cpu-2b-q4_0
GPU=$ROOT/local-llamacpp-1650ti-2b-q4_0
D=2026-09-22
OUT_CPU=$CPU/benchmarks/runs/$D-paired-sweep-cpu
OUT_GPU=$GPU/benchmarks/runs/$D-paired-sweep-1650ti
mkdir -p "$OUT_CPU" "$OUT_GPU"
LOG=$OUT_GPU/driver.log

pkg() { echo $(( $(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1) / 1000 )); }
pkgt() { sensors 2>/dev/null | awk '/Package id 0/{gsub(/[+°C]/,"",$4); print int($4)}'; }
gput() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits; }
thr() { cat /sys/devices/system/cpu/cpu0/thermal_throttle/package_throttle_count 2>/dev/null || echo na; }
state() { echo "$(date +%T) pkg=$(pkgt)C gpu=$(gput)C throttle=$(thr) $*" | tee -a "$LOG"; }

cooldown() {
  local t0=$SECONDS
  sleep 120
  while :; do
    local p g; p=$(pkgt); g=$(gput)
    if [ "$p" -le 50 ] && [ "$g" -le 45 ]; then break; fi
    if [ $((SECONDS - t0)) -ge 900 ]; then state "cooldown TIMEOUT"; break; fi
    sleep 10
  done
  state "cooldown done after $((SECONDS - t0))s"
}

run_arm() {  # dir name expect pass
  local dir=$1 rig=$2 exp=$3 pass=$4 out=$5
  state "START $exp pass$pass"
  if [ "$exp" = gpu ]; then
    (cd "$dir" && setsid nohup make serve THREADS=6 THREADS_BATCH=12 > "$out/pass$pass-server.log" 2>&1 &)
  else
    (cd "$dir" && setsid nohup make serve > "$out/pass$pass-server.log" 2>&1 &)
  fi
  for i in $(seq 1 120); do curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 2; done
  curl -fsS http://127.0.0.1:8080/health >/dev/null || { state "server FAILED"; return 1; }
  state "healthy $exp pass$pass"
  (cd "$dir" && python3 sweep.py --base http://127.0.0.1:8080/v1 --out "$out/pass$pass" \
      --rig "$rig" --expect-device "$exp") > "$out/pass$pass/sweep.log" 2>&1
  local rc=$?
  state "END $exp pass$pass rc=$rc"
  pkill -f "llama.cpp/build.*/llama-server" ; sleep 3
  while ss -ltn | grep -q ':8080 '; do sleep 1; done
}

mkdir -p "$OUT_CPU/pass1" "$OUT_CPU/pass2" "$OUT_GPU/pass1" "$OUT_GPU/pass2"
state "BEGIN ABBA"; cooldown
run_arm "$CPU" local-llamacpp-cpu-2b-q4_0 cpu 1 "$OUT_CPU"; cooldown
run_arm "$GPU" local-llamacpp-1650ti-2b-q4_0 gpu 1 "$OUT_GPU"; cooldown
run_arm "$GPU" local-llamacpp-1650ti-2b-q4_0 gpu 2 "$OUT_GPU"; cooldown
run_arm "$CPU" local-llamacpp-cpu-2b-q4_0 cpu 2 "$OUT_CPU"
state "DONE"
cp "$LOG" "$OUT_CPU/driver.log"
