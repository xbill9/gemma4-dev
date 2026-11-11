#!/bin/bash
# --- 2026-11-11 phase 2: fill the T4's last gaps without an SSH session; results go to GCS ---
# mistral.rs (build, then queue.sh runs its one-request arms, load sweep and accuracy) and the SGLang int4
# load sweep. Runs only while metadata t4-queue=phase2.
if [ "$(curl -s -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/attributes/t4-queue)" = phase2 ]; then
cat > /usr/local/bin/t4-phase2.sh <<'EOF'
#!/bin/bash
U=xbill_glitnir_com; H=/home/$U; G=$H/gemma4-dev; B=gs://comglitn-gemma4-t4-results
Q=$G/gpu-vllm-t4-2b-q4w4a16emb4/benchmarks/runs/2026-11-05-engines-t4
S4=$G/gpu-sglang-t4-2b-q4w4a16emb4/benchmarks/runs/2026-11-10-int4-t4
L=/var/log/t4-phase2.log
log() { echo "$(date -Is) $*" >> $L; echo "t4-phase2: $*" > /dev/ttyS0; }
up() {
  for p in gpu-vllm-t4-2b-q4w4a16emb4/benchmarks/runs/2026-11-05-engines-t4 \
           gpu-mistralrs-t4-2b-q4_0/benchmarks/runs \
           gpu-sglang-t4-2b-q4w4a16emb4/benchmarks/runs/2026-11-10-int4-t4/concurrency \
           gpu-sglang-t4-2b-q4w4a16emb4/benchmarks/runs/2026-11-10-int4-t4/accuracy \
           gpu-litertlm-t4-2b-int248/benchmarks/runs/2026-11-06-accuracy-t4; do
    [ -d $G/$p ] && gcloud storage rsync -r -q -x '.*\.gguf$|.*\.safetensors$' $G/$p $B/$p >> $L.rsync 2>&1
  done
  gcloud storage cp -q $Q/queue.log $L $B/phase2/ >/dev/null 2>&1
  [ -f /opt1/mistralrs-build.log ] && gcloud storage cp -q /opt1/mistralrs-build.log $B/phase2/ >/dev/null 2>&1
}
for i in $(seq 60); do nvidia-smi >/dev/null 2>&1 && break; sleep 10; done
mountpoint -q /opt1 || mount /opt1
[ -n "$(awk 'NR>1' /proc/swaps)" ] || swapon /opt1/swapfile
log "start; nvidia-smi ok=$(nvidia-smi >/dev/null 2>&1 && echo y); swap=$(awk 'NR>1{print $1}' /proc/swaps)"
( while true; do sleep 600; up; done ) & UPL=$!
# 1. SGLang int4 sweep support (same edit as the committed t4chain.py)
sed -i 's|"sglang": engines.SRIG, "mistralrs"|"sglang": engines.SRIG, "sglang4": engines.SRIG4, "mistralrs"|' $S4/t4chain.py
log "t4chain sglang4 sweep: $(grep -c sglang4 $S4/t4chain.py)"
# 1b. mistral.rs accuracy with thinking off (2026-11-11: the first T4 run left it at its default, on); keep that
#     BFCL as the thinking-default diagnostic, as on the 3050
MA=$G/gpu-mistralrs-t4-2b-q4_0/benchmarks/runs/2026-11-06-accuracy-t4/mistralrs
if ! grep -q JEV_ENABLE_THINKING $Q/t4chain.py; then
  sed -i 's|^        tenv = dict(os.environ, PYTHONUSERBASE="")  # the harness scripts are stdlib only$|&\n        if ARM.startswith("mistralrs"):  # mistral.rs v0.9.5 thinks by default on Gemma 4; the 3050/L4 runs set it off\n            tenv["JEV_ENABLE_THINKING"] = "0"|' $Q/t4chain.py
  if [ -f $MA/bfcl_simple-mistralrs.jsonl ]; then
    mkdir -p $MA/thinkdefault; mv $MA/bfcl_simple-mistralrs.jsonl $MA/bfcl_simple-mistralrs.log $MA/driver.log $MA/server.log $MA/thinkdefault/ 2>/dev/null
    rm -f $MA/gsm8k-mistralrs.jsonl $MA/gsm8k-mistralrs.log $MA/status.json
  fi
  chown -R $U: $MA $Q/t4chain.py
fi
log "t4chain thinking-off lines: $(grep -c JEV_ENABLE_THINKING $Q/t4chain.py)"
# 2. SGLang int4 load sweep (first: it fits before the 22:00 UTC stop whatever the build does)
if ! grep -q '"rc": 0' $S4/concurrency/sglang4/status.json 2>/dev/null; then
  log "sglang4 sweep start"
  sudo -u $U -i bash -c "cd $S4 && python3 t4chain.py sweep sglang4 $S4/concurrency/sglang4 > /dev/null 2>&1"
  log "sglang4 sweep rc=$?"
fi
# 3. mistral.rs build (CPU only; nothing on the GPU meanwhile)
if [ ! -x /opt1/mistralrs-v0.9.5/bin/mistralrs ]; then
  # /opt1/cuda-13.4 has no libnvrtc / libcurand to link; torch's cu13 wheels carry both (2026-11-11 link error)
  mkdir -p /opt1/cuda-extra-lib
  for n in nvrtc curand; do
    f=$(find /opt1/pyuser-engines /opt1/cuda-13.4 -name "lib$n.so*" ! -name "*builtins*" ! -name "*alt*" 2>/dev/null | sort | head -1)
    log "lib$n: ${f:-missing}"; [ -n "$f" ] && ln -sf "$f" /opt1/cuda-extra-lib/lib$n.so
  done
  echo /opt1/cuda-extra-lib > /etc/ld.so.conf.d/t4-cuda-extra.conf
  for f in /opt1/cuda-extra-lib/*.so; do d=$(dirname "$(readlink -f "$f")"); grep -qx "$d" /etc/ld.so.conf.d/t4-cuda-extra.conf || echo "$d" >> /etc/ld.so.conf.d/t4-cuda-extra.conf; done
  ldconfig
  log "mistral.rs build start; mmproj present=$(test -f /opt1/models/gemma-4-E2B-it-mmproj/gemma-4-E2B-it-mmproj.gguf && echo y || echo n); src present=$(test -d /opt1/mistral.rs && echo y || echo n); cargo=$(sudo -u $U -i bash -c 'command -v cargo || ls ~/.cargo/bin/cargo' 2>/dev/null)"
  # a file, not bash -c: sudo -i joins a multi-line command into one line
  cat > /usr/local/bin/t4-mrs-build.sh <<'BEOF'
set -x
command -v cargo >/dev/null || [ -x ~/.cargo/bin/cargo ] || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
export PATH=$HOME/.cargo/bin:/opt1/cuda-13.4/bin:$PATH CUDA_HOME=/opt1/cuda-13.4 CUDA_PATH=/opt1/cuda-13.4 CPATH=/opt1/cuda-extra-include CARGO_BUILD_JOBS=2 LIBRARY_PATH=/opt1/cuda-extra-lib
test -d /opt1/mistral.rs || git clone --depth 50 --branch v0.9.5 https://github.com/EricLBuehler/mistral.rs.git /opt1/mistral.rs
cd /opt1/mistral.rs && ls && grep -m1 "^version" Cargo.toml; (git log -1 --format=%h || true)
CUDA_COMPUTE_CAP=75 cargo install --locked --path mistralrs-cli --features cuda --root /opt1/mistralrs-v0.9.5
BEOF
  chmod +x /usr/local/bin/t4-mrs-build.sh
  sudo -u $U -i bash /usr/local/bin/t4-mrs-build.sh > /opt1/mistralrs-build.log 2>&1
  log "mistral.rs build rc=$? binary=$(test -x /opt1/mistralrs-v0.9.5/bin/mistralrs && echo y || echo n)"
  up
fi
# 4. queue.sh: everything done is skipped; with the binary it runs mistral.rs one-request, sweep, accuracy
sudo -u $U -i bash -c "cd $Q && bash queue.sh >> queue-acc.out 2>&1 < /dev/null"
log "queue.sh exited: $(tail -1 $Q/queue.log)"; up
kill $UPL; up
log "phase2 done"; up
EOF
chmod +x /usr/local/bin/t4-phase2.sh
setsid nohup /usr/local/bin/t4-phase2.sh > /dev/null 2>&1 < /dev/null &
fi
# --- end phase 2 ---
