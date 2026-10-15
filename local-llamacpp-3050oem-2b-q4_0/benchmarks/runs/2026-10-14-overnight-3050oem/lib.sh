# Shared by the overnight 3050 jobs. Sourced, not run.
ROOT=/home/xbill/gemma4-dev
BIN=/home/xbill/llama.cpp/build/bin/llama-server
M=/home/xbill/models
E2B=$M/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf
E4B=$M/gemma-4-E4B-it-qat-q4_0-exact/gemma-4-E4B-it-q4_0-exact.gguf
D2=$M/gemma-4-E2B-it-qat-assistant/gemma-4-E2B-it-qat-assistant-q4_0-exact.gguf
D4=$M/gemma-4-E4B-it-qat-assistant/gemma-4-E4B-it-qat-assistant-q4_0-exact.gguf
OT='per_layer_token_embd\.weight=CUDA0'
DRIVER=$ROOT/local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-overnight-3050oem/driver.log
state() { echo "$(date '+%F %T') $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader) $*" | tee -a "$DRIVER"; }
stop8080() { pkill -x llama-server; while pgrep -x llama-server >/dev/null; do sleep 1; done; }
cool() { until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" -le 45 ]; do sleep 5; done; }
# serve LOG CTX PARALLEL GGUF [DRAFTER N]: llama-server on 127.0.0.1:8080 with the rigs' flags.
serve() {
  local log=$1 ctx=$2 par=$3 m=$4 d=${5:-} n=${6:-}
  stop8080; cool
  local extra=()
  [ -n "$d" ] && extra=(-md "$d" --spec-type draft-mtp -ngld all --spec-draft-n-max "$n")
  setsid $BIN -m "$m" --host 127.0.0.1 --port 8080 -ngl 99 -c "$ctx" --parallel "$par" -ctk f16 -ctv f16 \
    -fa 1 -t 6 -tb 12 --reasoning off --metrics -ot "$OT" "${extra[@]}" > "$log" 2>&1 < /dev/null &
  local pid=$!
  for i in $(seq 300); do
    curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && { state "server up $(basename $log): $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader | tr '\n' ' ')"; return 0; }
    kill -0 $pid 2>/dev/null || { state "server FAILED $(basename $log)"; return 1; }
    sleep 1
  done
  state "server TIMEOUT $(basename $log)"; return 1
}
# commit_push DIR MESSAGE: commit only DIR (and nothing staged elsewhere), rebase, push.
commit_push() {
  local dir=$1 msg=$2
  ( cd $ROOT && git add "$dir" && \
    git -c user.name=xbill -c user.email=xbill9@gmail.com commit -q -m "$msg" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- "$dir" && \
    git -c user.name=xbill -c user.email=xbill9@gmail.com -c rebase.autoStash=true pull --rebase -q && git push -q ) \
    && state "PUSHED $dir: $(git -C $ROOT log --oneline -1 | cut -c1-80)" || state "COMMIT/PUSH FAILED $dir"
}
restore_live() {
  stop8080
  (cd $ROOT/local-llamacpp-3050oem-4b-q4_0 && setsid nohup make serve > run/llama-server.log 2>&1 < /dev/null &)
  for i in $(seq 120); do curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
  state "E4B live server restored: $(curl -fsS http://127.0.0.1:8080/health)"
}
