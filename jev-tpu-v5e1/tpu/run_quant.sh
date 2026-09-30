#!/bin/bash
# run_quant.sh <run-prefix> -- the 4-bit accuracy read on the v5e-1 TPU VM (see PREREGISTRATION.md).
# ../jev-tpu-31b's run_quant.sh retargeted to v5e: the node is a Cloud TPU queued resource (v5e has
# no Compute Engine path, ../HARDWARE.md), so it deletes itself through metadata jev-qr rather than
# `compute instances delete`, and the default arms are the ones that fit 14.49 GiB. Like the
# original, it is ../jev-tpu's run.sh with three changes: every arm serves the pinned image with the diffs in
# /opt/jev-tpu/patches applied to its tpu_inference, the boot budget is 60 minutes, and the JAX
# compile cache lives on the host (seeded from gs when metadata jev-xla-seed names a prefix,
# and uploaded under jev-xla-save when that is set).
# Metadata jev-arms, when set, replaces the default arm list: space-separated
# <model>=<tag>=<mode>, mode a +-joined list of read (the label read), load (fixed-length throughput via
# w4a16_client.py), long (the same with long, never-cached prompts: loadlong, at each length in
# metadata jev-long-input, default "1024 3584") and gen (gen_eval.py: GSM8K and BFCL simple; metadata
# jev-gen-max-tokens sets GSM8K's answer limit, default 768), or
# both (read+load); and an optional fourth field, comma-separated serve flags: override
# (--hf-overrides to Gemma4ForCausalLM), override-nolimit (the same, without
# --limit-mm-per-prompt), kv-bf16 (--kv-cache-dtype bfloat16; v6e picks fp8 by itself, v5e is unmeasured),
# tools (--enable-auto-tool-choice --tool-call-parser gemma4, which gen's BFCL task needs),
# gmu=<x>, mml=<n> and blocks=<n> (--gpu-memory-utilization, --max-model-len,
# --num-gpu-blocks-override for this arm only; they follow jev-serve-args, so they win). Metadata jev-patches, when set, replaces the list of diffs applied
# (default: kvshare.diff wna16.diff; 12B loads with the override instead of unified.diff). Metadata jev-tp sets the tensor-parallel size
# for every arm (default 1). Metadata jev-gcs-models, when set, names
# space-separated gs:// checkpoint directories copied to /opt/jev-tpu/models/<basename>, which
# an arm serves as /work/models/<basename>. When tests/ is in the bundle the unit tests run on the chip first,
# and with metadata jev-bench=1 so does w4a16_matmul_bench.py (or jev-bench names scripts in tpu/).
# A patch named tokamax-*.diff applies to the image's tokamax instead of tpu_inference. Metadata
# jev-env (space-separated KEY=VALUE) is set in every arm's vLLM container, and metadata
# jev-serve-args (space-separated, no quoting) is appended to every arm's serve flags. Metadata
# jev-swap-gb adds a swap file of that size before serving. Metadata jev-load-conc (default 16) lists the
# concurrencies the load mode runs at. Metadata jev-boot-timeout sets each arm's boot budget in seconds (default 3600). Host memory is sampled every 30 s
# into logs/<tag>.hostmem.txt while an arm boots, and the kernel's OOM-killer lines are kept on failure.
# Results and logs go to gs://$BUCKET/jev-tpu-v5e1/<run-prefix>/ as each arm finishes; the VM
# deletes its queued resource (metadata jev-qr) at the end, or itself as a Compute Engine instance
# (metadata jev-self-delete=instance, the v6e path).
set -u
PREFIX="$1"
BUCKET=aisprint-491218-bucket
DST="gs://$BUCKET/jev-tpu-v5e1/$PREFIX"
BASE=vllm/vllm-tpu@sha256:19a1a0526476f902eb83e1057f3d8938f35b457dd7716a30e9ab4f7bee90d507
PATCHED=jev-quant:patched
W=/opt/jev-tpu; L=/opt/jev-tpu-logs; mkdir -p $L
LOG=$L/run.log
export IMAGE=$PATCHED BOOT_TIMEOUT=3600 XLA_CACHE_DIR=/opt/xla-cache
md() { curl -s -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }
ZONE=$(md zone | awk -F/ '{print $NF}')
log() { echo "[jev-quant $(date -u +%FT%TZ)] $*" | tee -a $LOG > /dev/console; }
sync_up() { gcloud storage rsync -r -q $W/results "$DST/results" >/dev/null 2>&1; gcloud storage rsync -r -q $L "$DST/logs" >/dev/null 2>&1; }
attr() { local v; v=$(md attributes/$1); [ -n "$v" ] && [ "${v:0:1}" != "<" ] && printf '%s' "$v"; }
save_cache() {
  local to; to=$(attr jev-xla-save) || return 0
  gcloud storage rsync -r -q $XLA_CACHE_DIR "gs://$BUCKET/jev-tpu-v5e1/xla-cache/$to" >/dev/null 2>&1
  log "compile cache saved to $to: $(ls $XLA_CACHE_DIR | wc -l) entries"
}
finish() {
  log "DONE: $1"; save_cache; sync_up
  local qr
  if ! qr=$(attr jev-qr); then
    # A Compute Engine TPU VM (v6e) has no queued resource; metadata jev-self-delete=instance deletes it.
    if [ "$(attr jev-self-delete)" = instance ]; then
      sync_up; gcloud compute instances delete "$(hostname)" --zone "$ZONE" --quiet >> $LOG 2>&1; exit 0
    fi
    log "no jev-qr metadata: leaving the node up, delete it by hand"; exit 0
  fi
  gcloud alpha compute tpus queued-resources delete "$qr" --zone "$ZONE" --force --quiet --async >> $LOG 2>&1
  sync_up; exit 0
}
cx() { docker exec -e JEV_TOPK=32 -w /work vllm "$@"; }

log "start $PREFIX on $ZONE"
for i in 1 2 3 4 5; do docker pull -q "$BASE" >> $LOG 2>&1 && break; log "image pull attempt $i failed"; sleep 30; done
docker image inspect "$BASE" >/dev/null 2>&1 || finish "image pull failed"
log "image $(docker inspect --format '{{index .RepoDigests 0}}' "$BASE")"

# Patched image: apply the diffs to the installed tpu_inference, commit a derived image.
SITE=$(docker run --rm --entrypoint python3 "$BASE" -c "import os,tpu_inference;print(os.path.dirname(os.path.dirname(tpu_inference.__file__)))" 2>/dev/null | tail -1)
docker rm -f patch >/dev/null 2>&1
docker create --name patch "$BASE" >/dev/null  # never started; commit keeps the base entrypoint
rm -rf /opt/ti && mkdir -p /opt/ti && docker cp "patch:$SITE/tpu_inference" /opt/ti/
# tokamax is a regular site-packages install, not beside the source-installed tpu_inference.
TKSITE=$(docker run --rm --entrypoint python3 "$BASE" -c "import os,tokamax;print(os.path.dirname(os.path.dirname(tokamax.__file__)))" 2>/dev/null | tail -1)
rm -rf /opt/tk && mkdir -p /opt/tk && docker cp "patch:$TKSITE/tokamax" /opt/tk/
PATCHES=$(attr jev-patches) || PATCHES="kvshare.diff wna16.diff"
TP=$(attr jev-tp) || TP=1; export TP
BOOT_TIMEOUT=$(attr jev-boot-timeout) || BOOT_TIMEOUT=3600; export BOOT_TIMEOUT  # seconds per arm
log "tensor parallel size $TP"
for p in $PATCHES; do
  log "patch $p sha256 $(sha256sum $W/patches/$p | cut -c1-16)"
  dir=/opt/ti; case "$p" in tokamax-*) dir=/opt/tk ;; esac
  (cd $dir && patch -p1 --dry-run < $W/patches/$p > $L/patch-$p.log 2>&1 && patch -p1 < $W/patches/$p >> $L/patch-$p.log 2>&1) \
    || finish "patch $p did not apply: $(grep -iE 'fail|reject' $L/patch-$p.log | head -2 | tr '\n' ' ')"
done
docker cp /opt/ti/tpu_inference/. "patch:$SITE/tpu_inference/"
docker cp /opt/tk/tokamax/. "patch:$TKSITE/tokamax/"
VLLM_ENV=$(attr jev-env) || VLLM_ENV=; export VLLM_ENV
SERVE_ARGS=$(attr jev-serve-args) || SERVE_ARGS=
[ -n "$VLLM_ENV$SERVE_ARGS" ] && log "vllm env: ${VLLM_ENV:-none}; extra serve args: ${SERVE_ARGS:-none}"
docker commit patch "$PATCHED" >/dev/null && docker rm -f patch >/dev/null

if GCS_MODELS=$(attr jev-gcs-models); then
  for src in $GCS_MODELS; do
    t0=$(date +%s)
    mkdir -p $W/models && gcloud storage rsync -r -q "$src" "$W/models/$(basename "$src")" >> $LOG 2>&1 \
      || finish "could not copy $src"
    log "copied $(basename "$src"): $(du -sh "$W/models/$(basename "$src")" | cut -f1) in $(( $(date +%s) - t0 ))s"
  done
fi

if SWAP=$(attr jev-swap-gb); then
  # Never let the swap file fill the disk: copied checkpoints can already hold most of it, and with the
  # disk full nothing can be written, including the log upload, so the run ends with no record at all.
  FREE=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
  if [ "$SWAP" -gt $(( FREE - 10 )) ]; then log "swap capped at $(( FREE - 10 ))G (asked ${SWAP}G, ${FREE}G free)"; SWAP=$(( FREE - 10 )); fi
  [ "$SWAP" -gt 0 ] && { fallocate -l ${SWAP}G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile; } >> $LOG 2>&1
  log "swap: $(free -g | awk '/^Swap:/{print $2}') GiB; disk free: $(df -h / | awk 'NR==2{print $4}')"
fi

if SEED=$(attr jev-xla-seed); then
  mkdir -p $XLA_CACHE_DIR && gcloud storage rsync -r -q "gs://$BUCKET/jev-tpu-v5e1/xla-cache/$SEED" $XLA_CACHE_DIR >/dev/null 2>&1
  log "compile cache seeded from $SEED: $(ls $XLA_CACHE_DIR | wc -l) entries"
fi

# Unit tests and the per-layer bench on the chip, before any model holds it.
if [ -d $W/tests ]; then
  docker run --rm --privileged --net=host -v $W/tests:/tests --entrypoint bash "$PATCHED" -c \
    "pip install -q pytest >/dev/null 2>&1; cd /tests && python3 -m pytest -q -p no:cacheprovider ." > $L/pytest.log 2>&1
  log "pytest: $(tail -1 $L/pytest.log)"
fi
for bench in $(attr jev-bench); do  # 1 means w4a16_matmul_bench.py; otherwise script names in tpu/
  [ "$bench" = 1 ] && bench=w4a16_matmul_bench.py
  docker run --rm --privileged --net=host -v $W:/w --entrypoint python3 "$PATCHED" /w/tpu/$bench > $L/${bench%.py}.txt 2>&1
  log "$bench: $(tail -1 $L/${bench%.py}.txt)"
done
sync_up

# Suite, copied in and re-checked against Nimble's manifests (as ../jev-tpu).
mkdir -p /opt/suite && gcloud storage cp -q "gs://$BUCKET/jev-tpu/inputs/suite.tgz" /opt/suite/ && tar xzf /opt/suite/suite.tgz -C /opt/suite
python3 - /opt/suite/public /opt/suite/nimble/docs/assets/public-benchmarks/subsets > $L/suite-checksums.txt <<'PY'
import hashlib, json, os, sys
O, M = sys.argv[1:3]
for f in sorted(os.listdir(M)):
    sub = f[: -len("-manifest.json")]
    want = json.load(open(f"{M}/{f}"))["dataset_sha256"]
    got = hashlib.sha256(open(f"{O}/{sub}/all.jsonl", "rb").read()).hexdigest()
    print("MATCH" if got == want else "DIFF", sub, sum(1 for _ in open(f"{O}/{sub}/all.jsonl")))
PY
log "suite: $(grep -c ^MATCH $L/suite-checksums.txt) of 13 subsets match"
grep ^DIFF $L/suite-checksums.txt | awk '{print $2}' | while read s; do rm -rf /opt/suite/public/$s; done

# The read against one served model: smoke, four tasks, reversed options, latency, suite.
full_run() {
  local model=$1 tag=$2 run="$PREFIX-$2"
  cx pip install -q pybase64 >/dev/null 2>&1
  cx python3 -c "import vllm,sys;print('vllm',vllm.__version__)" > $L/$tag.version.txt 2>&1
  cx python3 run_eval.py --arm autoregressive --upstream http://localhost:8000 --model "$model" --run "$run-smoke" --limit 5 --concurrency 2 >> $LOG 2>&1
  local ok
  ok=$(python3 - "$W/results/$run-smoke" <<'PY'
import glob, json, sys
rs = [json.loads(l) for f in glob.glob(sys.argv[1] + "/*.jsonl") for l in open(f)]
ret = [r["reads"][0]["labels_returned"] for r in rs if "reads" in r]
print(("ok" if rs and all(x >= 1 for x in ret) else "bad") + f" {len(rs)} records, labels returned {ret}")
PY
)
  log "$tag smoke: $ok"
  [ "${ok%% *}" = ok ] || { sync_up; return 1; }
  local t0=$(date +%s)
  cx python3 run_eval.py --arm autoregressive --upstream http://localhost:8000 --model "$model" --run "$run" --concurrency 8 >> $LOG 2>&1
  log "$tag four tasks: $(( $(date +%s) - t0 ))s for $(cat $W/results/$run/*.jsonl | wc -l) decisions at concurrency 8"
  cx python3 run_eval.py --arm autoregressive --upstream http://localhost:8000 --model "$model" --run "$run" --concurrency 8 --variant reversed >> $LOG 2>&1
  cx python3 run_eval.py --arm autoregressive --upstream http://localhost:8000 --model "$model" --run "$run-latency" --limit 100 --concurrency 1 >> $LOG 2>&1
  mkdir -p $W/suite && cp -r /opt/suite/public $W/suite/ 2>/dev/null
  t0=$(date +%s)
  cx python3 nimble_suite/run_suite.py --arm autoregressive --upstream http://localhost:8000 --model "$model" --records /work/suite/public --run "$run-suite" --concurrency 8 >> $LOG 2>&1
  log "$tag suite: $(( $(date +%s) - t0 ))s for $(cat $W/results/$run-suite/*.jsonl | wc -l) records"
  sync_up
}

try_model() {  # try_model <model> <tag> [read|load|both] [override|override-nolimit|kv-bf16]
  local model=$1 tag=$2 mode=${3:-read} flags=${4:-} extra=() mm
  has() { [[ "+$mode+" == *"+$1+"* ]]; }
  mm='{"image":0,"audio":0,"video":0}'
  local arm_args=()
  for f in ${flags//,/ }; do  # comma-separated; per-arm flags go after jev-serve-args, so they win
    case "$f" in
      override) extra+=(--hf-overrides '{"architectures":["Gemma4ForCausalLM"]}') ;;
      override-nolimit) extra+=(--hf-overrides '{"architectures":["Gemma4ForCausalLM"]}'); mm= ;;
      kv-bf16) extra+=(--kv-cache-dtype bfloat16) ;;
      tools) arm_args+=(--enable-auto-tool-choice --tool-call-parser gemma4) ;;
      gmu=*) arm_args+=(--gpu-memory-utilization "${f#gmu=}") ;;
      mml=*) arm_args+=(--max-model-len "${f#mml=}") ;;
      blocks=*) arm_args+=(--num-gpu-blocks-override "${f#blocks=}") ;;
      *) log "unknown arm flag $f" ;;
    esac
  done
  extra+=($SERVE_ARGS "${arm_args[@]}")
  log "serving $model as $tag ($mode${flags:+, $flags})"
  ( while :; do echo "$(date -u +%T) $(free -m | awk '/^Mem:/{print "used",$3,"avail",$7} /^Swap:/{print "swap_used",$3}' | tr '\n' ' ')"; sleep 30; done ) > $L/$tag.hostmem.txt 2>&1 &
  local memlog=$!
  if MM_LIMIT=$mm bash $W/tpu/serve.sh "$model" "${extra[@]}" >> $LOG 2>&1; then
    log "$(tail -1 $LOG)"
    [ "$mode" = both ] && mode=read+load
    if has read; then full_run "$model" "$tag"; fi
    if has gen; then
      for task in gsm8k bfcl_simple; do
        JEV_GEN_MAX_TOKENS=$(attr jev-gen-max-tokens || echo 768) \
          python3 $W/gen_eval.py run http://localhost:8000 "$model" $task $W/results/$PREFIX-$tag-gen/$task.jsonl > $L/$tag.gen-$task.json 2>> $LOG
        log "$tag $task: $(cat $L/$tag.gen-$task.json)"
      done
      sync_up
    fi
    if has long; then
      for n in $(attr jev-long-input || echo "1024 3584"); do  # metadata jev-long-input: prompt lengths in tokens
        for c in $(attr jev-load-conc || echo 16); do
          JEV_LOAD_INPUT_TOKENS=$n JEV_LOAD_CONCURRENCY=$c python3 $W/tpu/w4a16_client.py loadlong "$model" $L/$tag.long$n.c$c.json > /dev/null 2>> $LOG
          log "$tag long $n at concurrency $c: $(python3 -c "import json;d=json.load(open('$L/$tag.long$n.c$c.json'));print(d['prompt_tokens_per_request'],'prompt tokens,',d['output_tok_per_s'],'out tok/s,',d['total_tok_per_s'],'total tok/s, ttft',d['ttft_median_s'],'s')" 2>&1)"
        done
      done
      sync_up
    fi
    if has load; then
      for c in $(attr jev-load-conc || echo 16); do  # metadata jev-load-conc: space-separated concurrencies
        JEV_LOAD_CONCURRENCY=$c python3 $W/tpu/w4a16_client.py load "$model" $L/$tag.load.c$c.json > /dev/null 2>&1
        log "$tag load at concurrency $c: $(python3 -c "import json;d=json.load(open('$L/$tag.load.c$c.json'));print(d['output_tok_per_s'],'tok/s, range',d['output_tok_per_s_min'],'to',d['output_tok_per_s_max'])" 2>&1)"
      done
    fi
  else
    log "$(grep -E '^(FAILED|READY)' $LOG | tail -1)"
    dmesg -T 2>/dev/null | grep -iE 'out of memory|oom-kill|killed process' > $L/$tag.oom.txt
    log "$tag host memory at the end: $(tail -1 $L/$tag.hostmem.txt); kernel OOM lines: $(wc -l < $L/$tag.oom.txt)"
  fi
  kill $memlog 2>/dev/null
  # serve.sh names the boot log by checkpoint, so a later arm serving the same one overwrites it: keep this arm's.
  cp "$L/$(echo "$model" | tr '/' '_').boot.log" "$L/$tag.boot.log" 2>/dev/null
  # Speculative-decoding counters (acceptance) are only in the server log; keep them.
  docker logs vllm 2>&1 | grep -iE 'spec.?decod|acceptance|draft' > $L/$tag.spec.txt || true
  docker rm -f vllm >/dev/null 2>&1; rm -rf /dev/shm/hf; sync_up
}

if ARMS=$(attr jev-arms); then
  for arm in $ARMS; do
    IFS='=' read -r model tag mode flags <<< "$arm"
    try_model "$model" "$tag" "$mode" "$flags"
  done
else
  # What fits 14.49 GiB usable (../HARDWARE.md, ../MODELS.md): E2B at bf16, and the W4A16 QAT
  # exports up to 12B. E4B and 12B at bf16, 26B and 31B do not fit one v5e chip.
  try_model google/gemma-4-E2B-it e2b-bf16
  try_model google/gemma-4-E2B-it-qat-w4a16-ct e2b-w4a16
  try_model google/gemma-4-E4B-it-qat-w4a16-ct e4b-w4a16
  try_model google/gemma-4-12B-it-qat-w4a16-ct 12b-w4a16 read override
fi
finish "all arms attempted"
