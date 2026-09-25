#!/bin/bash
# run_quant.sh <run-prefix> -- the 4-bit accuracy read on the v6e-1 VM (see PREREGISTRATION.md).
# ../jev-tpu's run.sh with three changes: every arm serves the pinned image with the diffs in
# /opt/jev-tpu/patches applied to its tpu_inference, the boot budget is 60 minutes, and the JAX
# compile cache lives on the host (seeded from gs when metadata jev-xla-seed names a prefix,
# and uploaded under jev-xla-save when that is set).
# Metadata jev-arms, when set, replaces the default arm list: space-separated
# <model>=<tag>=<mode>, mode one of read (the label read), load (fixed-length throughput via
# w4a16_client.py) or both. When tests/ is in the bundle the unit tests run on the chip first,
# and with metadata jev-bench=1 so does w4a16_matmul_bench.py.
# Results and logs go to gs://$BUCKET/jev-tpu-31b/<run-prefix>/ as each arm finishes; the VM
# deletes itself at the end.
set -u
PREFIX="$1"
BUCKET=aisprint-491218-bucket
DST="gs://$BUCKET/jev-tpu-31b/$PREFIX"
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
  gcloud storage rsync -r -q $XLA_CACHE_DIR "gs://$BUCKET/jev-tpu-31b/xla-cache/$to" >/dev/null 2>&1
  log "compile cache saved to $to: $(ls $XLA_CACHE_DIR | wc -l) entries"
}
finish() { log "DONE: $1"; save_cache; sync_up; gcloud compute instances delete "$(hostname)" --zone "$ZONE" --quiet; exit 0; }
cx() { docker exec -e JEV_TOPK=32 -w /work vllm "$@"; }

log "start $PREFIX on $ZONE"
docker pull -q "$BASE" >> $LOG 2>&1 || finish "image pull failed"
log "image $(docker inspect --format '{{index .RepoDigests 0}}' "$BASE")"

# Patched image: apply the diffs to the installed tpu_inference, commit a derived image.
SITE=$(docker run --rm --entrypoint python3 "$BASE" -c "import os,tpu_inference;print(os.path.dirname(os.path.dirname(tpu_inference.__file__)))" 2>/dev/null | tail -1)
docker rm -f patch >/dev/null 2>&1
docker create --name patch "$BASE" >/dev/null  # never started; commit keeps the base entrypoint
rm -rf /opt/ti && mkdir -p /opt/ti && docker cp "patch:$SITE/tpu_inference" /opt/ti/
for p in kvshare.diff wna16.diff unified.diff; do
  log "patch $p sha256 $(sha256sum $W/patches/$p | cut -c1-16)"
  (cd /opt/ti && patch -p1 --dry-run < $W/patches/$p > $L/patch-$p.log 2>&1 && patch -p1 < $W/patches/$p >> $L/patch-$p.log 2>&1) \
    || finish "patch $p did not apply: $(grep -iE 'fail|reject' $L/patch-$p.log | head -2 | tr '\n' ' ')"
done
docker cp /opt/ti/tpu_inference/. "patch:$SITE/tpu_inference/"
docker commit patch "$PATCHED" >/dev/null && docker rm -f patch >/dev/null

if SEED=$(attr jev-xla-seed); then
  mkdir -p $XLA_CACHE_DIR && gcloud storage rsync -r -q "gs://$BUCKET/jev-tpu-31b/xla-cache/$SEED" $XLA_CACHE_DIR >/dev/null 2>&1
  log "compile cache seeded from $SEED: $(ls $XLA_CACHE_DIR | wc -l) entries"
fi

# Unit tests and the per-layer bench on the chip, before any model holds it.
if [ -d $W/tests ]; then
  docker run --rm --privileged --net=host -v $W/tests:/tests --entrypoint bash "$PATCHED" -c \
    "pip install -q pytest >/dev/null 2>&1; cd /tests && python3 -m pytest -q -p no:cacheprovider ." > $L/pytest.log 2>&1
  log "pytest: $(tail -1 $L/pytest.log)"
fi
if [ "$(attr jev-bench)" = 1 ]; then
  docker run --rm --privileged --net=host -v $W:/w --entrypoint python3 "$PATCHED" /w/tpu/w4a16_matmul_bench.py > $L/matmul-bench.txt 2>&1
  log "matmul bench: $(tail -1 $L/matmul-bench.txt)"
fi
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

try_model() {  # try_model <model> <tag> [read|load|both]
  local model=$1 tag=$2 mode=${3:-read}
  log "serving $model as $tag ($mode)"
  if bash $W/tpu/serve.sh "$model" >> $LOG 2>&1; then
    log "$(tail -1 $LOG)"
    if [ "$mode" != load ]; then full_run "$model" "$tag"; fi
    if [ "$mode" != read ]; then
      python3 $W/tpu/w4a16_client.py load "$model" $L/$tag.load.json > /dev/null 2>&1
      log "$tag load: $(python3 -c "import json;d=json.load(open('$L/$tag.load.json'));print(d['output_tok_per_s'],'tok/s, range',d['output_tok_per_s_min'],'to',d['output_tok_per_s_max'])" 2>&1)"
    fi
  else
    log "$(grep -E '^(FAILED|READY)' $LOG | tail -1)"
  fi
  docker rm -f vllm >/dev/null 2>&1; rm -rf /dev/shm/hf; sync_up
}

if ARMS=$(attr jev-arms); then
  for arm in $ARMS; do
    IFS='=' read -r model tag mode <<< "$arm"
    try_model "$model" "$tag" "$mode"
  done
else
  try_model google/gemma-4-E2B-it-qat-w4a16-ct e2b-w4a16
  try_model google/gemma-4-E4B-it-qat-w4a16-ct e4b-w4a16
  try_model google/gemma-4-12B-it 12b-bf16
  try_model google/gemma-4-12B-it-qat-w4a16-ct 12b-w4a16
  try_model google/gemma-4-31B-it-qat-w4a16-ct 31b-w4a16
fi
finish "all arms attempted"
