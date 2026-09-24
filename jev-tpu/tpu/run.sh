#!/bin/bash
# run.sh <run-prefix> -- the whole measurement on the v6e-1 VM (see PREREGISTRATION.md).
# Registered arms first, then the exploratory quantized probes. Every model's results and
# logs are copied to gs://$BUCKET/jev-tpu/<run-prefix>/ as soon as that model finishes, and
# the VM deletes itself at the end.
set -u
PREFIX="$1"
BUCKET=aisprint-491218-bucket
DST="gs://$BUCKET/jev-tpu/$PREFIX"
W=/opt/jev-tpu; L=/opt/jev-tpu-logs; mkdir -p $L
LOG=$L/run.log
log() { echo "[jev-run $(date -u +%FT%TZ)] $*" | tee -a $LOG > /dev/console; }
sync_up() { gcloud storage rsync -r -q $W/results "$DST/results" >/dev/null 2>&1; gcloud storage rsync -r -q $L "$DST/logs" >/dev/null 2>&1; }
cx() { docker exec -w /work vllm "$@"; }

# Suite, copied in and re-checked against Nimble's manifests.
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
print("ok" if rs and all(r["labels_returned"] == len(r["labels"]) for r in rs) else f"bad {len(rs)}")
PY
)
  log "$tag smoke: $ok"
  [ "$ok" = ok ] || { sync_up; return 1; }
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

try_model() {  # try_model <model> <tag> [extra flags]; 0 if it served and ran
  local model=$1 tag=$2; shift 2
  log "serving $model $*"
  if bash $W/tpu/serve.sh "$model" "$@" >> $LOG 2>&1; then
    log "$(tail -1 $LOG)"; full_run "$model" "$tag"; local rc=$?
  else
    log "$(grep -E '^(FAILED|READY)' $LOG | tail -1)"; local rc=1
  fi
  docker rm -f vllm >/dev/null 2>&1; rm -rf /dev/shm/hf; sync_up; return $rc
}

log "start $PREFIX on $(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}'), image $(docker inspect --format '{{index .RepoDigests 0}}' vllm/vllm-tpu:nightly)"
try_model google/gemma-4-E2B-it e2b
try_model google/gemma-4-E4B-it e4b
try_model google/gemma-4-12B-it 12b
try_model RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic 26b-fp8 || try_model cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit 26b-awq
try_model google/gemma-4-31B-it-qat-w4a16-ct 31b-w4a16 || try_model cyankiwi/gemma-4-31B-it-AWQ-4bit 31b-awq
log "DONE"
sync_up
gcloud compute instances delete "$(hostname)" --zone "$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')" --quiet
