#!/bin/bash
# run_gpu.sh <run-prefix> -- the same read as tpu/run_quant.sh, on one NVIDIA GPU with the stock
# vllm/vllm-openai image and no patches: does upstream vLLM on CUDA load a checkpoint as it stands?
# Metadata: jev-gcs-models (one gs:// checkpoint directory, served as /work/models/<basename>),
# jev-gpu-image (default vllm/vllm-openai:latest; the digest is logged). Results and logs go to
# gs://$BUCKET/jev-tpu-31b/<run-prefix>/; the VM deletes itself at the end.
set -u
PREFIX="$1"
BUCKET=aisprint-491218-bucket
DST="gs://$BUCKET/jev-tpu-31b/$PREFIX"
W=/opt/jev-tpu; L=/opt/jev-tpu-logs; mkdir -p $L
LOG=$L/run.log
md() { curl -s -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }
attr() { local v; v=$(md attributes/$1); [ -n "$v" ] && [ "${v:0:1}" != "<" ] && printf '%s' "$v"; }
ZONE=$(md zone | awk -F/ '{print $NF}')
log() { echo "[jev-gpu $(date -u +%FT%TZ)] $*" | tee -a $LOG > /dev/console; }
sync_up() { gcloud storage rsync -r -q $W/results "$DST/results" >/dev/null 2>&1; gcloud storage rsync -r -q $L "$DST/logs" >/dev/null 2>&1; }
finish() { log "DONE: $1"; docker logs vllm > $L/vllm.full.log 2>&1; sync_up; gcloud compute instances delete "$(hostname)" --zone "$ZONE" --quiet; exit 0; }
cx() { docker exec -e JEV_TOPK=32 -w /work vllm "$@"; }

IMAGE=$(attr jev-gpu-image) || IMAGE=vllm/vllm-openai:latest
log "start $PREFIX on $ZONE"
for i in $(seq 1 60); do nvidia-smi >/dev/null 2>&1 && break; sleep 10; done
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > $L/gpu.txt 2>&1 || finish "no GPU driver"
log "gpu $(cat $L/gpu.txt)"
docker pull -q "$IMAGE" >> $LOG 2>&1 || finish "image pull failed"
log "image $(docker inspect --format '{{index .RepoDigests 0}}' "$IMAGE")"

SRC=$(attr jev-gcs-models) || finish "no jev-gcs-models"
NAME=$(basename "$SRC"); MODEL=/work/models/$NAME
mkdir -p $W/models && gcloud storage rsync -r -q "$SRC" "$W/models/$NAME" >> $LOG 2>&1 || finish "could not copy $SRC"
log "copied $NAME: $(du -sh $W/models/$NAME | cut -f1)"

# Suite, copied in and re-checked against Nimble's manifests (as tpu/run_quant.sh).
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
sync_up

# The flags of tpu/serve.sh, so the read is comparable record for record.
log "serving $MODEL"
docker rm -f vllm >/dev/null 2>&1
docker run -d --name vllm --gpus all --ipc=host --net=host -v $W:/work "$IMAGE" \
  "$MODEL" --served-model-name "$MODEL" --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 1 --max-model-len 2048 --max-num-seqs 16 --max-logprobs 32 \
  --generation-config vllm --limit-mm-per-prompt '{"image":0,"audio":0,"video":0}' --enable-prefix-caching >/dev/null
START=$(date +%s)
until curl -sf localhost:8000/v1/models | grep -q '"id"'; do
  if [ -z "$(docker ps -q -f name=vllm)" ] || [ $(( $(date +%s) - START )) -gt 3600 ]; then
    docker logs vllm > $L/boot.log 2>&1
    finish "FAILED to serve after $(( $(date +%s) - START ))s: $(grep -aoE '[A-Za-z_.]*(Error|Exception): .*' $L/boot.log | grep -av 'See root cause above' | tail -1 | cut -c1-400)"
  fi
  sleep 15
done
docker logs vllm > $L/boot.log 2>&1
log "READY after $(( $(date +%s) - START ))s"
grep -aE "Model loading took|model weights took|Available KV cache memory|GPU KV cache size|Maximum concurrency|quantization|MoE|Marlin|kernel" $L/boot.log | grep -v "Warning" | cut -c1-300 > $L/memory.txt
log "memory: $(grep -aE 'Model loading took|GPU KV cache size' $L/memory.txt | sed -E 's/.*\] //' | tr '\n' ' ')"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > $L/nvidia-smi.txt 2>&1

run="$PREFIX-26b-q4w4"
cx pip install -q pybase64 >/dev/null 2>&1
cx python3 -c "import vllm;print('vllm',vllm.__version__)" > $L/version.txt 2>&1
log "$(cat $L/version.txt)"
cx python3 run_eval.py --arm autoregressive --upstream http://localhost:8000 --model "$MODEL" --run "$run-smoke" --limit 5 --concurrency 2 >> $LOG 2>&1
ok=$(python3 - "$W/results/$run-smoke" <<'PY'
import glob, json, sys
rs = [json.loads(l) for f in glob.glob(sys.argv[1] + "/*.jsonl") for l in open(f)]
ret = [r["reads"][0]["labels_returned"] for r in rs if "reads" in r]
print(("ok" if rs and all(x >= 1 for x in ret) else "bad") + f" {len(rs)} records, labels returned {ret}")
PY
)
log "smoke: $ok"
[ "${ok%% *}" = ok ] || finish "smoke failed"
t0=$(date +%s)
cx python3 run_eval.py --arm autoregressive --upstream http://localhost:8000 --model "$MODEL" --run "$run" --concurrency 8 >> $LOG 2>&1
log "four tasks: $(( $(date +%s) - t0 ))s for $(cat $W/results/$run/*.jsonl | wc -l) decisions"
cx python3 run_eval.py --arm autoregressive --upstream http://localhost:8000 --model "$MODEL" --run "$run" --concurrency 8 --variant reversed >> $LOG 2>&1
cx python3 run_eval.py --arm autoregressive --upstream http://localhost:8000 --model "$MODEL" --run "$run-latency" --limit 100 --concurrency 1 >> $LOG 2>&1
mkdir -p $W/suite && cp -r /opt/suite/public $W/suite/ 2>/dev/null
t0=$(date +%s)
cx python3 nimble_suite/run_suite.py --arm autoregressive --upstream http://localhost:8000 --model "$MODEL" --records /work/suite/public --run "$run-suite" --concurrency 8 >> $LOG 2>&1
log "suite: $(( $(date +%s) - t0 ))s for $(cat $W/results/$run-suite/*.jsonl | wc -l) records"
python3 $W/tpu/w4a16_client.py load "$MODEL" $L/26b-q4w4.load.json > /dev/null 2>&1
log "load: $(python3 -c "import json;d=json.load(open('$L/26b-q4w4.load.json'));print(d['output_tok_per_s'],'tok/s, range',d['output_tok_per_s_min'],'to',d['output_tok_per_s_max'])" 2>&1)"
finish "all steps attempted"
