#!/usr/bin/env bash
# Runs on the EC2 L4 instance for run 2026-09-24-l4-suite (see PREREGISTRATION.md).
# Rebuilds the 13-subset suite and checks its hashes, reads it with three arms
# against localhost, serves the results on :8000, then powers off (terminate).
set -uxo pipefail
REPO=/opt/jev/repo/jev; W=/opt/jev/suite; OUT=/opt/jev/out; mkdir -p $W $OUT /opt/hf
log() { echo "[$(date -Is)] $*" | tee -a $OUT/suite.log; }
IMAGE=vllm/vllm-openai:nightly-e9757321527ca1ecd514c07c1418dd2c53da3d19
RUN=2026-09-24-l4-suite

docker pull $IMAGE
docker pull python:3.12
docker run --rm -v $REPO:/repo:ro -v $W:/w python:3.12 bash -c \
  "pip install -q pyarrow && bash /repo/nimble_suite/build.sh /w" > $OUT/build.log 2>&1
log "build exit $? ; $(grep -c ^MATCH $OUT/build.log) of 13 subsets match"

serve() {
  local model="$1"; shift
  docker rm -f vllm >/dev/null 2>&1; rm -rf /opt/hf/hub; df -h / | tail -1 >> $OUT/suite.log
  set +x
  HF_TOKEN=$(aws secretsmanager get-secret-value --region us-east-1 --secret-id vllm/hf-token --query SecretString --output text 2>/dev/null || true)
  docker run -d --name vllm --ipc=host --gpus all -e HF_TOKEN="$HF_TOKEN" -e HF_HUB_DISABLE_XET=1 \
    -v /opt/hf:/root/.cache/huggingface -v $REPO:/work -v $W:/suite:ro -p 127.0.0.1:8000:8000 $IMAGE \
    --model "$model" --served-model-name "$model" --host 0.0.0.0 --port 8000 \
    --attention-backend TRITON_ATTN --gpu-memory-utilization 0.92 \
    --max-model-len 2048 --max-num-seqs 16 --max-logprobs 32 \
    --enforce-eager --enable-prefix-caching "$@"
  set -x
  for i in $(seq 1 120); do curl -sf localhost:8000/v1/models | grep -q '"id"' && { log "ready $model after $((i*10))s"; return 0; }; sleep 10; done
  log "TIMEOUT $model"; docker logs vllm > $OUT/fail-$(echo $model | tr / _).log 2>&1; return 1
}
run() { docker exec -w /work vllm python3 nimble_suite/run_suite.py --upstream http://localhost:8000 --records /suite/public --run $RUN "$@" >> $OUT/suite.log 2>&1; log "run exit $? $*"; }

serve cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit && run --arm autoregressive --model cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit --concurrency 8
serve cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4 --diffusion-config '{"canvas_length": 64}' && \
  run --arm diffusion --model cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4 --concurrency 4 --reads 4
serve google/gemma-4-E4B-it --dtype bfloat16 && run --arm autoregressive --model google/gemma-4-E4B-it --concurrency 8

set +x
{ nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  docker inspect --format 'image: {{.Config.Image}} digest: {{.Image}}' vllm
  docker exec vllm python3 -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__)"
  T=$(curl -s -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')
  echo "instance-type: $(curl -s -H "X-aws-ec2-metadata-token: $T" http://169.254.169.254/latest/meta-data/instance-type)"
  echo "az: $(curl -s -H "X-aws-ec2-metadata-token: $T" http://169.254.169.254/latest/meta-data/placement/availability-zone)"
} > $OUT/host.txt 2>&1
set -x
docker rm -f vllm >/dev/null 2>&1
cp -r $REPO/results/$RUN $OUT/results 2>/dev/null
log DONE; touch $OUT/DONE
(cd /opt/jev && tar czf out.tgz out)
cd /opt/jev && timeout 3600 python3 -m http.server 8000 --bind 0.0.0.0
shutdown -h now
