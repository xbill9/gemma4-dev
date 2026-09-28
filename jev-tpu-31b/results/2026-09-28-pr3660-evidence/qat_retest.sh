#!/bin/bash
# Retest the Gemma 4 QAT W4A16 (-qat-w4a16-ct) serving paths with PR #3660 (+ #3299) on one v6e chip.
# Image: vllm/vllm-tpu:nightly with the PR branch's tpu_inference installed over the image's copy.
# Per model: boot (serve.sh flags from jev-tpu-31b), greedy sample outputs, fixed-length load.
set -u
H=$HOME/qat; L=$H/logs; mkdir -p $L
log() { echo "[qat $(date -u +%T)] $*" | tee -a $L/run.log; }

if ! command -v docker >/dev/null; then sudo apt-get update -qq && sudo apt-get install -y -qq docker.io >/dev/null; fi
sudo docker pull -q vllm/vllm-tpu:nightly >/dev/null
log "base $(sudo docker inspect --format '{{index .RepoDigests 0}}' vllm/vllm-tpu:nightly)"

# HF token from Secret Manager.
PROJECT=$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/project/project-id)
gcloud secrets versions access latest --secret=hf-token --project=$PROJECT | sudo tee /root/hf_token >/dev/null
sudo test -s /root/hf_token && log "hf token ok" || { log "no hf token"; exit 1; }

# PR branch + #3299, installed into a derived image.
sudo rm -rf $H/ti && git clone -q --depth 1 -b gemma4-w4a16-moe https://github.com/xbill9/tpu-inference $H/ti
log "pr head $(git -C $H/ti log --oneline -1)"
git -C $H/ti apply $H/kvshare.diff && log "applied #3299"
TOKAMAX=$(grep '^tokamax==' $H/ti/requirements.txt)
sudo docker rm -f build >/dev/null 2>&1
# The image runs tpu_inference as an editable install from /workspace/tpu_inference: replace that tree.
sudo docker run --name build -v $H/ti:/src --entrypoint bash vllm/vllm-tpu:nightly -c   "pip install -q '$TOKAMAX' && rm -rf /workspace/tpu_inference/tpu_inference && cp -r /src/tpu_inference /workspace/tpu_inference/    && cd / && python3 -c 'import tpu_inference.layers.jax.quantization.wna16 as w,tokamax,vllm;print(w.__file__, hasattr(w.WNA16FusedMoEMethod,\"_expert_shape\"), vllm.__version__)'" > $L/build.log 2>&1
log "build: $(tail -1 $L/build.log)"
grep -q ' True ' $L/build.log || { log "PR code not active in image"; exit 1; }
sudo docker commit --change 'ENTRYPOINT ["/entrypoint.sh"]' --change 'WORKDIR /workspace/tpu_inference' build qat:pr3660 >/dev/null && sudo docker rm build >/dev/null

serve() {  # serve <model> [extra flags...]
  local model=$1; shift
  sudo docker rm -f vllm >/dev/null 2>&1
  sudo docker run -d --name vllm --privileged --net=host --shm-size 10gb -v /dev/shm:/dev/shm \
    -e HF_TOKEN="$(sudo cat /root/hf_token)" -e HF_HOME=/dev/shm/hf qat:pr3660 \
    vllm serve "$model" --served-model-name "$model" --host 127.0.0.1 --port 8000 \
      --tensor-parallel-size 1 --max-model-len 2048 --max-num-seqs 16 --max-logprobs 32 \
      --generation-config vllm --enable-prefix-caching "$@" >/dev/null
  local t0=$(date +%s)
  while :; do
    if curl -sf localhost:8000/v1/models | grep -q '"id"'; then echo "READY after $(( $(date +%s) - t0 ))s"; return 0; fi
    if [ -z "$(sudo docker ps -q -f name=vllm)" ] || [ $(( $(date +%s) - t0 )) -gt 1500 ]; then
      echo "FAILED after $(( $(date +%s) - t0 ))s: $(sudo docker logs vllm 2>&1 | grep -av tpu_info | grep -aoE '[A-Za-z_.]*(Error|Exception): .*|RESOURCE_EXHAUSTED.*' | grep -av 'See root cause above' | tail -1 | cut -c1-300)"
      return 1
    fi
    sleep 15
  done
}

for arm in "google/gemma-4-E2B-it-qat-w4a16-ct e2b" "google/gemma-4-E4B-it-qat-w4a16-ct e4b" \
           "google/gemma-4-12B-it-qat-w4a16-ct 12b override"; do
  set -- $arm; model=$1 tag=$2 flags=${3:-}
  extra=(--limit-mm-per-prompt '{"image":0,"audio":0,"video":0}')
  [ "$flags" = override ] && extra=(--hf-overrides '{"architectures":["Gemma4ForCausalLM"]}')
  log "serving $model"
  r=$(serve "$model" "${extra[@]}"); log "$tag $r"
  sudo docker logs vllm > $L/$tag.boot.log 2>&1
  log "$tag memory: $(grep -aoE 'total_hbm_used_gb=[0-9.]+GiB|TPU KV cache size: [0-9,]+ tokens' $L/$tag.boot.log | tail -2 | tr '\n' ' ')"
  log "$tag wna16 methods: $(grep -acE 'WNA16|wna16' $L/$tag.boot.log) log lines; pytorch fallback: $(grep -ac 'Falling back to vLLM-native' $L/$tag.boot.log)"
  if [ "${r%% *}" = READY ]; then
    python3 $H/w4a16_client.py record "$model" $L/$tag.record.json > $L/$tag.record.txt 2>&1
    log "$tag sample: $(python3 -c "import json;r=json.load(open('$L/$tag.record.json'))['rows'];print(' || '.join(x['text'].replace(chr(10),' ')[:90] for x in r[:3]))" 2>&1)"
    python3 $H/w4a16_client.py load "$model" $L/$tag.load.json > /dev/null 2>&1
    log "$tag load: $(python3 -c "import json;d=json.load(open('$L/$tag.load.json'));print(d['output_tok_per_s'],'tok/s, range',d['output_tok_per_s_min'],'to',d['output_tok_per_s_max'])" 2>&1)"
  fi
  sudo docker rm -f vllm >/dev/null 2>&1; sudo rm -rf /dev/shm/hf
done
log "DONE"
