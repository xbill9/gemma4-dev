#!/bin/bash
# w4a16_probe.sh -- on the v6e-1 VM: patch the JAX-path W4A16 method into the pinned vLLM TPU
# image, run its unit tests on the chip, then serve 12B bf16 (reference), 12B W4A16 and 31B
# W4A16 in turn with the jev-tpu flags. Logs go to gs://$BUCKET/jev-tpu-31b/<run>/ as each
# stage finishes. The VM deletes itself at the end unless metadata jev-keep=1 and 31B serves.
set -u
md() { curl -s -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }
RUN=$(md attributes/jev-run); KEEP=$(md attributes/jev-keep)
ZONE=$(md zone | awk -F/ '{print $NF}')
BASE=vllm/vllm-tpu@sha256:19a1a0526476f902eb83e1057f3d8938f35b457dd7716a30e9ab4f7bee90d507
IMG=jev-w4a16:patched
DST="gs://aisprint-491218-bucket/jev-tpu-31b/$RUN"
W=/opt/w4a16; L=/opt/w4a16-logs; mkdir -p $L
export IMAGE=$IMG LOGS=$L
log() { echo "[w4a16 $(date -u +%FT%TZ)] $*" | tee -a $L/run.log > /dev/console; }
sync_up() { gcloud storage rsync -r -q $L "$DST" >/dev/null 2>&1; }
finish() {
  log "DONE: $1"; sync_up
  # jev-keep=1 keeps the VM only while 31B is serving; any failure still deletes it.
  if [ "$KEEP" = 1 ] && [ "$1" = "31B serving on :8000" ]; then log "jev-keep=1: leaving the VM up"; exit 0; fi
  gcloud compute instances delete "$(hostname)" --zone "$ZONE" --quiet
  exit 0
}
CLIENT="python3 $W/tpu/w4a16_client.py"

log "start $RUN on $ZONE"
docker pull -q "$BASE" >> $L/run.log 2>&1 || finish "image pull failed"
log "image $(docker inspect --format '{{index .RepoDigests 0}}' "$BASE")"

# Patch the installed tpu_inference on the host, then commit it into a derived image.
command -v patch >/dev/null || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq patch >/dev/null; }
SITE=$(docker run --rm --entrypoint python3 "$BASE" -c "import os,tpu_inference;print(os.path.dirname(os.path.dirname(tpu_inference.__file__)))" 2>/dev/null | tail -1)
log "tpu_inference installed under $SITE"
docker rm -f patch >/dev/null 2>&1
# Created, never started: docker cp works on it, and commit keeps the base entrypoint/cmd.
docker create --name patch "$BASE" >/dev/null
rm -rf /opt/ti && mkdir -p /opt/ti && docker cp "patch:$SITE/tpu_inference" /opt/ti/
if ! (cd /opt/ti && patch -p1 --dry-run < $W/wna16.diff > $L/patch.log 2>&1 && patch -p1 < $W/wna16.diff >> $L/patch.log 2>&1); then
  log "patch did not apply to this image's tpu_inference: $(grep -iE 'fail|reject' $L/patch.log | head -3 | tr '\n' ' ')"
  finish "patch failed"
fi
log "patch applied: $(grep -c '^patching file' $L/patch.log) files, $(grep -ciE 'offset|fuzz' $L/patch.log) hunks with offset or fuzz"
docker cp /opt/ti/tpu_inference/. "patch:$SITE/tpu_inference/"
docker commit patch "$IMG" >/dev/null && docker rm -f patch >/dev/null
sync_up

# Unit tests on the chip: the new ones, plus the existing compressed-tensors dispatch tests.
docker run --rm --privileged --net=host -v $W/tests:/tests --entrypoint bash "$IMG" -c \
  "pip install -q pytest >/dev/null 2>&1; cd /tests && python3 -m pytest -q -p no:cacheprovider test_wna16.py test_compressed_tensors.py" \
  > $L/pytest.log 2>&1
log "pytest: $(tail -1 $L/pytest.log)"
sync_up

serve() {  # serve <model> <tag>; 0 when serving
  log "serving $1"
  if bash $W/tpu/serve.sh "$1" > $L/$2.serve.txt 2>&1; then
    log "$(tail -1 $L/$2.serve.txt)"
    grep -E "total_hbm|hbm_limit|KV cache|kv_cache|GiB|model weights|num_blocks" "$L/$(echo "$1" | tr '/' '_').boot.log" | cut -c1-300 > $L/$2.memory.txt
    return 0
  fi
  log "$(tail -1 $L/$2.serve.txt)"; return 1
}
stop() { docker rm -f vllm >/dev/null 2>&1; rm -rf /dev/shm/hf; sync_up; }

# 12B bf16: the reference the 12B W4A16 outputs are compared against.
if serve google/gemma-4-12B-it 12b-bf16; then
  $CLIENT record google/gemma-4-12B-it $L/12b-bf16.json > $L/12b-bf16.txt 2>&1
fi
stop

# 12B W4A16: the first model through the new path.
if serve google/gemma-4-12B-it-qat-w4a16-ct 12b-w4a16; then
  $CLIENT record google/gemma-4-12B-it-qat-w4a16-ct $L/12b-w4a16.json > $L/12b-w4a16.txt 2>&1
  [ -f $L/12b-bf16.json ] && $CLIENT compare $L/12b-bf16.json $L/12b-w4a16.json > $L/12b-compare.txt 2>&1
  log "12B compare: $(tail -1 $L/12b-compare.txt 2>/dev/null)"
  $CLIENT load google/gemma-4-12B-it-qat-w4a16-ct $L/12b-w4a16.load.json > /dev/null 2>&1
  log "12B W4A16 load: $(cat $L/12b-w4a16.load.json 2>/dev/null | tr -d '\n ')"
fi
stop

# 31B W4A16 on one chip: the goal.
if serve google/gemma-4-31B-it-qat-w4a16-ct 31b-w4a16; then
  $CLIENT record google/gemma-4-31B-it-qat-w4a16-ct $L/31b-w4a16.json > $L/31b-w4a16.txt 2>&1
  $CLIENT load google/gemma-4-31B-it-qat-w4a16-ct $L/31b-w4a16.load.json > /dev/null 2>&1
  log "31B W4A16 load: $(cat $L/31b-w4a16.load.json 2>/dev/null | tr -d '\n ')"
  docker logs vllm > $L/31b-w4a16.full.log 2>&1
  sync_up
  [ "$KEEP" = 1 ] && finish "31B serving on :8000"
  finish "31B served"
fi
finish "31B did not serve"
