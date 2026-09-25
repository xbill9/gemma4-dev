#!/bin/bash
# probe_stages.sh <run> <patches> <stage>... -- on a v6e-1 VM with Docker and /root/hf_token.
#
#   <patches>  comma-separated diff files (paths under $W) applied in order to the pinned
#              image's tpu_inference; the result is committed as jev-probe:patched.
#   <stage>    [base:]<model>=<tag>[@<ref-tag>]
#              serve <model> (on the unpatched image when prefixed base:), record greedy outputs
#              as <tag>.json, compare them with <ref-tag>.json when given, then measure throughput.
#
# Stages run in order and a failed stage does not stop the rest. Logs go to
# gs://$BUCKET/jev-tpu-31b/<run>/ after every stage. When $W/tpu/w4a16_matmul_bench.py is
# present it runs after the unit tests. The VM is left running; delete it after.
set -u
RUN=$1; PATCHES=$2; shift 2
BASE=vllm/vllm-tpu@sha256:19a1a0526476f902eb83e1057f3d8938f35b457dd7716a30e9ab4f7bee90d507
PATCHED=jev-probe:patched
DST="gs://aisprint-491218-bucket/jev-tpu-31b/$RUN"
W=/opt/w4a16; L=/opt/probe-logs/$RUN; mkdir -p $L
export LOGS=$L
# Reference outputs from earlier runs, compared against by <tag>@<ref-tag>.
[ -d $W/refs ] && cp $W/refs/*.json $L/
CLIENT="python3 $W/tpu/w4a16_client.py"
log() { echo "[probe $(date -u +%FT%TZ)] $*" | tee -a $L/run.log > /dev/console; }
sync_up() { gcloud storage rsync -r -q $L "$DST" >/dev/null 2>&1; }

log "start $RUN: patches $PATCHES; stages $*"
docker pull -q "$BASE" >> $L/run.log 2>&1
SITE=$(docker run --rm --entrypoint python3 "$BASE" -c "import os,tpu_inference;print(os.path.dirname(os.path.dirname(tpu_inference.__file__)))" 2>/dev/null | tail -1)
docker rm -f patch >/dev/null 2>&1
# Created, never started: docker cp works on it, and commit keeps the base entrypoint/cmd.
docker create --name patch "$BASE" >/dev/null
rm -rf /opt/ti && mkdir -p /opt/ti && docker cp "patch:$SITE/tpu_inference" /opt/ti/
for p in ${PATCHES//,/ }; do
  if ! (cd /opt/ti && patch -p1 --dry-run < $W/$p > $L/patch-$p.log 2>&1 && patch -p1 < $W/$p >> $L/patch-$p.log 2>&1); then
    log "patch $p did not apply: $(grep -iE 'fail|reject' $L/patch-$p.log | head -3 | tr '\n' ' ')"; sync_up; exit 1
  fi
  log "patch $p applied: $(grep -c '^patching file' $L/patch-$p.log) files, $(grep -ciE 'offset|fuzz' $L/patch-$p.log) hunks with offset or fuzz"
done
docker cp /opt/ti/tpu_inference/. "patch:$SITE/tpu_inference/"
docker commit patch "$PATCHED" >/dev/null && docker rm -f patch >/dev/null

# Unit tests for every patched area, on the chip. A server left running holds the TPU.
docker rm -f vllm >/dev/null 2>&1
docker run --rm --privileged --net=host -v $W/tests:/tests --entrypoint bash "$PATCHED" -c \
  "pip install -q pytest >/dev/null 2>&1; cd /tests && python3 -m pytest -q -p no:cacheprovider ." > $L/pytest.log 2>&1
log "pytest: $(tail -1 $L/pytest.log)"
if [ -f $W/tpu/w4a16_matmul_bench.py ]; then
  docker run --rm --privileged --net=host -v $W:/w --entrypoint python3 "$PATCHED" /w/tpu/w4a16_matmul_bench.py > $L/matmul-bench.txt 2>&1
  log "matmul bench: $(tail -1 $L/matmul-bench.txt)"
fi
sync_up

for stage in "$@"; do
  image=$PATCHED; spec=$stage
  case $spec in base:*) image=$BASE; spec=${spec#base:};; esac
  model=${spec%%=*}; rest=${spec#*=}; tag=${rest%%@*}; ref=""
  [ "$rest" != "$tag" ] && ref=${rest#*@}
  log "serving $model as $tag on $([ "$image" = "$BASE" ] && echo base || echo patched) image"
  if IMAGE=$image bash $W/tpu/serve.sh "$model" > $L/$tag.serve.txt 2>&1; then
    log "$(tail -1 $L/$tag.serve.txt)"
    grep -E "total_hbm|hbm_limit|KV cache|kv_cache|GiB|model weights|num_blocks" "$L/$(echo "$model" | tr '/' '_').boot.log" | cut -c1-300 > $L/$tag.memory.txt
    $CLIENT record "$model" $L/$tag.json > $L/$tag.txt 2>&1
    if [ -n "$ref" ] && [ -f $L/$ref.json ]; then
      $CLIENT compare $L/$ref.json $L/$tag.json > $L/$tag.vs.$ref.txt 2>&1
      log "$tag vs $ref: $(tail -1 $L/$tag.vs.$ref.txt)"
    fi
    $CLIENT load "$model" $L/$tag.load.json > /dev/null 2>&1
    log "$tag load: $(tr -d '\n ' < $L/$tag.load.json 2>/dev/null)"
    mv "$L/$(echo "$model" | tr '/' '_').boot.log" $L/$tag.boot.log
  else
    log "$(tail -1 $L/$tag.serve.txt)"
    mv "$L/$(echo "$model" | tr '/' '_').boot.log" $L/$tag.boot.log 2>/dev/null
  fi
  docker rm -f vllm >/dev/null 2>&1; rm -rf /dev/shm/hf; sync_up
done
log "DONE $RUN"
sync_up
