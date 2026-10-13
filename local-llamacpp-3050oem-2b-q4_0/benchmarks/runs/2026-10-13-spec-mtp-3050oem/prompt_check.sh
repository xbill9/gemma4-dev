#!/bin/bash
# Serving config (-c 131072): VRAM and prompt-phase speed with and without the drafter.
# ~1,000-token prompt, 4 requests, prompt cache off. Run from the rig root.
set -eu
set -a; . ./tpu.env; set +a
D=benchmarks/runs/2026-10-13-spec-mtp-3050oem
DRAFT=/home/xbill/models/gemma-4-E2B-it-qat-assistant/gemma-4-E2B-it-qat-assistant-q8_0.gguf
P=$(python3 -c "print('Summarize this text. ' + 'The quick brown fox jumps over the lazy dog near the old river bank. ' * 70)")
for arm in none mtp-n2; do
  extra=(); [ $arm = mtp-n2 ] && extra=(-md $DRAFT --spec-type draft-mtp -ngld all --spec-draft-n-max 2)
  $LLAMA_SERVER_BIN -m $MODEL_PATH --host 127.0.0.1 --port 8099 -ngl 99 -c 131072 -fa 1 -t 6 -tb 12 \
    --parallel 1 --reasoning off -ot "$OVERRIDE_TENSORS" "${extra[@]}" > $D/prompt-$arm.log 2>&1 &
  pid=$!; until curl -fs localhost:8099/health >/dev/null; do sleep 1; done
  echo "$arm vram $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader)"
  for i in 1 2 3 4; do
    curl -s localhost:8099/v1/chat/completions -H 'Content-Type: application/json' \
      -d "$(python3 -c "import json,sys;print(json.dumps({'messages':[{'role':'user','content':'Run $i. '+sys.argv[1]}],'max_tokens':64,'temperature':0,'cache_prompt':False}))" "$P")" \
      | python3 -c "import json,sys;t=json.load(sys.stdin)['timings'];print('$arm run $i prompt_n',t['prompt_n'],'prompt t/s',round(t['prompt_per_second'],1),'prompt ms',round(t['prompt_ms'],1),'decode',round(t['predicted_per_second'],1))"
  done
  kill $pid; wait $pid 2>/dev/null || true
done
