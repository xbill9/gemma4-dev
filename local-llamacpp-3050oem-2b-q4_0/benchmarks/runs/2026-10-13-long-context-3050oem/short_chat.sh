#!/bin/bash
# Does reserving a long context slow a short chat? Same request, 5 times, against a
# server at each -c, generating exactly 256 tokens (ignore_eos). Reports llama-server's
# own decode and prompt timings per request.
set -eu
cd "$(dirname "$0")"
BIN=/home/xbill/llama.cpp/build/bin
M=/home/xbill/models/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf
for c in 8192 131072; do
  until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" -le 40 ]; do sleep 5; done
  $BIN/llama-server -m $M --host 127.0.0.1 --port 8099 -ngl 99 -c $c -ctk f16 -ctv f16 -fa 1 -t 6 -tb 12 \
    --parallel 1 --reasoning off -ot 'per_layer_token_embd\.weight=CUDA0' > short-c$c.log 2>&1 &
  pid=$!
  until curl -fs localhost:8099/health >/dev/null; do sleep 1; done
  for i in 1 2 3 4 5 6; do
    curl -s localhost:8099/v1/chat/completions -H 'Content-Type: application/json' \
      -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Run $i. Write a long story about a lighthouse keeper.\"}],\"max_tokens\":256,\"ignore_eos\":true,\"cache_prompt\":false}" \
      | python3 -c "import json,sys;t=json.load(sys.stdin)['timings'];print('c=$c run=$i decode',round(t['predicted_per_second'],2),'n',t['predicted_n'],'prompt',round(t['prompt_per_second'],1))"
  done
  kill $pid; wait $pid 2>/dev/null || true
done
