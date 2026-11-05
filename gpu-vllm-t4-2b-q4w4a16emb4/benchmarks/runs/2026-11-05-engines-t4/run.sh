#!/bin/bash
# One request at a time, every engine on the Tesla T4 that is built, every draft length 1-4 where the engine
# has a drafter. engines.py is the 3050 harness with this host's paths (see its docstring).
# Nothing else runs on the two vCPUs meanwhile (the mistral.rs build runs before or after, never during);
# its arms run later with ARMS=mistralrs,... once the binary exists.
set -u
cd "$(dirname "$0")"
ARMS=${ARMS:-llama,llama-d1,llama-d2,llama-d3,llama-d4,vllm,vllm-d1,vllm-d2,vllm-d3,vllm-d4,pytorch,litert-int248,litert-int248-d,sglang,sglang-d1,sglang-d2,sglang-d3,sglang-d4}
say() { echo "$(date +%F\ %T) $*" | tee -a run.log; }
[ -n "$(awk 'NR>1' /proc/swaps)" ] || sudo swapon /opt1/swapfile
{ nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv,noheader
  nproc; free -m | sed -n 2p
  PYTHONUSERBASE=/opt1/pyuser-engines /usr/bin/python3.13 -c "import importlib.metadata as m; print({p: m.version(p) for p in ('vllm','sglang','torch','transformers','triton','flashinfer-python','litert-lm','protobuf')})" 2>/dev/null
  /opt1/llama.cpp/build/bin/llama-server --version 2>&1 | head -1; } > host.txt
say "START $ARMS"
python3 engines.py . "$ARMS" > run.out 2>&1
say "END rc=$?"
python3 summarize.py . > SUMMARY.txt 2>&1
say DONE
