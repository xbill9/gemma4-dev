# Bring Gemma-4-E2B Option B back up on inf2

Three ways back, fastest first. All three start from a stock Neuron DLAMI; nothing account-specific
survives from the original build box (see the last section).

## 1. Prebuilt image (minutes)

Use the rig's MCP tool, which picks the latest regional Neuron DLAMI, adds a 16 GB swapfile and
starts the container:

```
create_inf2_instance(subnet_id, security_group_id, iam_instance_profile, serving="optb")
```

It runs `OPTB_IMAGE`, default `xbill9/gemma4-optb:slim`. For the fastest build on the same
inf2.xlarge, set `OPTB_IMAGE=docker.io/xbill9/gemma4-optb:tp2-slim` (TP=2, 2048 context,
~61 tok/s). `DOCKER.md` lists every tag, the host each needs, and the `docker run` line for doing
it by hand.

## 2. Neffs from Hugging Face (no compile)

`xbill9/gemma-4-E2B-it-inferentia2` holds the compiled neffs and the servers. On a Neuron DLAMI
with the PyTorch 2.8 venv:

```bash
export PATH=/opt/aws_neuronx_venv_pytorch_2_8/bin:$PATH   # put it on PATH; `source activate` fails under SSM's /bin/sh
pip install transformers==5.13.0
pip install --extra-index-url https://pip.repos.neuron.amazonaws.com neuronx-distributed==0.17.26814   # TP servers only

sudo mkdir -p /workspace && sudo chown $USER /workspace && cd /workspace
hf download xbill9/gemma-4-E2B-it-inferentia2 --local-dir /workspace
hf download google/gemma-4-E2B-it --local-dir /workspace/real-gemma4-E2B-it   # gated: needs an HF token

# TP=2, 2048 context, both NeuronCores (reads /workspace/tpa_pre and /workspace/tpa_dec)
nohup python optb_server_tp_slim.py > server.log 2>&1 &     # inf2.xlarge (add swap first)
# or: nohup python optb_server_tp.py > server.log 2>&1 &    # inf2.8xlarge

# single core, 512 context
KV_MAX=512 KV_BUCKET=128 KV_PRE_OUT=/workspace/kv_pre_512.pt KV_DEC_OUT=/workspace/kv_dec_512.pt \
  nohup python optb_server.py > server.log 2>&1 &
```

The neffs load only with the versions they were compiled against: torch_neuronx 2.8.0.2.12,
neuronx-cc 2.23.6484.0, libneuronxla 2.2.15515.0. If the current DLAMI ships newer ones, pin those
three (the full pin list is in `Dockerfile`) or use path 1.

## 3. Compile from scratch

Same venv and `/workspace` layout as path 2, minus the neff download. Every script reads the
model from `/workspace/real-gemma4-E2B-it`.

```bash
# single core: prefill + decode graphs, written to KV_PRE_OUT / KV_DEC_OUT
KV_MAX=2048 KV_BUCKET=512 KV_PRE_OUT=/workspace/kv_pre_2048.pt KV_DEC_OUT=/workspace/kv_dec_2048.pt \
  python optb_kv.py trace

# TP=2 + KV aliasing: writes /workspace/tpa_pre and /workspace/tpa_dec
KV_MAX=2048 KV_BUCKET=512 python tp_alias_trace.py
```

A 256/64 single-core pair took about 18 minutes to compile; larger contexts take longer.
`optb_kv.py trace` ends by generating on device and on CPU and printing `SEQ_MATCH`, which must be
`True`; `optb_kv.py cpu` checks the prefill and decode logic in eager mode without compiling. `README.md` has the recipe and the gotchas behind it.

## What no longer exists

Deleted by 2026-10-05, all in account 106059658660. Do not follow older notes that name them.

- AMIs `ami-0c13e7feb3fe2e01e` (us-east-1) and `ami-0f07bf96d8551d36c` (us-east-2), and snapshot
  `snap-0149d318c5f36cfb9`
- S3 backup `s3://xbill-gemma4-patches-2b/optb-backup/` (the bucket is gone)
- ECR images in `gemma4-optb` (us-east-2); the repository remains, empty

The us-east-2 launch template `lt-0bca7ea41c9e213e5` (`gemma4-optb-lt-e2`) still exists but
launches the deleted us-east-2 AMI, so `fleet_e2.json`, which uses it, no longer starts anything.
