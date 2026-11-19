# CLAUDE.md — `lambda-jax-g6-2b`

Gemma 4 E2B under **pure JAX** (this tree's own Gemma 4 port) as an **AWS Lambda function on a GPU** (Lambda Managed
Instances, one NVIDIA L4 in a g6.2xlarge). Created 2026-11-19 for the Lambda serving-engine sweep, from
`gpu-jax-g6-2b`: `image/engine/` holds its `jax_openai_server.py`, `jax_engine.py` and `ports/gemma4/` loader and
model byte-for-byte, with its serving flags (bf16 reference checkpoint `google/gemma-4-E2B-it`, `--quant-mode fp16`,
`--ple-bits 4 --int8-lm-head`, 4,096 tokens).

- **AWS has no JAX Lambda image**: AWS's Lambda base image plus `jax[cuda13]==0.11.1` (the version the G6 first
  serve recorded), which brings its own CUDA libraries from pip.
- **The checkpoint is in the hub cache at `/opt/hf`** (`google/gemma-4-E2B-it` @ `4adf1574`, main on 2026-11-19),
  read with `HF_HUB_OFFLINE=1`. XLA's compilation cache goes to `/tmp/jax-cache`, so every new environment compiles.
- **24 GiB of host memory** (6 vCPUs): the loader reads the 10 GB checkpoint through host memory.
- One request at a time (`Gemma4EModelJAX` refuses B > 1).

## Shared with the other lambda-*-g6 rigs

`lmi.py`, `builder.py`, `chat_ui.py`, `bench.py`, `republish.py`, `tests/test_lmi.py` and the `Makefile` are the same
in each `lambda-*-g6-*` rig except `lambda-vllm-g6-2b-q4w4a16emb4`, which they were generalized from; `ENGINE` in
`tpu.env` picks `vllm`, `sglang` or `cmd`. A fix belongs in every copy.

- **Images are built on EC2, not here** (`make builder`, then `make image`): one c7i.4xlarge named
  `lambda-image-builder`, found by tag, shared by every rig. The rig's `image/` goes to S3 as a tarball; the
  builder runs buildx and pushes to ECR. `make builder-down` terminates it.
- **One capacity provider for every rig** (`gemma4-e2b-g6`, MaxVCpuCount 16 = two g6.2xlarge, the account's
  G-instance quota). Two rigs can be up at once; `make down` on one before `make up` on a third.
- **A published version keeps its minimum running and billed.** `make down` releases the instance and keeps the
  function; `make destroy` deletes the function. Republish with `republish.py` while an environment runs
  (see `lambda-vllm-g6-2b-q4w4a16emb4/CLAUDE.md`).
- **No streaming**: the handler returns the whole completion, so time to first token is not measurable through Lambda.

## Chat page

`make ui` serves `chat_ui.py` on http://localhost:UI_PORT (`tpu.env`); each Send is one synchronous `Invoke` of
`$LATEST.PUBLISHED` with this machine's AWS credentials. Ports: vLLM 8768, SGLang 8769, llama.cpp 8770,
PyTorch 8771, JAX 8772, so all five pages can run at once.

## Commands

`make builder | image | deploy | status | ui | ask Q='…' | bench OUT=… | up | down | destroy | builder-down | test`.
