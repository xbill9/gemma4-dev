# CLAUDE.md — `lambda-llamacpp-g6-2b-q4_0`

Gemma 4 E2B under **llama.cpp** as an **AWS Lambda function on a GPU** (Lambda Managed Instances, one NVIDIA L4 in a
g6.2xlarge). Created 2026-11-19 for the Lambda serving-engine sweep, from `gpu-llamacpp-g6-2b-q4_0` and the G6
flight's `llama-d2` arm: the exact Q4_0 GGUF (SHA-256 `419db9a6…`), Google's QAT assistant drafter as exact Q4_0
at 2 drafted tokens, llama.cpp at `7182ef71e`.

- **AWS has no llama.cpp Lambda image.** The Dockerfile builds `llama-server` on AWS's Lambda base image (CUDA
  13.0.3, AL2023) with NVIDIA's CUDA 13.0 toolkit packages, for sm_89, statically linked against ggml, without
  `-march=native` (the builder is Intel, the g6 is AMD EPYC).
- **`image/engine_ext.py` is the engine extension** (AWS's vLLM/SGLang images carry their own for those two
  engines only): it runs `ENGINE_CMD` from `tpu.env` during Init and waits for `/health`; `image/handler.py`
  proxies each invocation to `/v1/chat/completions`.
- **4 slots of 8,192 tokens** (`-c 32768 --parallel 4`), so `PER_ENV_MAX_CONCURRENCY=4`. The flight ran
  `-c 8192` for its short prompts. `--jinja` so `chat_template_kwargs` reach the template.
- 4 vCPUs (16 GiB at 4 GiB per vCPU), so `-t 4 -tb 4`; the G6 rig ran `-t 4 -tb 8` on 8.

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
