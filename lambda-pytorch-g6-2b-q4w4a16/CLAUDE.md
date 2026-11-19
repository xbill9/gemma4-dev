# CLAUDE.md — `lambda-pytorch-g6-2b-q4w4a16`

Gemma 4 E2B under the **hand-built PyTorch int4 engine** as an **AWS Lambda function on a GPU** (Lambda Managed
Instances, one NVIDIA L4 in a g6.2xlarge). Created 2026-11-19 for the Lambda serving-engine sweep, from
`gpu-pytorch-g6-2b-q4w4a16`: `image/engine/` holds its `e2b_engine.py`, `w4gemv.py`, `w4attn.py`,
`gemma4_tools.py` and `openai_server.py` byte-for-byte. **Keep them identical to that rig's.**

- **The image is AWS's Lambda PyTorch image** (torch 2.14.1, Python 3.13, CUDA 13.0.3); the G6 rig ran the DLAMI's
  torch 2.13. The engine's pins are installed with torch held at the image's version.
- **Both checkpoints are in the hub cache at `/opt/hf`**, read with `HF_HUB_OFFLINE=1`: the engine loads them by
  repo name (`snapshot_download` without a revision), so `image/fetch.py cache` writes `refs/main` at the pinned
  revision. Everything writable goes to `/tmp` (the image is read-only on Lambda).
- **`image/engine_ext.py`** runs `ENGINE_CMD` during Init; `image/handler.py` proxies each invocation.
- One request at a time (the engine does not batch): `PER_ENV_MAX_CONCURRENCY=4` lets Lambda queue a few.

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
