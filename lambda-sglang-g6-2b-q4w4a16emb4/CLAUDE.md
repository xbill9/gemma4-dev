# CLAUDE.md — `lambda-sglang-g6-2b-q4w4a16emb4`

Gemma 4 E2B under **SGLang 0.5.23** as an **AWS Lambda function on a GPU** (Lambda Managed Instances, one NVIDIA L4
in a g6.2xlarge). Created 2026-11-19 for the Lambda serving-engine sweep. Slot 1 is the only slot that differs from
`gpu-sglang-g6-2b-q4w4a16emb4`: same checkpoint (`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4` @ `db715f83`),
same SGLang version, same two patches (`image/patches/`), same serving values.

- **The image is AWS's Lambda SGLang image** (`public.ecr.aws/lambda/deep-learning-containers/sglang`, pinned by
  digest): SGLang 0.5.23+dlc1, Python 3.13, CUDA 13.0.3. Its engine extension starts `sglang.launch_server` during
  Init with `--mem-fraction-static`, `--attention-backend` and `--tp-size` from `SGLANG_*` variables, plus
  `SGLANG_SERVER_ARGS`; its handler proxies each invocation. The Dockerfile applies the patches and fails if
  either no longer applies.
- **`ATTENTION_BACKEND=triton`**: AWS's extension always passes `--attention-backend` (flashinfer by default);
  the G6 server chose triton for Gemma 4 on its own.
- No drafter (`SPECULATIVE_MODEL` empty), as in the G6 rig's `tpu.env`.

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
