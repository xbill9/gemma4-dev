# gpu-vllm-g6-31b-w4a16

Serve **`google/gemma-4-31B-it-qat-w4a16-ct`** (31B) with **vLLM 0.30.0** on **AWS EC2 G6** — an x86_64 host with
one **NVIDIA L4** (Ada, SM 8.9) — through a single-file MCP server that provisions the
instance with boto3.

> **Nothing measured yet.** Forked from [`gpu-vllm-g6-2b`](../gpu-vllm-g6-2b) on 2026-09-29. See
> [`CLAUDE.md`](CLAUDE.md).

## Why

The raw EC2 counterpart of the SageMaker run in [sagemaker-gemma](https://github.com/xbill9/sagemaker-gemma)
(`docs/runs/2026-09-28-l4-26b-31b`), so SageMaker's managed endpoint can be compared with a plain `g6` on the
same GPU and vLLM version. Google's 31B QAT. It does not start on one L4 at standard settings (19.77 GiB of weights exceed the 0.9 cap; start-up OOM). SageMaker `gemma-4-31b-qat-l4-2xl` (2026-09-28) served it with the settings in tpu.env on a 32 GiB host: 18.7 GiB, 12.6 tok/s decode, 39.45 at 16 parallel (capped by 4 running requests), 40/40.

## Settings

| Setting | Value |
| --- | --- |
| Model (`MODEL_NAME`) | `google/gemma-4-31B-it-qat-w4a16-ct` |
| Encoding (slot 5) | `w4a16` |
| Runtime | `vllm/vllm-openai:v0.30.0` (the vLLM release the SageMaker runs used) |
| Instance | `g6.2xlarge` (1× NVIDIA L4) |
| Region | `us-east-2` (where the SageMaker runs placed) |
| `MAX_MODEL_LEN` | 1024 |
| `GPU_MEMORY_UTILIZATION` | 0.97 |
| `MAX_NUM_SEQS` | 4 |
| `EXTRA_VLLM_ARGS` | `--max-num-batched-tokens 1024 --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}'` |

Authoritative values live in [`tpu.env`](tpu.env).

## Operation

The tools, quick start and teardown are the parent's unchanged:
[`../gpu-vllm-g6-2b/README.md`](../gpu-vllm-g6-2b/README.md). Run `./set_env.sh` from this directory so this
rig's `tpu.env` is the one exported.
