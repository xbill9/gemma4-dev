# gpu-vllm-g6-4b-w4a16

Serve **`google/gemma-4-E4B-it-qat-w4a16-ct`** (E4B) with **vLLM 0.30.0** on **AWS EC2 G6** — an x86_64 host with
one **NVIDIA L4** (Ada, SM 8.9) — through a single-file MCP server that provisions the
instance with boto3.

> **Nothing measured yet.** Forked from [`gpu-vllm-g6-2b`](../gpu-vllm-g6-2b) on 2026-09-29. See
> [`CLAUDE.md`](CLAUDE.md).

## Why

The raw EC2 counterpart of the SageMaker run in [sagemaker-gemma](https://github.com/xbill9/sagemaker-gemma)
(`docs/runs/2026-09-27-e4b-qat-vs-bf16`), so SageMaker's managed endpoint can be compared with a plain `g6` on the
same GPU and vLLM version. Google's E4B QAT. SageMaker `gemma-4-e4b-qat` (2026-09-27): 11.04 GiB, 60.7 tok/s decode, 599.5 at 16 parallel. Also serves the repack `xbill9/gemma-4-E4B-it-qat-q4_0-w4a16-ct` by MODEL_NAME override.

## Settings

| Setting | Value |
| --- | --- |
| Model (`MODEL_NAME`) | `google/gemma-4-E4B-it-qat-w4a16-ct` |
| Encoding (slot 5) | `w4a16` |
| Runtime | `vllm/vllm-openai:v0.30.0` (the vLLM release the SageMaker runs used) |
| Instance | `g6.xlarge` (1× NVIDIA L4) |
| Region | `us-east-2` (where the SageMaker runs placed) |
| `MAX_MODEL_LEN` | 8192 |
| `GPU_MEMORY_UTILIZATION` | 0.90 |
| `MAX_NUM_SEQS` | unset — vLLM's default, as on SageMaker |
| `EXTRA_VLLM_ARGS` | empty |

Authoritative values live in [`tpu.env`](tpu.env).

## Operation

The tools, quick start and teardown are the parent's unchanged:
[`../gpu-vllm-g6-2b/README.md`](../gpu-vllm-g6-2b/README.md). Run `./set_env.sh` from this directory so this
rig's `tpu.env` is the one exported.
