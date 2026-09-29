# AGENTS.md — `gpu-vllm-g6-31b-w4a16`

Serving rig: **`google/gemma-4-31B-it-qat-w4a16-ct`** (31B) under **vLLM** on **AWS EC2 G6** — an x86_64 host
with one **NVIDIA L4** (Ada, SM 8.9).

> **NOTHING HAS BEEN MEASURED ON THIS RIG.** Forked from [`gpu-vllm-g6-2b`](../gpu-vllm-g6-2b) on 2026-09-29.
> `benchmarks/runs/` and `benchmarks/reports/` are empty on purpose; the parent's
> 2026-08-30 E2B run was not copied. Every G6/Ada/vLLM fact in the parent's `CLAUDE.md` is a
> fact about **E2B bf16 on vLLM 0.28.0**; read it there, and do not quote it as this rig's.

## Why this rig exists

It is the **raw EC2 counterpart of a SageMaker run** in
[sagemaker-gemma](https://github.com/xbill9/sagemaker-gemma) (`docs/runs/2026-09-28-l4-26b-31b`): the same checkpoint on the same GPU, vLLM
0.30.0 in both, so the difference measured is SageMaker's managed endpoint against a plain
`g6` instance. SageMaker charges 1.40× the EC2 price for `g6.xlarge` in us-east-2
($1.1267/h against $0.8048/h, AWS Pricing API, 2026-09-29).

Google's 31B QAT. It does not start on one L4 at standard settings (19.77 GiB of weights exceed the 0.9 cap; start-up OOM). SageMaker `gemma-4-31b-qat-l4-2xl` (2026-09-28) served it with the settings in tpu.env on a 32 GiB host: 18.7 GiB, 12.6 tok/s decode, 39.45 at 16 parallel (capped by 4 running requests), 40/40.

## Settings

The env file decides; the name describes (`../NAMING.md`). `tests/test_server.py` asserts
`tpu.env` and `server.py` agree on every value below.

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

What changed in `server.py` at the fork, and only this:

- `MAX_NUM_SEQS` empty omits `--max-num-seqs`, so vLLM uses its own default. The parent pinned
  8, which would cap a 16-parallel run and break the comparison with SageMaker.
- `EXTRA_VLLM_ARGS` is split like a shell line and each argument re-quoted into `start.sh`, so a
  JSON value such as `--limit-mm-per-prompt '{"image":0}'` survives `tpu.env` → `start.sh`.
- Defaults for `MODEL_NAME`, `INSTANCE_TYPE`, `AWS_REGION`, `MAX_MODEL_LEN`,
  `GPU_MEMORY_UTILIZATION` and `VLLM_IMAGE` follow `tpu.env`.

## Not yet verified on this rig

- That the upstream `vllm/vllm-openai:v0.30.0` image serves this checkpoint as the SageMaker
  container (same vLLM version, AWS build) did.
- That the DLAMI and root volume inherited from the parent hold this checkpoint's download.

**`CLAUDE.md` is authoritative where this file disagrees with it.**
