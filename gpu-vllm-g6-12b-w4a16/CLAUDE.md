# CLAUDE.md — `gpu-vllm-g6-12b-w4a16`

Serving rig: **`google/gemma-4-12B-it-qat-w4a16-ct`** (12B) under **vLLM** on **AWS EC2 G6** — an x86_64 host
with one **NVIDIA L4** (Ada, SM 8.9).

> **NOTHING HAS BEEN MEASURED ON THIS RIG.** Forked from [`gpu-vllm-g6-2b`](../gpu-vllm-g6-2b) on 2026-09-29.
> `benchmarks/runs/` and `benchmarks/reports/` are empty on purpose; the parent's
> 2026-08-30 E2B run was not copied. Every G6/Ada/vLLM fact in the parent's `CLAUDE.md` is a
> fact about **E2B bf16 on vLLM 0.28.0**; read it there, and do not quote it as this rig's.

## Why this rig exists

It is the **raw EC2 counterpart of a SageMaker run** in
[sagemaker-gemma](https://github.com/xbill9/sagemaker-gemma) (`docs/runs/2026-09-28-12b-qat-l4`): the same checkpoint on the same GPU, vLLM
0.30.0 in both, so the difference measured is SageMaker's managed endpoint against a plain
`g6` instance. SageMaker charges 1.40× the EC2 price for `g6.xlarge` in us-east-2
($1.1267/h against $0.8048/h, AWS Pricing API, 2026-09-29).

Google's 12B QAT. SageMaker `gemma-4-12b-qat-l4` (2026-09-28): 8.28 GiB, 29.3 tok/s decode, 358.05 at 16 parallel. Also serves the repack `xbill9/gemma-4-12B-it-qat-q4_0-w4a16-ct` by MODEL_NAME override. 12B bf16 (22.4 GiB) does not fit one L4, hence no bare `-12b` rig.

## Settings

The env file decides; the name describes (`../NAMING.md`). `tests/test_server.py` asserts
`tpu.env` and `server.py` agree on every value below.

| Setting | Value |
| --- | --- |
| Model (`MODEL_NAME`) | `google/gemma-4-12B-it-qat-w4a16-ct` |
| Encoding (slot 5) | `w4a16` |
| Runtime | `vllm/vllm-openai:v0.30.0` (the vLLM release the SageMaker runs used) |
| Instance | `g6.xlarge` (1× NVIDIA L4) |
| Region | `us-east-2` (where the SageMaker runs placed) |
| `MAX_MODEL_LEN` | 8192 |
| `GPU_MEMORY_UTILIZATION` | 0.90 |
| `MAX_NUM_SEQS` | unset — vLLM's default, as on SageMaker |
| `EXTRA_VLLM_ARGS` | empty |

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
