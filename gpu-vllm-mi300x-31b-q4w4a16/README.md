# gpu-vllm-mi300x-31b-q4w4a16

One **AMD Instinct MI300X** serving `xbill9/gemma-4-31B-it-qat-q4_0-w4a16-ct-text` through
**vLLM** in Docker, on a DigitalOcean GPU droplet reached through AMD Developer Cloud.

Read the name per `../NAMING.md`: platform `gpu`, runtime `vllm`, hardware `mi300x`, model `31b`, encoding **`q4w4a16`**: Google's quantization-aware-trained (QAT) 31B weights repacked into compressed-tensors W4A16 with every group of 32 kept on its trained Q4_0 grid. Google's own `-qat-w4a16-ct` export re-rounds those groups and would be `w4a16`.

| | |
| --- | --- |
| GPU | AMD Instinct MI300X VF, `gfx942`, 191.69 GiB (measured on `../gpu-vllm-mi300x-2b`) |
| Image | `vllm/vllm-openai-rocm:nightly-rocm100` — run `check_image` first |
| Model | `xbill9/gemma-4-31B-it-qat-q4_0-w4a16-ct-text` — 19.3 GB, text only, Apache-2.0, ungated |
| Kernel expected | `TritonW4A16LinearKernel` (the only ROCm W4A16 path that fits gfx942 + bf16 + group 32) |
| Context | 32768 of 262144, matching its twin and the E2B rigs |
| Cost | $1.99/hr, **billed while powered off too** |

**Status 2026-10-02: forked, not provisioned. Nothing measured.** No 31B checkpoint has been served on this card in this monorepo.

## Why it exists

It is the 4-bit half of the headline A/B pair, against `../gpu-vllm-mi300x-31b` (bf16 31B). **This directory holds the study's preregistration**, `PREREGISTRATION.md`, because the 31B pair is what the study is for. On a TPU v5e the repacks
mattered because they made big models fit. On 192 GB both builds fit, so the question is speed:
31B decode at low concurrency streams every weight byte per token, which is where 4-bit should win
if it wins anywhere on this card.

## Quick start

```bash
./init.sh                       # deps, skill snapshots, MCP registration
echo 'DIGITALOCEAN_ACCESS_TOKEN=...' > .env && chmod 600 .env
```

Then, through the agent:

```
list_droplets → gpu_status → check_image → download_weights → deploy_vllm
  → serving_status → server_logs(grep="Using") → verify_capabilities
```

`tpu.env` is the committed source of truth. It is named `tpu.env` like every sibling; the name is a
monorepo convention, not a claim that this rig is a TPU.

## What is different from `../gpu-vllm-mi300x-2b`

- `check_image` also reads the image's ROCm mixed-precision kernel list and fails when
  `EXPECT_MP_KERNEL` is missing.
- `LIMIT_MM_PER_PROMPT` is empty: the `-text` checkpoint has no towers, so the flag is dropped and
  `verify_capabilities` skips vision.
- `benchmarking_suite.py` takes the report's `weights_dtype`, `quantization` and `parameters_b`
  from `tpu.env`, so a 4-bit run cannot be filed as bf16.

Create and destroy are deliberately not tools. Both are dollar-per-hour decisions that stay a
human step in the DigitalOcean console.
