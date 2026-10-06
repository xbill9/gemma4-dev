# gpu-vllm-mi300x-31b

One **AMD Instinct MI300X** serving `google/gemma-4-31B-it` through
**vLLM** in Docker, on a DigitalOcean GPU droplet reached through AMD Developer Cloud.

Read the name per `../NAMING.md`: platform `gpu`, runtime `vllm`, hardware `mi300x`, model `31b`, and **no fifth slot**: these are the reference bf16 weights.

| | |
| --- | --- |
| GPU | AMD Instinct MI300X VF, `gfx942`, 191.69 GiB (measured on `../gpu-vllm-mi300x-2b`) |
| Image | `vllm/vllm-openai-rocm:nightly-rocm100` — run `check_image` first |
| Model | `google/gemma-4-31B-it` — 62.55 GB bf16, Apache-2.0, ungated; served text-only (`{"image": 0, "audio": 0}`) |
| Kernel expected | none — bf16 needs no mixed-precision kernel |
| Context | 32768 of 262144, matching its twin and the E2B rigs |
| Cost | $1.99/hr, **billed while powered off too** |

**Status 2026-10-02: forked, not provisioned. Nothing measured.** No 31B checkpoint has been served on this card in this monorepo.

## Why it exists

It is the bf16 control for `../gpu-vllm-mi300x-31b-q4w4a16`. It is served text-only so it matches the `-text` repack; the vision tower's weights are still resident, a known residual difference. The study is `../gpu-vllm-mi300x-31b-q4w4a16/PREREGISTRATION.md`. On a TPU v5e the repacks
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
  → serving_status → verify_capabilities
```

`tpu.env` is the committed source of truth. It is named `tpu.env` like every sibling; the name is a
monorepo convention, not a claim that this rig is a TPU.

## What is different from `../gpu-vllm-mi300x-2b`

- `LIMIT_MM_PER_PROMPT` is `{"image": 0, "audio": 0}`: the checkpoint is multimodal but served
  text-only, to match the `-text` repack, and `verify_capabilities` skips vision.
- `EXPECT_MP_KERNEL` is empty, so `check_image` asks only the two bf16 questions.
- `benchmarking_suite.py` takes the report's `weights_dtype`, `quantization` and `parameters_b`
  from `tpu.env`, so the report says 31B rather than the E2B default.

Create and destroy are deliberately not tools. Both are dollar-per-hour decisions that stay a
human step in the DigitalOcean console.
