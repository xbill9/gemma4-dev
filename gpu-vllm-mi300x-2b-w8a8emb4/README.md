# gpu-vllm-mi300x-2b-w8a8emb4

One **AMD Instinct MI300X** serving `xbill9/gemma-4-E2B-it-qat-w8a8-ct-text-emb4` through
**vLLM** in Docker, on a DigitalOcean GPU droplet reached through AMD Developer Cloud.

Read the name per `../NAMING.md`: platform `gpu`, runtime `vllm`, hardware `mi300x`, model `2b`,
encoding **`w8a8emb4`**: int8 W8A8 from the QAT weights with the vocabulary tables stored int4.

| | |
| --- | --- |
| GPU | AMD Instinct MI300X VF, `gfx942`, 191.69 GiB (measured on the bf16 sibling) |
| Image | `vllm/vllm-openai-rocm:nightly-rocm100` — run `check_image` first |
| Model | `xbill9/gemma-4-E2B-it-qat-w8a8-ct-text-emb4` — 3.69 GB, text only, Apache-2.0, ungated |
| Kernel expected | none required; read the boot log |
| Context | 32768, matching the bf16 sibling |
| Cost | $1.99/hr, **billed while powered off too** |

**Status 2026-10-07: forked, not provisioned. Nothing measured.**

## Why it exists

One arm of the E2B data-type sweep on MI300X, `../gpu-vllm-mi300x-2b/DTYPE-SWEEP.md`: every E2B repack on Hugging Face
served on one card, one image digest, one day, against the bf16 reference.

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

## What is different from the bf16 sibling

- `LIMIT_MM_PER_PROMPT` is empty: the `-text` checkpoint has no towers, so the flag is dropped and
  `verify_capabilities` skips vision.
- `benchmarking_suite.py` takes the report's `weights_dtype`, `quantization` and `parameters_b`
  from `tpu.env`, so a quantized run cannot be filed as bf16.

Create and destroy are deliberately not tools. Both are dollar-per-hour decisions that stay a
human step in the DigitalOcean console.
