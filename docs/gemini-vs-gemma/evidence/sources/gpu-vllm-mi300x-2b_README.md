# gpu-vllm-mi300x-2b

One **AMD Instinct MI300X** serving `google/gemma-4-E2B-it` through **vLLM** in Docker, on a
DigitalOcean GPU droplet reached through AMD Developer Cloud.

Read the name per `../NAMING.md`: platform `gpu` (a general-purpose GPU attached to a VM, any
cloud — the cloud provider is deliberately not a slot), runtime `vllm`, hardware `mi300x`,
model `2b`. No fifth slot: these are the reference bf16 weights, not a quantized encoding.

| | |
| --- | --- |
| GPU | AMD Instinct MI300X VF, `gfx942:sramecc+:xnack-`, 191.69 GiB |
| Host | Debian 13 (trixie), kernel `6.12.94+deb13-amd64`, 20 vCPU, 240 GB RAM |
| Image | `vllm/vllm-openai-rocm:nightly-rocm100` |
| vLLM | 0.29.1rc1.dev187+gaf1c01499 · torch 2.12.0+rocm10.0.0 · transformers 5.17.0 |
| Model | `google/gemma-4-E2B-it` — effective 2B, Apache-2.0, ungated, 10.28 GB bf16 |
| Context | 32768 of a supported 131072 |
| KV cache | 9,026,017 tokens — 275.45x concurrency at 32768 |
| Cost | $1.99/hr, **billed while powered off too** |

Status 2026-09-16: serving. Text, thinking, tool calling and vision verified against the live
endpoint. Audio is unreachable — see below.

## The sibling rule

Rigs here are siblings, not layers. Nothing is imported across a rig boundary, and a pattern
in one rig does not hold in another. This is the only AMD rig in the monorepo and the only one
whose control plane is DigitalOcean rather than Google Cloud or EC2, so **do not carry a
`gcloud` assumption into it**. Its closest relative is `~/amd-gputools`, a different repo.

## Quick start

```bash
./init.sh                       # deps, skill snapshots, MCP registration
echo 'DIGITALOCEAN_ACCESS_TOKEN=...' > .env && chmod 600 .env
```

Then, through the agent:

```
list_droplets → gpu_status → check_image → deploy_vllm → serving_status → verify_capabilities
```

`tpu.env` is the committed source of truth for everything else. It is named `tpu.env` like
every sibling including the `gpu-*` and `local-*` ones; the name is a monorepo convention, not
a claim that this rig is a TPU.

## The image is the whole problem

| Image | vLLM | transformers | Serves Gemma 4 |
| --- | --- | --- | --- |
| `vllm/vllm-openai-rocm:nightly-rocm100` | 0.29.1rc1.dev187 | 5.17.0 | ✅ |
| `rocm/vllm:rocm10.0.0_..._vllm_0.27.0` | 0.27.1.dev5 | 5.16.1 | ❌ |
| `rocm/vllm:rocm7.13.0_gfx94X-dcgpu_..._vllm_0.19.1` | 0.19.1 | 5.8.1 | ✅ |

**The newest image AMD publishes cannot load this model.** It dies in config parsing, before
the GPU is touched:

```
AmbiguousGlobalPerLayerAttributeError: 'head_dim' is a per-layer attribute
and may vary across layers.
```

Gemma 4 runs 256-wide heads on its sliding-attention layers and 512 on its full-attention
ones. Transformers ≥ 5.15 reports `head_dim` as a per-layer attribute and raises on a global
read. vLLM's generic `get_head_size()` asks for it with `getattr(..., 0)` — and the default
never applies, because the accessor raises rather than returning nothing. Upstream fixes this
with `Gemma4ModelArchConfigConvertor`; that build predates it.

This is a **version-pairing bug, not a ROCm one**. The same image runs bf16 GEMMs on gfx942
correctly and would serve a model whose head dimensions are uniform. The 0.19.1 image escapes
it from the other side: transformers 5.8.1 predates per-layer attributes entirely.

`check_image` asks a candidate image both questions — the convertor registry and the gfx942
arch list — before anything is pulled.

The pip route in vLLM's own Gemma 4 recipe, `wheels.vllm.ai/rocm/nightly/rocm721`, is staler
than either vendor image at 0.20.2rc1. **Container nightlies are the only current path**, and
they rebuild daily: pin `nightly-rocm100-<sha>` if drift matters.

## What works, and what does not

| Capability | Status |
| --- | --- |
| Text | ✅ |
| Thinking (`reasoning` field, `gemma4` parser) | ✅ — on by default in this template |
| Tool calling (`gemma4` parser, auto tool choice) | ✅ |
| Vision (image input, 280-token default budget) | ✅ |
| Audio | ❌ **not a flag away** |

E2B carries a conformer audio encoder, and no ROCm vLLM image ships the `vllm[audio]` extras
— `librosa` and `soundfile` are absent from all three tried, so audio fails at request time
whatever `--limit-mm-per-prompt` says. `audio: 0` is the honest setting and skips allocating
encoder memory for a path that cannot be reached. Serving audio needs a derived image.

## Gotchas that cost time

- **A fresh droplet has no `/dev/kfd`.** `amdgpu` fails to bind during provisioning and
  unloads, leaving a card `lspci` can see and no ROCm process can use. **Reboot once** —
  nothing needs installing.
- **`rocm-smi` and `amd-smi` exit 0 when they fail.** Parse the output; never branch on the
  exit status.
- **`torch.cuda.get_arch_list()` returns `[]` without devices mapped in**, which looks like a
  build with no kernels for you and is not.
- **The official image's ENTRYPOINT is already `["vllm","serve"]`** — the model id is the
  first argument. AMD's images need the subcommand spelled out.
- **Powering the droplet off does not stop billing.** Only destroying it does, and no tool
  here destroys one.

## Layout

```
server.py         the MCP server — 19 tools, single file, MCPServer
tpu.env           committed configuration, no secrets
tests/            offline unittest; the mcp package is mocked before import
.claude/skills/   generated snapshot — edit the sources, then `make skill`
```

`make check` runs lint and the tests. Tests are `unittest`, never pytest, and are fully
offline: no API call, no SSH, no token.
