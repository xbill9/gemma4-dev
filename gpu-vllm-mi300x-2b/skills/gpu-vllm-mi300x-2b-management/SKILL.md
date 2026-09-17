---
name: gpu-vllm-mi300x-2b-management
description: Manage the AMD MI300X droplet serving Gemma 4 E2B through vLLM. Use when the user asks about MI300X, Instinct, gfx942, ROCm, AMD Developer Cloud or DigitalOcean GPU droplets, about starting, stopping or debugging vLLM on AMD, about which ROCm vLLM image can load Gemma 4, or about thinking, tool calling, vision or audio on a self-hosted Gemma 4 endpoint. Triggers include "MI300X", "ROCm", "gfx942", "rocm-smi", "vllm on AMD", "gemma 4 endpoint", "/dev/kfd".
---

# MI300X vLLM Management

Operate one AMD Instinct MI300X serving `google/gemma-4-E2B-it` through vLLM: reach the
droplet, verify the card, start the model server, and diagnose the endpoint. Two ways to act:

1. **Preferred — MCP agent tools.** If the `gpu-vllm-mi300x-2b` MCP server is connected in
   this session, use its tools (catalog below). They carry the flags this checkpoint needs
   and the image caveats that decide whether it loads at all.
2. **Fallback — direct `ssh` + `docker`.** If the server is not connected, offer to register
   the bundled one (see "Registering the MCP server"), or run the equivalent commands from
   the "Standard lifecycle" section.

## There is no GPU on the machine running this skill

The card lives on a DigitalOcean GPU droplet reached through AMD Developer Cloud. `rocm-smi`,
`amd-smi`, `rocminfo` and `hipcc` are not local and never will be. A ROCm command in a local
shell is a bug, not a check — reach the hardware through `run_on_droplet`, `gpu_status`, or
explicit `ssh`.

**Powering a droplet off does not stop DigitalOcean billing it.** The resources stay reserved
and the hourly rate keeps running. Only destroying it stops the meter, and this server
deliberately has no tool that destroys one.

## Bundled files

- `mcp/server.py` — the MCPServer agent (snapshot of the rig-root `server.py`; the rig-root
  copy is authoritative if the two differ).
- `mcp/project-setup.sh` — one-command installer: copies this skill into a target project and
  registers the MCP server.
- `mcp/requirements.txt`, `mcp/tpu.env` — dependencies and the committed configuration.

## The image decides whether the model loads at all

This is the single most expensive thing to get wrong, and it is free to check.

| Image | vLLM | transformers | Serves Gemma 4 |
| --- | --- | --- | --- |
| `vllm/vllm-openai-rocm:nightly-rocm100` | 0.29.1rc1.dev187 | 5.17.0 | ✅ current default |
| `rocm/vllm:...vllm_0.27.0` | 0.27.1.dev5 | 5.16.1 | ❌ **cannot load it** |
| `rocm/vllm:...vllm_0.19.1` | 0.19.1 | 5.8.1 | ✅ older fallback |

The middle row is the newest image AMD publishes. It fails in config parsing, before the GPU
is touched:

```
AmbiguousGlobalPerLayerAttributeError: 'head_dim' is a per-layer attribute
```

Gemma 4 runs 256-wide heads on sliding-attention layers and 512 on full-attention ones.
Transformers ≥ 5.15 reports `head_dim` as per-layer and raises on a global read; vLLM's
generic `get_head_size()` asks with `getattr(..., 0)`, and the default never applies because
the accessor raises rather than returning nothing. Upstream fixes it with
`Gemma4ModelArchConfigConvertor`, which that build predates.

**Run `check_image` before changing `VLLM_IMAGE`.** It reads the convertor registry and the
gfx942 arch list out of a candidate image, which turns a 62 GB mistake into a question.

The pip route in vLLM's own Gemma 4 recipe (`wheels.vllm.ai/rocm/nightly/rocm721`) is staler
than both vendor images — 0.20.2rc1. Container nightlies are the only current path.

## Two flag shapes, keyed off the image

`vllm/vllm-openai-rocm` sets `ENTRYPOINT ["vllm","serve"]`, so **the model id is the first
argument**. AMD's `rocm/vllm` images have no entrypoint and need `vllm serve` spelled out.
`deploy_vllm` picks the right one; if you are typing docker by hand, getting this backwards
is an unrecognised-arguments error one way and `vllm: command not found` the other.

## Required serving flags

```
--max-model-len 32768 --gpu-memory-utilization 0.90
--enable-auto-tool-choice --reasoning-parser gemma4 --tool-call-parser gemma4
--chat-template /app/vllm/examples/tool_chat_template_gemma4.jinja
--limit-mm-per-prompt '{"image": 4, "audio": 0}' --async-scheduling
```

The chat template ships **inside** the image, so it needs no mount. `--max-model-len 32768`
is a workload choice — the checkpoint supports 131072 and this card would hold it; at 32768
the measured KV budget is 9,026,017 tokens and 275.45x concurrency.

## Audio is off and a flag will not turn it on

E2B carries a conformer audio encoder. No ROCm vLLM image ships the `vllm[audio]` extras —
`librosa` and `soundfile` are absent from every image tried — so audio requests fail at
request time whatever `--limit-mm-per-prompt` says. `audio: 0` is the honest setting and also
skips allocating encoder memory for an unreachable path. Serving audio needs a derived image,
not a configuration change.

## Standard lifecycle

1. `list_droplets` — what is tagged, and is it on.
2. `gpu_status` — does the card report. **If `/dev/kfd` is missing, reboot once**: on a fresh
   droplet `amdgpu` fails to bind during provisioning and unloads, leaving a card `lspci` can
   see and nothing can use. Nothing needs installing. Use `reboot_droplet`.
3. `check_image` — can this image load the checkpoint.
4. `pull_image`, `download_weights` — 35 GB and 10 GB respectively; weights are cached on the
   host and mounted in, so a later image swap costs no re-download.
5. `deploy_vllm` — start serving. Returns as soon as the container starts, which is minutes
   before the endpoint answers.
6. `serving_status` — container state and endpoint readiness, reported separately so "up but
   still loading" is distinguishable from "down".
7. `verify_capabilities` — text, thinking, tool calling and vision, each probed with a request
   whose right answer is known in advance.
8. `analyze_logs` — the rig triaging itself; falls back to the raw tail when the endpoint is
   the thing that is broken.

## Tool catalog

| Tool | What it does |
| --- | --- |
| `list_droplets` | Every droplet tagged for this rig, with status and address |
| `droplet_status` | One droplet's power state, size, address and hourly cost |
| `start_droplet` | Power on |
| `stop_droplet` | Power off — **does not stop billing** |
| `reboot_droplet` | Reboot — the fix for a missing `/dev/kfd` |
| `action_status` | Poll a DigitalOcean action id |
| `ssh_command` | Print the ssh command with the address resolved |
| `run_on_droplet` | Run one command over SSH |
| `gpu_status` | Parse `rocm-smi`; never trusts its exit code |
| `check_image` | Ask an image whether it can load Gemma 4, before pulling it |
| `pull_image` | Pull the serving image |
| `download_weights` | Cache the checkpoint into `HF_CACHE` |
| `deploy_vllm` | Start the model server with this rig's flags |
| `stop_vllm` | Stop it and release the card |
| `serving_status` | Container state + endpoint readiness |
| `server_logs` | Tail the logs, minus the benign ROCm queue warning |
| `query_model` | One chat completion, reporting reasoning tokens |
| `verify_capabilities` | Probe all four working modalities |
| `analyze_logs` | Feed the logs to the model and ask what is wrong |
| `get_help` | This server's tools and active configuration |

## Cautions

- **`rocm-smi` and `amd-smi` exit 0 when they fail.** With the driver uninitialised,
  `rocm-smi` prints "Driver not initialized" to stderr, nothing to stdout, and exits 0. Never
  branch on their exit status — parse the output. `gpu_status` does.
- **`torch.cuda.get_arch_list()` returns `[]` with no devices mapped in.** That reads like
  "this build has no kernels for you" and is nothing of the sort. Map `/dev/kfd` and
  `/dev/dri` in for the check or do not believe the answer.
- **Import `vllm.config` before `vllm.transformers_utils.*`**, or a circular-import
  `ImportError` makes a working image look broken.
- **Thinking is on by default in this chat template.** A plain prompt spends reasoning
  tokens; budget `max_tokens` accordingly.
- **A running container holds the card's memory.** A second one asking for 0.90 of it fails
  on free VRAM, not on anything interesting. Use `deploy_vllm(replace=True)`.
- **No create or destroy tools, by design.** Both are dollar-per-hour decisions and stay a
  deliberate step in the DigitalOcean console.

## Registering the MCP server

```bash
./mcp/project-setup.sh /path/to/project          # per-project .mcp.json
./mcp/project-setup.sh --global                  # user scope
```

The registered key must equal the rig directory name — it prefixes every tool
(`mcp__gpu-vllm-mi300x-2b__serving_status`), and two rigs registered under one key are
indistinguishable at the call site.

The DigitalOcean token goes in `.env` (gitignored, mode 0600) or the environment, as
`DIGITALOCEAN_ACCESS_TOKEN`. **Never in `tpu.env`**, which is committed.

## Further reading

Neither is embedded here — they are published, so a snapshot would only drift.

- [`devto-gemma4-mi300x-mcp.md`](../../../devto-gemma4-mi300x-mcp.md) (rig root) — step-by-step
  deployment of this rig through these tools, with the MCP SDK 2.x details.
- [The port write-up](https://github.com/xbill9/amd-gputools/blob/main/devto-gemma4-mi300x-vllm.md)
  in `amd-gputools` — the three-image comparison this rig's image caveat comes from, including the
  `head_dim` traceback in full and the two free pre-pull checks.
