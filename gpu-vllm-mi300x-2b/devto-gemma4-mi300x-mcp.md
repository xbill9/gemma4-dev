---
title: "Driving an MI300X from Claude Code: Gemma 4 E2B, vLLM, and an MCP Server That Vets Its Own Image"
published: false
series: Gemma4
description: "Step by step deployment of Gemma 4 E2B to one AMD Instinct MI300X on a DigitalOcean GPU droplet, managed by a single-file Python MCP server on the MCP SDK 2.x. The server's first job is telling you which ROCm image can load the model at all."
tags: mcp, gemma, amd, claudecode
---

This article is a step by step deployment guide for Gemma 4 E2B onto one AMD Instinct MI300X, driven end to end from Claude Code. A single-file Python MCP server provides the tools to reach the droplet, vet the serving image, cache the weights, deploy vLLM, and verify every modality the checkpoint claims.

https://github.com/xbill9/gemma4-dev/tree/main/gpu-vllm-mi300x-2b

The measured porting story behind it — three images, one card, and the one line that separates them — is in [the amd-gputools write-up](https://github.com/xbill9/amd-gputools/blob/main/devto-gemma4-mi300x-vllm.md).

---

#### What is this project trying to Do?

This is a DevOps/SRE assistant for a Gemma 4 model served by vLLM on an AMD MI300X. The card lives on a DigitalOcean GPU droplet reached through AMD Developer Cloud, so the MCP server drives it entirely remotely: the DigitalOcean v2 API for lifecycle, SSH for everything else. **There is no GPU on the machine running the server, and nothing in it may assume one.** A `rocm-smi` in a local shell is a bug, not a check.

The AMD path differs from the NVIDIA and TPU rigs in this monorepo in one way that dominates everything else: **which image you pick decides whether the model loads at all**, and the newest vendor image is the one that fails. So the server's most useful tool is not the one that deploys — it is the one that vets an image before you spend 35 GB pulling it.

---

#### Where do I start?

The same incremental approach as the sibling rigs. First get the environment and Claude Code configuration working. Then bring up the MCP server over stdio and validate it against the droplet with read-only tools. Only then let it vet an image, cache weights, deploy, and verify.

The ordering matters more here than on a serverless target. Every step before `deploy_vllm` is cheap and reversible; the deploy is the one that takes minutes and holds 172 GB of VRAM.

---

#### At This Point You Should Have…

- Python 3.10 or newer — `mcp` 2.x declares `Requires-Python >=3.10`
- Claude Code installed and working
- A DigitalOcean or AMD Developer Cloud account with a GPU droplet already created
- That droplet **tagged** — this rig defaults to the tag `gemma`
- An SSH key that reaches it, and its path in `tpu.env`
- A `DIGITALOCEAN_ACCESS_TOKEN` from the **My AMD Team** account if you went through devcloud, not a personal one

The droplet is a deliberate prerequisite rather than something the server creates. **There are no create or destroy tools here, on purpose.** Both are dollar-per-hour decisions, and an MI300X droplet is $1.99/hour — so they stay a human step in the console.

---

#### Setup the Basic Environment

```shell
cd ~
git clone https://github.com/xbill9/gemma4-dev
cd gemma4-dev/gpu-vllm-mi300x-2b
./init.sh
```

`init.sh` installs the dependencies into the system `python3`, refreshes the skill snapshots, and registers the MCP server in this directory. It runs non-interactively — no `read` in its error path — so it is safe from a script.

On Debian 13 the `pip install` is refused by PEP 668. Use the distro packages:

```shell
apt-get install -y python3-httpx python3-dotenv
```

**Never create a virtualenv.** The constraint is the venv, not the install path: these rigs deploy as a docker image with system-wide site-packages, and a venv diverges the dev environment from the thing that actually runs.

Then the token, which never goes in the committed config:

```shell
echo 'DIGITALOCEAN_ACCESS_TOKEN=dop_v1_...' > .env && chmod 600 .env
```

`tpu.env` is committed and holds identifiers only. `.env` is gitignored, mode 0600, and holds the token. The server reads `tpu.env` at import and a real environment variable always wins over it.

---

#### What the Server Is Built On

A single file, `server.py`, on the **MCP Python SDK 2.x**:

```python
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

mcp = MCPServer(MCP_SERVER_NAME)
READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True)
```

2.x renamed FastMCP to `MCPServer`; `tool()`, `list_tools()` and `run()` are unchanged. Importing `mcp.server.fastmcp` against a 2.x install fails outright, so `requirements.txt` pins `mcp>=2,<3`.

The registered name is derived from the directory, not written out:

```python
RIG_NAME = PROJECT_DIR.name
MCP_SERVER_NAME = os.environ.get("MCP_SERVER_NAME", RIG_NAME)
```

That name prefixes every tool — `mcp__gpu-vllm-mi300x-2b__serving_status` — so two rigs registered under one key would be indistinguishable at the call site. Deriving it means a fork renames itself by being moved.

Every tool is `async def`, returns markdown, and ends in a blind `except Exception` that returns an `❌` string. That is the design, not an oversight: **an exception escaping a tool kills the server process for every later call.**

---

#### Reach the Hardware Before Trusting It

With the server registered, the first two calls are read-only:

```
list_droplets
gpu_status
```

If the card does not report, the answer is almost always the same one, and it is not a ROCm problem:

```
❌ /dev/kfd is missing, so no ROCm process can use the card.

📡 On a freshly provisioned droplet this is expected: amdgpu failed to bind
during provisioning and unloaded. Reboot it once — nothing needs installing.
```

On a fresh droplet `lspci` shows the card and `/dev/dri/renderD128` exists, but `amdgpu` has already failed to bind and unloaded. **Reboot, don't debug.**

`gpu_status` also encodes a trap worth stating plainly: **`rocm-smi` and `amd-smi` exit 0 when they fail.** With the driver uninitialised, `rocm-smi` prints "Driver not initialized" to stderr, prints nothing to stdout, and exits 0. An earlier version of this tool in the sibling toolkit trusted the exit code and reported a healthy `✅` over an empty table. The parsed output decides here; the exit code is not consulted at all.

---

#### Vet the Image — the Step That Saves the Day

```
check_image
```

This is the tool that earns its place on AMD. It runs a probe inside a candidate image and asks two questions:

```python
import vllm.config  # first, or a circular import makes a good image look broken
from vllm.transformers_utils.model_arch_config_convertor import (
    MODEL_ARCH_CONFIG_CONVERTORS as M,
)
print("convertor", M.get("gemma4"))
print("archs", [a for a in torch.cuda.get_arch_list() if "gfx94" in a])
```

**Does its vLLM register `Gemma4ModelArchConfigConvertor`?** Without it the image raises during config parsing, before the GPU is touched:

```
AmbiguousGlobalPerLayerAttributeError: 'head_dim' is a per-layer attribute
and may vary across layers.
```

Gemma 4 runs 256-wide heads on its sliding-attention layers and 512 on its full-attention ones. Transformers ≥ 5.15 models that honestly, making `head_dim` a per-layer attribute that *raises* on a global read. vLLM's generic `get_head_size()` asks for it with `getattr(..., 0)` — and the default never applies, because the accessor raises rather than returning nothing.

**Does its torch carry `gfx942`?** Note the probe maps `/dev/kfd` and `/dev/dri` in. Without devices, `torch.cuda.get_arch_list()` returns `[]`, which reads exactly like "this build has no kernels for you" and is nothing of the sort.

The answers, for the three images available today:

| Image | vLLM | transformers | Loads Gemma 4 |
| --- | --- | --- | --- |
| `vllm/vllm-openai-rocm:nightly-rocm100` | 0.29.1rc1.dev187 | 5.17.0 | ✅ |
| `rocm/vllm:...vllm_0.27.0` | 0.27.1.dev5 | 5.16.1 | ❌ |
| `rocm/vllm:...vllm_0.19.1` | 0.19.1 | 5.8.1 | ✅ |

**Newer is not safer.** The working images are the nightly and the *oldest* vendor build; the broken one sits between them. The pip channel vLLM's own Gemma 4 recipe names, `wheels.vllm.ai/rocm/nightly/rocm721`, is staler than all three at 0.20.2rc1 — older than the build that fails.

---

#### Stage and Deploy

```
pull_image
download_weights
deploy_vllm
```

Weights are cached on the host and mounted into the container, so a later image swap costs nothing in re-download. E2B is Apache-2.0 and ungated — no HF token, which turns a first run into a vLLM problem rather than a login problem.

`deploy_vllm` builds the argv, and one detail is keyed off the image name:

```python
def _is_official_image(image: str) -> bool:
    """True when the image already has `vllm serve` as its ENTRYPOINT."""
    return image.split(":", 1)[0].startswith("vllm/")
```

`vllm/vllm-openai-rocm` sets `ENTRYPOINT ["vllm","serve"]`, so **the model id is the first argument**. AMD's `rocm/vllm` images have no entrypoint and need the subcommand spelled out. Getting it backwards is an unrecognised-arguments error one way and `vllm: command not found` the other.

The flags the checkpoint needs:

```
--max-model-len 32768 --gpu-memory-utilization 0.90
--enable-auto-tool-choice --reasoning-parser gemma4 --tool-call-parser gemma4
--chat-template /app/vllm/examples/tool_chat_template_gemma4.jinja
--limit-mm-per-prompt '{"image": 4, "audio": 0}' --async-scheduling
```

The chat template ships **inside** the image, so it needs no mount.

---

#### Up Is Not Ready

```
serving_status
```

`docker run` returns in a second; the endpoint answers minutes later, after weights, `torch.compile`, and graph capture. So the tool reports the container and the endpoint as two separate facts:

```
📡 vllm on debian-gpu-mi300x1-192gb-devcloud-atl1

- container: Up 2 minutes
- image: vllm/vllm-openai-rocm:nightly-rocm100
- endpoint: not answering yet on 127.0.0.1:8000

📡 Container is up but the API is not serving yet — still loading.
```

"Up but loading" and "down" look identical if you only check one of them.

The endpoint is called over SSH against `127.0.0.1` rather than across the internet. There is no promise a firewall lets the workstation reach port 8000, and testing the port tells you about the firewall; testing from inside tells you about vLLM.

When it is ready, the KV budget on one card:

```
GPU KV cache size: 9,026,017 tokens
Maximum concurrency for 32,768 tokens per request: 275.45x
```

E2B shares KV across 20 of its 35 layers and runs one KV head at 256, so context is nearly free here. `--max-model-len 32768` is a workload choice; the checkpoint supports 131072 and this card would hold it.

---

#### Verify What It Claims

```
verify_capabilities
```

A flag being accepted is not evidence it did anything, so each modality gets a request whose correct answer is known in advance:

| Capability | | Result |
| --- | --- | --- |
| text | ✅ | The AMD MI300X is based on the CDNA 3 architecture. |
| thinking | ✅ | 1827 chars, 610 reasoning tokens |
| tool calling | ✅ | tool_calls → get_weather{"city": "Reykjavik"} |
| vision | ✅ | alternating bright red and royal blue squares |

The vision probe is a 158-byte checkerboard PNG embedded in `server.py` as base64, not a fetched photo. It needs no network from the droplet, and the right answer is known before the model gives one.

Thinking is on by default in this template — the plain text probe above spent 32 reasoning tokens without being asked — so budget `max_tokens` accordingly.

**Audio is not probed, and that is not an omission.** E2B carries a conformer audio encoder, and no ROCm vLLM image ships the `vllm[audio]` extras: `librosa` and `soundfile` are absent from all three. Audio requests fail at request time whatever `--limit-mm-per-prompt` says, so `audio: 0` is the honest setting — it also skips allocating encoder memory for a path that cannot be reached. Serving audio needs a derived image, not a flag.

---

#### Let the Rig Triage Itself

```
analyze_logs
```

This feeds the serving logs back to the model running on the card and asks it what is wrong. It is a neat trick with an obvious limit: it only works while the endpoint answers, which is exactly when you least need it. So a dead endpoint returns the raw tail instead of an error about the analysis.

The log filter drops `queue_controller.cpp` lines by default. On a VF they are benign, they appear on every queue creation, and they will drown everything you actually want to read.

---

#### Testing Without a GPU, a Droplet, or a Token

`make test` runs 33 offline cases. The whole `mcp` package is mocked before `server` is imported, so nothing reaches the API, opens SSH, or needs a token.

One detail is worth copying if you build your own:

```python
class _FakeMCPServer:
    def tool(self, *args, **kwargs):
        def decorator(fn):
            return fn
        return decorator
```

A bare `MagicMock` is not enough. `@mcp.tool()` would then return a MagicMock instead of the decorated coroutine, and every tool test fails with "'MagicMock' object can't be awaited" — which reads as a broken server and is really a broken fake.

The tests that matter most are the ones asserting the deployment contract: that the official image omits the `serve` subcommand and the vendor image includes it, that `audio` is 0, that both devices are mapped in, and that a missing convertor in `check_image` output produces an `❌` rather than a shrug.

---

#### Summary

The goal was to deploy Gemma 4 E2B to one MI300X and manage it from Claude Code. The key to the solution was making image selection a tool rather than a guess: on AMD the newest vendor image cannot load this checkpoint, and finding that out from a traceback costs a 62 GB pull and an hour. The results were:

- **20 MCP tools** in one file on the MCP SDK 2.x, driving the droplet over the DigitalOcean v2 API and SSH, with no local GPU assumption anywhere
- **`check_image` turns the expensive question into a free one** — convertor registry and `gfx942` arch list, read before anything is pulled
- Text, thinking, tool calling and vision **verified against the live endpoint**, each with a known-answer probe
- **Audio unreachable on every ROCm image** for want of two Python packages
- No create or destroy tools, because $1.99/hour decisions belong to a human
