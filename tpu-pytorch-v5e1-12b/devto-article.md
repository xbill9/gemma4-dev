---
title: "Teaching Claude Code to Wear the Pager: A TPU DevOps Skill Built on Google Cloud Flex-Start and MCP"
published: false
description: "How tpu-pytorch-v5e1-12b packages Google Cloud TPU operations as a Claude Code skill + MCP server — zone-sweeping capacity hunts, one-tool-call vLLM deploys, Gemma-4-powered log triage, an idiot-proof install guide, and a teardown step it will nag you about."
tags: ai, claudecode, googlecloud, tpu
cover_image: https://raw.githubusercontent.com/xbill9/tpu-pytorch-v5e1-12b/main/devto-cover.jpg
---

<!-- TODO before publishing: generate devto-cover.jpg and commit it, then fill in the real tool output in the Dogfooding section. -->

> **TL;DR:** [tpu-pytorch-v5e1-12b](https://github.com/xbill9/tpu-pytorch-v5e1-12b) wraps Google Cloud TPU operations in a tiny FastMCP server and packages it as a Claude Code skill. You type "find me a v6e chip and stand up Gemma 4" into Claude Code, and it just... does it. It sweeps zones until one grants capacity, boots the VM with a startup script that installs Docker and launches vLLM, polls until the model answers, and hands you an OpenAI-compatible endpoint. Then you say "benchmark it" and "something looks wrong, check the logs" — and the log triage is done by the very Gemma 4 model the agent just deployed. When you're done, it tears everything down. Without leaving your terminal.

## Background: why another infra tool?

Most TPU workflows are **forty gcloud invocations in a trench coat**. You look up which zones have your accelerator family, discover the quota that matters isn't visible in any `describe` command, guess a zone, wait, get a stockout, guess again, forget the boot disk is 10 GB and watch the vLLM image die of "no space left on device", SSH in (blocked), SSH in through IAP (works), and finally — twenty minutes and three browser tabs later — curl a health endpoint. Want to do it again next week? Hope you kept notes. (Narrator: they didn't.)

Google Cloud's **flex-start** capacity model — TPU chips granted on request, billed by the chip-hour until deleted — makes the *getting a chip* part genuinely accessible. What's missing is an operator who already knows the sharp edges.

This repo glues that operator into **Claude Code**, so your coding agent can provision, serve, verify, benchmark, and destroy TPU capacity as a natural part of a session. It ships as two things in one repo:

1. A **Model Context Protocol (MCP) server** (`tpu-devops`, a single-file FastMCP app in `server.py`) exposing thirty-one tools.
2. A **Claude Code skill** (`tpu-management`) that teaches Claude *when* and *how* to use those tools well.

## Flex-start: TPU capacity with a meter running

Flex-start is the interesting economic primitive here. The deal:

1. You ask for capacity (a GCE flex-start VM for v6e/v5p, or a queued resource for v5e).
2. Google grants it when available — sometimes instantly, sometimes not, sometimes only in a zone you didn't ask for.
3. From the moment it's granted, **the meter runs until you delete it** (v6e: $1.35/chip-hour), and the instance hard-stops at a 4-hour max run.

So the workflow is inherently bursty: acquire, serve, do your work, *tear down*. Perfect for an agent; miserable by hand. A few sharp edges the server handles for you:

- **Quota is per region AND per TPU family, and the dimension that matters is invisible** — `gcloud compute regions describe` won't show it. The only reliable probe is attempting creation, which fails fast on zero quota. The `find_tpu_vm` and `find_tpu` tools weaponize this: sweep every candidate zone, attempt creation, keep the first one that sticks, clean up the failures, and remember which zones struck out.
- **The default boot disk cannot hold vLLM.** 10 GB vs. a multi-GB `vllm/vllm-tpu:nightly` image is not a fair fight. The creation tools bake in the proven flags: 200 GB boot disk, a startup script that installs Docker, cloud-platform scopes.
- **Secrets never touch instance metadata.** The startup script fetches your Hugging Face token from Secret Manager at boot (with a ~30-minute retry loop, so an IAM grant applied late still lands).
- **SSH is often blocked; the serial console isn't.** The startup script mirrors its log to `/dev/console`, and `get_tpu_vm_serial_log` follows the boot from outside — no port 22 required.

## What is MCP, in one minute

The **Model Context Protocol** is an open standard for connecting AI assistants to tools and data. Before it, giving a model access to some service meant writing a bespoke integration for each assistant — N assistants × M services, everyone reinventing the same plumbing. MCP collapses that: a tool author writes one **MCP server** that exposes typed tools, and any MCP-capable client (Claude Code, Claude Desktop, and a growing list of others) can discover and call them with no per-client glue code.

An MCP server is usually a small local process that speaks JSON-RPC over stdio. The client launches it, asks "what tools do you have?", and from then on the model can call them like functions.

The `tpu-devops` server exposes thirty-one, organized around the serving lifecycle:

| Task | Tools |
| :--- | :--- |
| **Acquire (flex-start VMs, v6e/v5p)** | `create_tpu_vm_instance`, `find_tpu_vm` (zone sweep), `wait_for_vllm_ready`, `list_tpu_vm_instances`, `get_tpu_vm_serial_log`, `get_tpu_vm_endpoint`, `destroy_tpu_vm_instance` |
| **Acquire (queued resources, v5e)** | `find_tpu`, `create_tpu_queued_resource`, `manage_queued_resource`, `describe_queued_resource`, `list_queued_resources`, `get_zones_with_available_quota`, `find_gpu`, `estimate_deployment_cost`, `destroy_queued_resource` |
| **Serve** | `manage_vllm_docker` (start/stop/restart/status/log — also swaps models), `get_vllm_endpoint`, `get_vllm_deployment_config`, `save_hf_token` |
| **Diagnose** | `get_system_status`, `verify_model_health`, `get_model_details`, `get_metrics`, `get_vllm_docker_logs`, `get_tpu_system_logs`, `get_cloud_logging_logs`, `analyze_cloud_logging` (Gemma-4-powered log triage) |
| **Use & measure** | `query_queued_gemma4` (with optional TTFT/throughput stats), `run_vllm_benchmark`, `get_help` |

Errors come back as `❌ ...` text strings rather than protocol errors, so the agent can read them and react — retry a different zone, suggest the missing IAM grant, whatever the string says.

## And what's a Claude Code *skill*?

If MCP is the *hands* (the tools Claude can physically call), a **skill** is the *muscle memory* — a markdown file (`SKILL.md`) plus bundled resources that load into Claude's context and teach it the workflow: which tool to reach for, in what order, with which constraints.

For `tpu-management`, the skill encodes things like:

- **Status first.** `get_system_status` or `list_queued_resources` before creating anything — never provision on top of capacity you already have.
- The **standard lifecycle**: acquire → wait for ACTIVE → serve → verify → (benchmark) → tear down. Each step names its tool.
- The **required vLLM flags for Gemma 4 on TPU** (`--tool-call-parser gemma4`, `--disable_chunked_mm_input`, tensor-parallel sizing per accelerator) — so a hand-composed serve command is right the first time.
- **Field notes from live deployments**: the boot-disk trap, the IAP tunnel workaround, which QAT checkpoints are known-broken upstream and what to serve instead.
- Flex-start bills until deletion and can't be paused — **always confirm teardown of idle resources with the user**, and warn that `manage_queued_resource` is the destructive one (it deletes every *other* queued resource in the zone).

The skill also bundles the MCP server itself (`mcp/server.py`), its requirements, an installer script, the TPU VM startup-script template, and a full TPU getting-started guide (`references/tpu-guide.md`) — so it's self-contained: install the skill, and you have everything needed to also stand up the server.

## Installing it: the "I just want it to work" edition

You need four things: **Python 3.10+**, **Claude Code**, an authenticated **`gcloud` CLI** (with alpha components and the TPU API enabled), and a **Hugging Face token** for the model weights. Pick *one* of the paths below.

### Path A: The plugin marketplace (fewest keystrokes)

Inside Claude Code, type:

```
/plugin marketplace add xbill9/tpu-pytorch-v5e1-12b
/plugin install tpu-management@tpu-pytorch-v5e1-12b
```

This installs the skill **and** auto-registers the MCP server. The plugin manifest carries no project id or credentials (as it should!) — the server reads `GOOGLE_CLOUD_PROJECT`, `MODEL_NAME`, `ACCELERATOR_TYPE`, and friends from your environment, falling back to your active gcloud config.

### Path B: Clone and bootstrap (this repo)

```bash
# 1. Get the code
git clone https://github.com/xbill9/tpu-pytorch-v5e1-12b.git
cd tpu-pytorch-v5e1-12b

# 2. One-command setup: installs deps and registers the MCP server
#    in .mcp.json with your GCP project id
./init.sh

# 3. Restart Claude Code in this directory and approve the server
#    when prompted. Verify with:
/mcp        # should list tpu-devops
```

That's genuinely it. The installer is idempotent — rerun it if anything looks off.

### Path C: Install into *your* project

From a clone of the repo:

```bash
make init TARGET=/path/to/your/project ARGS='--project <gcp-project-id>'
```

This copies the skill into `<project>/.claude/skills/tpu-management/` and merges the `tpu-devops` entry into that project's `.mcp.json` without touching your other servers. Restart Claude Code in the target project, approve the server, done.

### Path D: Zip install (no clone at all)

```bash
curl -L -o /tmp/tpu-management-skill.zip \
  https://github.com/xbill9/tpu-pytorch-v5e1-12b/raw/main/dist/tpu-management-skill.zip
mkdir -p ~/.claude/skills && unzip -o /tmp/tpu-management-skill.zip -d ~/.claude/skills/
~/.claude/skills/tpu-management/mcp/project-setup.sh --global   # optional: register the MCP server
```

The zip is self-installing: the installer script rides along inside it, so the skill can stand up its own server anywhere.

### Troubleshooting, the whole guide

- `/mcp` doesn't list the server → restart Claude Code in the project directory.
- Creation fails instantly with `Quota 'TPUS_PER_TPU_FAMILY' exceeded. Limit: 0.0` → that region has no quota for your TPU family; let `find_tpu_vm` sweep, or request quota via the console.
- `gcloud compute ssh` hangs then times out → the block is upstream of your VPC; tunnel through IAP (`--tunnel-through-iap`). Boot progress needs no SSH at all — `get_tpu_vm_serial_log`.
- Anything else → ask Claude to call `get_help`; failures come back as readable `❌ ...` strings.

## Examples: a session in practice

Once installed, you talk to it in plain English. A real flow looks like:

**You:** *"Find me a v6e chip somewhere with capacity and stand up Gemma 4."*

Claude calls:

```python
find_tpu_vm()
# ✅ Created flex-start VM in europe-west4-a (us-east5: quota 0, skipped)

wait_for_vllm_ready()
# ✅ vLLM is serving!
# • Endpoint: http://<vm-ip>:8000/v1
# • Boot → serving: ~8m30s (image pull + XLA compile dominate)
```

**You:** *"Is it actually healthy? Give me a feel for the latency."*

```python
verify_model_health()
# ✅ Model healthy: google/gemma-4-E2B-it

query_queued_gemma4(prompt="Explain KV-cache sharing in two sentences.",
                    include_stats=True)
# ✅ TTFT: 16ms · 213 tok/s
```

**You:** *"Benchmark it at 8 and 64 concurrent streams."*

```python
run_vllm_benchmark(max_concurrency=8,  save_result=True)
run_vllm_benchmark(max_concurrency=64, save_result=True)
# ✅ c=8:  1,209 output tok/s · TTFT p50 27ms
# ✅ c=64: 2,140 output tok/s · TTFT p50 122ms
```

**You:** *"Requests started 500ing — what's going on?"*

```python
analyze_cloud_logging(query="severity>=ERROR", hours=1)
# ✅ Triage (by the deployed Gemma 4): OOM in the vLLM container after a
#   max-model-len override; recommend restart with the sized defaults.
```

Yes, that's the self-hosted model reading its own server's logs. It has opinions.

**You:** *"We're done for today — tear it all down."*

```python
destroy_tpu_vm_instance()
# ✅ Deleted. Flex-start billing stopped.
```

That last step is the one the skill will *push* you toward: flex-start bills until deletion, so an idle endpoint is a leaking faucet, and the agent is trained to mention it rather than politely watch you pay.

## Dogfooding: about those benchmark numbers 🐕🍖

If the term is new to you: **"eating your own dog food"** means using your own product for real work, not just demoing it. It's the difference between "this should work" and "I ship with this every day." If a tool is good enough for your users, it should be good enough for you — and if it isn't, you'll be the first to feel the pain and fix it.

This repo dogfoods itself at every layer:

- The **skill is active inside its own repository** — open Claude Code in a clone and the `tpu-management` skill and `tpu-devops` server are already wired up, so every development session doubles as an integration test.
- The repo's companion deep-dive — *[Gemma 4 E2B on a Single TPU v6e Chip](https://github.com/xbill9/tpu-pytorch-v5e1-12b/blob/main/devto-post.md)* — was **produced with this agent**: the concurrency sweep ran through `run_vllm_benchmark save_result=True`, the boot timeline came off `get_tpu_vm_serial_log`, and the cost table started from `estimate_deployment_cost`. The pipeline promoted in this article is the pipeline that produced its own field data.
- **The field notes flowed back into the skill.** The boot-disk trap, the IAP workaround, the QAT checkpoints that won't load (filed upstream as [tpu-inference #3225](https://github.com/vllm-project/tpu-inference/issues/3225)) — every one was hit for real during dogfooding, then encoded into `SKILL.md` so Claude sidesteps it next time.

<!-- TODO: run a fresh find → serve → benchmark → destroy flow and paste the real tool calls + output here. -->

Dogfooding is the cheapest credibility there is: no cherry-picked gallery, no "results may vary" fine print — the agent's real deployments produced the numbers in a published benchmark report, receipts and all. If the zone sweep had silently leaked half-provisioned VMs, my billing console would be the evidence.

## Links

- **Repo:** [github.com/xbill9/tpu-pytorch-v5e1-12b](https://github.com/xbill9/tpu-pytorch-v5e1-12b)
- **Flex-start (DWS) pricing:** [cloud.google.com/products/dws/pricing](https://cloud.google.com/products/dws/pricing)
- **vLLM on TPU docs:** [docs.vllm.ai/projects/tpu](https://docs.vllm.ai/projects/tpu/en/latest/)
- **Model Context Protocol:** [modelcontextprotocol.io](https://modelcontextprotocol.io)

*This is a third-party community project, not affiliated with or endorsed by Anthropic or Google. Bring your own GCP project and Hugging Face token — and remember flex-start capacity bills by the chip-hour until deleted, so acquire late, benchmark in batches, and make teardown the last tool call of every session.*
