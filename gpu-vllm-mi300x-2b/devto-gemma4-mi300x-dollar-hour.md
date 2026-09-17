---
title: "Serving Gemma 4 on an AMD MI300X: What $1.99 an Hour Buys"
published: false
description: "A step by step deployment of Gemma 4 E2B to a single AMD Instinct MI300X on AMD Developer Cloud, driven by Python MCP tools, and the throughput a 191.7 GiB card returns for its hourly rate."
tags: amd, vllm, rocm, machinelearning
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/gpu-vllm-mi300x-2b/devto-dollar-hour-cover.02475528.jpg
---

*This article provides a step by step deployment guide for **Gemma 4 E2B** to an **AMD Instinct MI300X** hosted GPU enabled system. A suite of Python MCP tools is built to simplify management of the vLLM hosted deployment. The card is reached through AMD Developer Cloud, which is DigitalOcean underneath, and the workstation driving it has no AMD GPU in it at all.*

[github.com/xbill9/gemma4-dev/tree/main/gpu-vllm-mi300x-2b](https://github.com/xbill9/gemma4-dev/tree/main/gpu-vllm-mi300x-2b)

| | |
|---|---|
| Model | `google/gemma-4-E2B-it`, reference bf16 release |
| Hardware | 1x AMD Instinct MI300X, `gfx942`, CDNA 3, 191.7 GiB HBM3 |
| Host | `gpu-mi300x1-192gb-devcloud` — 20 vCPU, 240 GB RAM, 720 GB disk, region `atl1` |
| Control plane | AMD Developer Cloud / DigitalOcean v2 API — no `gcloud`, no EC2 |
| Software | `vllm/vllm-openai-rocm:nightly-rocm100`, vLLM `0.3.1.dev3+g0bfc7a15d` |
| Rate | **$1.99 per hour**, on-demand |
| Result | **305 tok/s** single stream, **10,284 tok/s** at 64 streams, **2,713 tok/s per dollar-hour** |

---

## Where Do I Start?

The card is a DigitalOcean GPU droplet reached through AMD Developer Cloud (`devcloud.amd.com`) — same v2 API, same droplet ids, token from the My AMD Team account. Creating and destroying it are console actions, deliberately: both are dollar-per-hour decisions and neither belongs in a tool an agent can call.

Everything after creation is scripted. The droplet carries a tag, every tool is scoped to that tag, and nothing in the toolkit can touch an instance that does not have it.

## At This Point You Should Have…

- A GPU droplet running, tagged, and reachable by SSH key
- `DIGITALOCEAN_ACCESS_TOKEN` in a mode 0600 `.env` beside the server
- Docker on the droplet with `/dev/kfd` and `/dev/dri` present
- A Hugging Face token on the droplet for the gated Gemma 4 weights
- Python 3 on your workstation, and no expectation of a GPU in it

## There Is No GPU on the Machine You Run This From

`rocm-smi`, `amd-smi`, `rocminfo` and `hipcc` do not exist locally and never will. A ROCm command in a local shell is a bug, not a check. Every reading in this article came back over SSH or through the DigitalOcean API.

That constraint shapes the whole toolkit. The MCP server runs on the workstation, holds no device, and reaches the hardware the same way you would by hand.

## The MCP Server Speaks stdio

One file, `server.py`, exposes 21 tools over stdio: droplet lifecycle, image checks, deploy, logs, benchmark cells, and the SRE probes below. Every subprocess goes through one `run_command(cmd: list[str])` helper using `asyncio.create_subprocess_exec` — never a shell.

```
$ grep -c "^@mcp.tool" server.py
21
```

## What the Droplet Is

```
$ droplet_status
📡 debian-gpu-mi300x1-192gb-devcloud-atl1 (601418522)

- status: `active`
- size: `gpu-mi300x1-192gb-devcloud` — 20 vCPU, 240 GB RAM, 720 GB disk
- region: atl1
- cost: $1.99/hr — billed while powered off, too
```

The rate comes from the DigitalOcean v2 API's own `droplet.size.price_hourly` field, read at the time of the run, not from a pricing page.

## The Card Before Anything Runs

```
$ gpu_status
✅ GPU reporting on `debian-gpu-mi300x1-192gb-devcloud-atl1`.

| Card | Product | GPU use % | VRAM used % |
| card0 | Aqua Vanjaram [Instinct MI300X VF] | 0 | 0 |
```

It presents as an SR-IOV virtual function, which looks like a partition and is not one: all 304 compute units and the whole 191.7 GiB are there. Neither `rocm-smi` nor `amd-smi` sets a useful exit code, so the tool parses the output and ignores the status code entirely.

## Which Image

`vllm/vllm-openai-rocm:nightly-rocm100`. Read the version out of the running container rather than trusting the tag, because a nightly tag moves under you:

```
$ docker exec vllm python3 -c "import vllm;print(vllm.__version__)"
0.3.1.dev3+g0bfc7a15d
```

Not every ROCm vLLM image can load this checkpoint — Gemma 4 runs 256-wide heads on its sliding-attention layers and 512 on its full-attention ones, and an image whose vLLM predates that split cannot parse the config. The measured comparison of three images is in the [companion article](https://github.com/xbill9/amd-gputools). Here the practical rule is enough: pin what you measured, and check a new image before you adopt it.

## Starting the Server

```
$ deploy_vllm
✅ Started `vllm` serving `google/gemma-4-E2B-it`.

- image: `vllm/vllm-openai-rocm:nightly-rocm100`
- context: 32768, gpu-memory-utilization 0.90
- multimodal: `{"image": 4, "audio": 0}`
```

The serve arguments, in full:

```
vllm serve google/gemma-4-E2B-it --host 0.0.0.0 --port 8000 \
  --max-model-len 32768 --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --reasoning-parser gemma4 --tool-call-parser gemma4 \
  --chat-template /app/vllm/examples/tool_chat_template_gemma4.jinja \
  --limit-mm-per-prompt '{"image": 4, "audio": 0}' --async-scheduling
```

`audio` is `0` on purpose. E2B has a conformer audio encoder and no ROCm vLLM image ships the `vllm[audio]` extras, so a non-zero limit would allocate encoder memory for a path that cannot be used.

## Up Is Not Ready

`docker run` returns in about a second. The endpoint does not answer for another two and a half minutes, while weights load, `torch.compile` runs and graphs are captured.

```
# polling /v1/models every 10s from the moment docker run returned
READY after 160s
{"object":"list","data":[{"id":"google/gemma-4-E2B-it","max_model_len":32768,...}]}
```

This is why `serving_status` reports the container and the endpoint as two separate facts. A running container is not a serving model, and conflating them turns a normal boot into a phantom hang.

```
$ serving_status
✅ container: Up 3 minutes | endpoint: answering on `127.0.0.1:8000`
   served: `google/gemma-4-E2B-it`
```

## Does It Actually Serve?

A flag being accepted is not evidence it did anything, so every modality is probed with a request whose correct answer is known in advance. The vision probe is a generated checkerboard rather than a fetched photo, so the expected answer is a fact about the image and not a caption.

```
$ verify_capabilities
✅ 4/4 capabilities verified.

| text         | ✅ | The AMD MI300X is based on the CDNA 3 architecture. |
| thinking     | ✅ | 1404 chars, 447 reasoning tokens |
| tool calling | ✅ | tool_calls → get_weather{"city": "Reykjavik"} |
| vision       | ✅ | alternating bright red and royal blue squares |

📡 Audio is not probed — no ROCm vLLM image ships the `vllm[audio]` extras.
```

## What the Engine Allocates

The boot log is worth reading before sizing anything, because the engine states its own arithmetic:

```
Available KV cache memory: 155.04 GiB
GPU KV cache size: 9,026,017 tokens
Maximum concurrency for 32,768 tokens per request: 275.45x
```

Nine million tokens of KV on one card. Hold that number — it decides which of the two ceilings below you actually hit.

## The Sweep

Four concurrencies against four context lengths, 128 output tokens throughout, three repeats per cell, load generated by vLLM's own bench client in a second container with no GPU device attached.

```
$ python3 benchmarking_suite.py --droplet debian-gpu-mi300x1-192gb-devcloud-atl1 \
    --run-id 2026-09-17-vllm-sweep-mi300x-rerun --repeat 3 --seed-base 27000
16 cells, 12 runnable, 4 infeasible at max_model_len 32768
```

The 32,768 row is infeasible rather than missing: 32,768 input plus 128 output does not fit a 32,768 context, and a cell that cannot exist is recorded as such rather than dropped.

One detail is load-bearing. `vllm bench serve` derives its prompts from `--seed`, which defaults to 0, and this deployment caches prefixes — so two runs sharing a seed measure the cache, not the card. Every cell and every repeat here uses a seed no other run in the sweep uses.

## Raw Tokens per Second

Output tokens per second, median of three repeats, worst-cell coefficient of variation 7.96%:

| context | 1 stream | 4 | 16 | 64 |
|---|---|---|---|---|
| 128 | 340.9 | 1,124.1 | 3,569.1 | 🥇 **10,284.0** |
| 1,024 | 305.1 | 969.1 | 2,653.6 | 5,398.2 |
| 8,192 | 192.8 | 445.5 | 688.2 | 702.7 |

Counting prefill as well, the busiest cell moves **48,583 tokens a second** — 64 streams at 1,024 context. Single-stream latency at 1,024 context is 32.67 ms to first token and 3.04 ms per token after it.

## Where the Ceiling Is

Two things happen as context grows, and only one of them is the one people plan for.

Time per output token barely moves: 2.85 ms at 128 context, 3.04 at 1,024, 3.9 at 8,192. Time to first token rises steeply: 12.97 ms, 32.67 ms, 169.61 ms. Decode stays cheap; prefill gets expensive.

So the 8,192 row flattens — 688 tok/s at 16 streams, 703 at 64 — while the 128 row keeps climbing to 10,284. **The constraint at long context is prefill, not memory.** The heaviest cell in the grid wants 64 x 8,192 = 524,288 KV tokens against the 9,026,017 the engine allocated — 5.8% of the pool, by arithmetic.

That inverts the sizing rule the TPU rigs in this monorepo run on, where KV capacity is the thing you run out of first. On a 192 GB card serving a 2B model, it never becomes the binding constraint.

## And Price/Performance?

Throughput divided by hourly rate, in=1024, out=128. Every row is the same checkpoint under vLLM, each from its own schema-valid report:

| rig | $/hr | basis | 1 | 4 | 16 | 64 |
|---|---|---|---|---|---|---|
| **MI300X** | 1.99 | on-demand | 153 | 487 | 1,333 | 🥇 **2,713** |
| TPU v5e-1 | 0.5779 | spot | 🥇 208 | 🥇 711 | 🥇 1,572 | 1,972 |
| TPU v6e-1 | 2.97 | on-demand | 67 | 233 | 587 | 719 |
| NVIDIA L4 | 0.94 | spot | 49 | 187 | — | — |

Tokens per second per dollar-hour. The MI300X wins the busiest column outright and the v5e wins the other three — but those two rows are not priced the same way, which is the whole of the next section.

## The Same Table, Priced the Same Way

A spot rate against an on-demand rate is a discount, not a hardware result. Put all three on on-demand list and the ranking is uniform:

| rig | $/hr | 1 | 4 | 16 | 64 |
|---|---|---|---|---|---|
| 🥇 **MI300X** | 1.99 | **153** | **487** | **1,333** | **2,713** |
| 🥈 TPU v5e-1 | 1.20 | 100 | 342 | 757 | 950 |
| 🥉 TPU v6e-1 | 2.70 | 74 | 257 | 646 | 790 |

At matched pricing the MI300X returns **1.4x to 2.9x** the tokens per dollar of a v5e — 1.53x, 1.42x, 1.76x and 2.86x across the four concurrencies, by arithmetic over the two measured rows above.

## The Spot Asterisk

That advantage has one real qualifier: DigitalOcean publishes no preemptible tier for GPU droplets, so $1.99 is a floor. The v5e spot rate of $0.5779 per chip-hour is a purchasable option, and at 16 streams or fewer it buys more tokens per dollar than this card does.

So the honest form of the finding is conditional. If you want a non-preemptible box, or you keep dozens of streams busy, this is the best tokens-per-dollar measured across these rigs. If your traffic is light and you tolerate preemption, a spot TPU is cheaper.

## What Stops the Meter

Powering the droplet off does **not** stop DigitalOcean billing it. The resources stay reserved and the hourly rate keeps running; only destroying the droplet stops the meter.

That is why this toolkit ships no `create` and no `destroy` tool. Both are dollar-per-hour decisions, they stay a deliberate step in the console, and a rate quoted per hour means nothing until you know which hours you are paying for.

```
$ stop_vllm
✅ `docker rm -f vllm` exited 0.
```

That frees the card's memory. It does not free your wallet.

## Summary

The goal of this article was to deploy Gemma 4 E2B to a single AMD Instinct MI300X and measure what its hourly rate returns in tokens. The key to the solution was a tag-scoped MCP toolkit that reaches the card over SSH and the DigitalOcean API, because the machine running it has no GPU. The measured results were:

- **305 tok/s** single stream at 1,024 context, 32.67 ms to first token
- **10,284 tok/s** at 64 streams and 128 context; **48,583 tok/s** counting prefill
- **2,713 tokens per second per dollar-hour**, 1.4x to 2.9x a TPU v5e at matched on-demand pricing
- **160 seconds** from `docker run` to an endpoint that answers
- KV pool of **9,026,017 tokens**, of which the heaviest cell used 5.8% — the long-context ceiling is prefill, not memory
- Text, thinking, tool calling and vision all verified against the live endpoint; audio is unreachable on every ROCm image tried

One droplet, one MI300X, region `atl1`, three repeats per cell with a unique prompt seed each, worst-cell coefficient of variation 7.96%. The comparison rows differ from this one in ways worth naming once: the TPU and L4 reports are single runs per cell on earlier vLLM builds, the v5e and L4 figures were measured on spot capacity while the MI300X and v6e were on-demand, and nothing else in this monorepo serves this checkpoint on AMD, so there is no twin to difference against.

The strategy for using MCP for AMD Instinct deployment and benchmarking was validated with an incremental step by step approach.
