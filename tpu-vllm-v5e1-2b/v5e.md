---
title: Serving Gemma 4 2B on a Single TPU v5e Chip with MCP and Antigravity CLI
published: true
series: Gemma4
date: 2026-08-05 09:45:00 UTC
tags: antigravitycli,python,mcps,gemma
---

This article is the v5e follow-on to the [v6e-1 debugging guide](https://xbill999.medium.com/debugging-deployments-with-gemma-4b-tpu-v6e-1-mcp-and-antigravity-cli-c9846231237a). Same MCP tooling, same Antigravity CLI driver, smaller and cheaper silicon — and a different set of failure modes. This time the model is `google/gemma-4-E2B-it` on a single Cloud TPU v5e chip, with measured throughput, measured latency, and a cost breakdown that actually uses the numbers from the sweep.

#### What this project is trying to do

Same brief as before: a DevOps/SRE assistant whose brain is a self-hosted Gemma 4 model. The MCP server provisions the TPU, deploys the vLLM container, discovers the endpoint, and then uses that endpoint to analyze Cloud Logging output. 31 tools, one `server.py`, stdio transport.

What changed is the target: **TPU v6e-1 → TPU v5e-1**, and **Gemma 4 4B → Gemma 4 2B**.

The interesting question this article answers: v5e is roughly half the price of v6e per chip-hour. Is it half the machine, or worse?

#### The chip, on paper

| Spec (per chip) | TPU v5e (`v5litepod`) | TPU v6e (Trillium) | Ratio |
| --- | --- | --- | --- |
| HBM capacity | 16 GB | 32 GB | 2.0× |
| HBM bandwidth | 800 GiBps | 1,638 GBps | ~2× |
| Peak BF16 | 197 TFLOPs | 918 TFLOPs | 4.66× |
| Peak INT8 | 393 TOPs | 1,836 TOPs | 4.67× |
| Single-chip machine type | `ct5lp-hightpu-1t` | `ct6e-standard-1t` | — |
| On-demand list | ~$1.20 / chip-hr | ~$2.70 / chip-hr | 2.25× |
| Flex-start (see below) | ~$0.60 / chip-hr | ~$1.35 / chip-hr | 2.25× |

Spec figures are Google's own, from the [v5e](https://docs.cloud.google.com/tpu/docs/v5e) and [v6e](https://docs.cloud.google.com/tpu/docs/v6e) documentation. Note that Google quotes v5e bandwidth in **GiBps** and v6e in **GBps** — normalize the units and the real ratio is ~1.9×, not a clean 2×. I'm not going to pretend that's a meaningful difference at this altitude, but don't quote "2× the bandwidth" as if it were exact.

That table is the whole story in miniature. v6e is **2.25× the price** for **2× the memory and roughly 2× the bandwidth** but **4.7× the raw FLOPS**. For a 2B decode-bound workload — which is bandwidth-bound, not FLOPS-bound — v5e is priced almost exactly right. For prefill-heavy or long-context work, where you actually burn the matrix units, v6e's 4.7× starts to matter.

The sweep below bears this out.

#### Antigravity CLI

Antigravity CLI is the successor to Gemini CLI — the terminal-driven, agent-assisted coding tool. Install instructions:

[Getting Started with Antigravity CLI](https://medium.com/google-cloud/getting-started-with-antigravity-cli-26c5da90951f)

Start it and authenticate against a Google Cloud project:

```plaintext
agy
```

#### One config file, four consumers

The single biggest structural change from the v6e rig: deployment parameters live in exactly one place, `tpu.env`, and everything reads from it.

```shell
# tpu.env — the single source of truth for this rig.
GOOGLE_CLOUD_PROJECT=aisprint-491218
GOOGLE_CLOUD_REGION=us-west4
GOOGLE_CLOUD_ZONE=us-west4-a

MODEL_NAME=google/gemma-4-E2B-it
ACCELERATOR_TYPE=v5litepod-1
TENSOR_PARALLEL_SIZE=1
```

It is consumed by `server.py` (via `load_dotenv`), `mcp-run.sh` (which is what the MCP config actually launches), the `Makefile` (via `-include`), and `set_env.sh`. A real environment variable always wins over the file in all four — `load_dotenv` doesn't overwrite, the wrapper only exports what's unset, and the Makefile uses `?=` — so `GOOGLE_CLOUD_ZONE=europe-west4-a make status` still works as a one-off override.

Change the zone once. Not in five places. This sounds like housekeeping until you spend an afternoon debugging a deploy that was reading a stale zone out of `mcp_config.json`.

The MCP config is correspondingly boring, which is the point:

```json
{
  "mcpServers": {
    "tpu-2B-v5e1-devops-agent": {
      "command": "./mcp-run.sh",
      "args": [],
      "env": {}
    }
  }
}
```

Compare that to the v6e-1 version, which inlined seven environment variables into the JSON. Every one of those was a place the config could drift.

(That absolute path is worth a second look if you're copying this: it points into a *different* checkout tree than the one this article was written from. Absolute paths in MCP configs are exactly the kind of thing that survives a directory rename and then fails silently at launch. `mcp-run.sh` exists to keep everything *else* out of this file — but the path to `mcp-run.sh` itself is still a hardcoded string.)

#### Gotcha #1: v5e is spelled `v5litepod`

Before anything else. gcloud does not know what a "v5e-1" is.

- Accelerator type: `v5litepod-1`, **not** `v5e-1`
- Flex-start runtime version: `v2-alpha-tpuv5-lite`
- Machine type (if you go the GCE route): `ct5lp-hightpu-1t`, **not** `ct6e-standard-1t`

"v5e-1" is fine in prose. It will never work in a gcloud argument. That last line matters more than it looks — `ct6e-standard-1t` is a *v6e* machine type, and if it shows up in your notes labelled as v5e, every memory number downstream of it is wrong by a factor of two. I found exactly that mislabel in this repo's own demo page while writing this article.

#### Gotcha #2: quota is not availability, and availability is not the provisioning model

This is the one that cost the most time.

The MCP agent has a `get_zones_with_available_quota` tool that scans `TPUV5sLitepodPerProjectPerZoneForTPUAPI` across every zone:

```plaintext
> get_zones_with_available_quota
```

It came back with **44 zones with non-zero quota**. Every single one. That number is useless, and here's why: quota only means *creation is permitted*. It says nothing about whether capacity exists, and — critically — nothing about whether the **provisioning model** you asked for is supported there.

Flex-start `v5litepod-1` is a much narrower thing than the quota table suggests:

```plaintext
> create a v5litepod-1 queued resource in europe-west4-a

  ERROR: FLEX_START provisioning model is not supported for accelerator type
  "v5litepod-1" in location "europe-west4-a"
```

Same rejection in `europe-west4-b`. Accepted in `us-west4-a`. Verified by attempting creation in each — there is no API that will tell you this in advance.

So the default zone this project shipped with (`europe-west4-a`, inherited from the v6e rig) **could never provision this rig**, in any amount of retrying. Worth noting: the reference documentation lists `europe-west4-b` as flex-start-capable for v5e, but its example uses `v5litepod-4`. The single-chip shape is narrower than the table implies.

#### Gotcha #3: a screenful of failed zones usually means a broken flag, not a full region

The agent keeps mutable state in `tpu_zones_status.md` — `find_tpu` writes failures into it and reads it back to skip known-bad zones. At one point it looked like this:

```plaintext
| Zone | Quota Available | TPU v5e-1 Started Successfully | Details |
| europe-west4-a | Yes | No | creation failed |
| europe-west4-b | Yes | No | creation failed |
| europe-west1-b | Yes | No | creation failed |
| ... 40 more rows ...
```

Forty-odd zones "out of capacity" is not a capacity story. It's a flag story. Every one of those attempts was passing `--network=vpc-glitnir` — a VPC that does not exist in this project. `aisprint-491218` has only the auto-mode `default` network.

The fix was to let `TPU_NETWORK` / `TPU_SUBNETWORK` default to empty so gcloud uses the project default, then reset the table and reseed it from a live quota scan. The lesson generalizes: **if a resource fails in every zone, it isn't the zones.**

Note also that the file is *mutable state, not documentation*. Don't hand-edit it as if it were docs — `find_tpu` will overwrite you.

#### Deploying

With the zone sorted, the deploy is a single MCP call. The generated gcloud command:

```shell
gcloud alpha compute tpus queued-resources create vllm-gemma4-qr \
  --node-id=vllm-gemma4-qr-node \
  --project=aisprint-491218 \
  --zone=us-west4-a \
  --accelerator-type=v5litepod-1 \
  --runtime-version=v2-alpha-tpuv5-lite \
  --provisioning-model=flex-start \
  --max-run-duration=4h \
  --valid-until-duration=4h \
  --labels=purpose=flex-start \
  --metadata-from-file=startup-script=startup_script.sh
```

Those two duration flags are a **local policy choice, not a platform limit**. Flex-start itself allows up to **seven days** (`maxRunDurationSeconds`, defaulting to the full seven). This rig caps at 4h deliberately: it's a demo box, and an auto-expiring TPU is cheaper than a remembered one. `--valid-until-duration=4h` bounds the other end — how long the request sits in `WAITING_FOR_RESOURCES` before giving up rather than queuing indefinitely.

The startup script pulls `vllm/vllm-tpu:nightly` and serves:

```shell
vllm serve google/gemma-4-E2B-it \
  --max-model-len 16384 \
  --tensor-parallel-size 1 \
  --disable_chunked_mm_input \
  --max_num_batched_tokens 4096 \
  --limit-mm-per-prompt '{"image":4,"audio":1}' \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4
```

`--tensor-parallel-size` is **1**. v5e-1 is a single chip. If you see `4` anywhere in a v5e-1 config, it's copy-paste from a larger topology and the engine will fail to initialize.

Two structural details worth stealing:

**The startup script fetches the HF token itself.** The rendered script is uploaded as *instance metadata*, so a baked-in token would be readable from the instance by anyone with `compute.instances.get`. Instead it reads `hf-token` from Secret Manager at boot via the metadata server, retrying for 30 minutes so an IAM grant applied *after* creation still lands. Shell tracing is off across the entire token section.

**Serving flags live in exactly one function.** `_vllm_serve_flags()` builds the vLLM arg list from `MAX_MODEL_LEN`, `MAX_NUM_BATCHED_TOKENS`, `LIMIT_MM_PER_PROMPT`, and `TENSOR_PARALLEL_SIZE`. The startup script template takes the same values as placeholders. Both deploy paths and the generated one-liner therefore cannot disagree.

That template is consumed by `str.format()`, incidentally, which means **any literal `{` or `}` you add to that bash file breaks the deploy at format time** — a shell brace expansion, a `${VAR}`, a JSON literal. Escape as `{{` / `}}`. The `--limit-mm-per-prompt` JSON above is exactly the trap.

#### Gotcha #4: the memory math is tighter than you think

Here is where the 16 GB starts to bite, and it's not obvious from the parameter count.

Gemma 4 E2B is a "2B" model, but the resident weight footprint under vLLM is **~8.97 GiB**, not the ~4.5 GiB you'd naively compute from 2B parameters at bfloat16. The multimodal towers are not free.

On a v5e chip that arithmetic goes:

```plaintext
Physical HBM                    16 GB   (~15.5 GiB)
vLLM utilization cap (0.9)      ~13.9 GiB
Resident model weights          ~8.97 GiB
─────────────────────────────────────────
Available for KV cache          ~5.0 GiB
```

Versus the same model on a v6e chip, which has ~19.8 GiB left for KV — **roughly 4× the cache pool for a 2.25× price**.

At the observed ~18 KiB/token KV footprint (15 layers materialize a cache, bfloat16), ~5.0 GiB works out to roughly **290,000 tokens**, or about **17× concurrent requests** at the configured 16,384-token context.

*Caveat, stated plainly:* the ~5.0 GiB and ~290K figures are arithmetic derived from the measured weight footprint and the published chip capacity, not read out of a v5e engine log. The 8.97 GiB weights and 18 KiB/token are measured. Treat the KV numbers as a good estimate, and confirm against your own `vllm` init log before you size a fleet on them.

The 17× concurrency ceiling is not theoretical. It's exactly what falls over in the sweep.

#### Measured: the concurrency × context sweep

Run with `vllm bench serve` inside the container on the TPU VM, 128 output tokens per request, 20 grid points. **Zero failures across the whole grid** — worth noting, because the v6e-1 grid on the same model had 27 of 156 points fail or get skipped.

| Conc | Context | req/s | Output tok/s | Total tok/s | Mean TTFT | p99 TTFT | TPOT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 0.99 | 126.9 | 254 | 22.5 ms | 75 ms | 7.76 ms |
| 1 | 1,024 | 0.96 | 123.4 | 1,110 | 48.4 ms | 105 ms | 7.79 ms |
| 1 | 4,096 | 0.87 | 110.8 | 3,655 | 149.5 ms | 208 ms | 7.92 ms |
| 1 | 8,192 | 0.83 | 105.8 | 6,875 | 174.5 ms | 180 ms | 8.15 ms |
| 1 | 15,000 | 0.70 | 90.1 | 10,643 | 351.2 ms | 358 ms | 8.42 ms |
| 4 | 128 | 3.38 | 432.5 | 865 | 55.7 ms | 94 ms | 8.84 ms |
| 4 | 1,024 | 3.40 | 435.1 | 3,916 | 32.4 ms | 42 ms | 8.97 ms |
| 4 | 4,096 | 3.12 | 399.7 | 13,190 | 61.5 ms | 100 ms | 9.47 ms |
| 4 | 8,192 | 2.78 | 355.8 | 23,126 | 92.9 ms | 169 ms | 10.41 ms |
| 4 | 15,000 | 1.45 | 185.3 | 21,904 | 615.0 ms | 2,616 ms | 13.23 ms |
| 16 | 128 | 8.41 | 1,076.9 | 2,154 | 169.5 ms | 289 ms | 13.59 ms |
| 16 | 1,024 | 7.33 | 938.8 | 8,449 | 198.5 ms | 352 ms | 15.53 ms |
| 16 | 4,096 | 4.79 | 612.8 | 20,221 | 365.8 ms | 974 ms | 23.10 ms |
| 16 | 8,192 | 3.40 | 434.8 | 28,260 | 983.1 ms | 2,207 ms | 29.20 ms |
| 16 | 15,000 | 1.32 | 169.3 | 20,003 | 3,319 ms | 8,220 ms | 68.73 ms |
| 64 | 128 | **14.67** | **1,877.7** | 3,755 | 341.7 ms | 398 ms | 31.52 ms |
| 64 | 1,024 | 11.52 | 1,474.9 | 13,274 | 686.5 ms | 1,347 ms | 37.68 ms |
| 64 | 4,096 | 6.43 | 822.9 | 27,154 | 2,505 ms | 5,277 ms | 55.99 ms |
| 64 | 8,192 | 2.66 | 340.3 | 22,119 | 3,077 ms | 6,870 ms | 69.24 ms |
| 64 | 15,000 | 1.59 | 203.1 | 24,003 | 7,755 ms | 16,516 ms | 72.96 ms |

Peak decode: **1,877.7 output tok/s** at 64 concurrent, short context.
Peak aggregate: **28,260 total tok/s** at 16 concurrent, 8K context — that's prefill throughput, and it's the number to quote if your workload is RAG-shaped.

#### Reading the sweep

**Single-stream decode is 90–127 tok/s and remarkably flat.** TPOT moves from 7.76 ms to 8.42 ms as context goes from 128 to 15,000 tokens — an 8% degradation across a 117× context increase. Decode on a 2B model is bandwidth-bound, and attention over even 15K tokens barely registers against the weight streaming cost. This is the number that matters for an interactive agent: **a single user gets a consistent ~120 tok/s regardless of how much context you stuff in.**

**Batching scales cleanly up to about 4K context, then stops.** Speedup over single-stream, by context:

| Context | 4 conc | 16 conc | 64 conc |
| ---: | ---: | ---: | ---: |
| 128 | 3.41× | 8.49× | **14.80×** |
| 1,024 | 3.53× | 7.61× | 11.96× |
| 4,096 | 3.61× | 5.53× | 7.43× |
| 8,192 | 3.36× | 4.11× | **3.22×** |
| 15,000 | 2.06× | 1.88× | 2.26× |

Look at the 8,192 row. Going from 16 concurrent to 64 concurrent makes throughput **go down** — 434.8 tok/s to 340.3 tok/s. That is the ~5 GiB KV pool saturating. 64 requests × 8,192 tokens is ~524K tokens of KV demand against a pool that holds ~290K. vLLM starts preempting and recomputing, and you pay for the same prefill twice.

**The 15,000-token column is a wall, not a slope.** At concurrency 1 it's fine — 90 tok/s, 351 ms TTFT. At concurrency 16, p99 TTFT is **8.2 seconds**. At 64, it's **16.5 seconds**. Nothing failed, which is the good news; vLLM queued rather than OOM'd. But a 16-second p99 is a broken user experience.

**The practical envelope for this rig:**

| Workload | Recommended concurrency | What you get |
| --- | --- | --- |
| Chat / short prompts (≤1K) | 64 | ~14.7 req/s, sub-700 ms TTFT |
| Tool-calling agent (≤4K) | 16 | ~4.8 req/s, ~366 ms TTFT, p99 under 1 s |
| RAG / long docs (8K) | 16 | 28,260 tok/s aggregate, ~1 s TTFT |
| Max context (15K) | 1–4 | Anything above 4 has an unusable p99 |

![](sweep_lines_v5e1.png)

![](sweep_heatmap_v5e1.png)

#### Cost

Now the part the sweep pays for.

First, an honesty note about the flex-start column, because I got this wrong on my first pass. **Google does not publish a per-chip-hour flex-start rate.** Flex-start is billed under [Dynamic Workload Scheduler pricing](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/dws), described only as *"discounted (up to 53%) for vCPUs, GPUs, and TPUs."* So every flex number below is **on-demand list minus an assumed 50%**, which is the conservative end of "up to 53%". Treat them as upper bounds on the flex price, and therefore the cost-per-token figures as upper bounds too.

On-demand rates are list, us-west4:

| Configuration | Flex-start (derived) | On-demand (list) |
| --- | ---: | ---: |
| TPU v5e-1 (16 GB HBM) | ~$0.60 / hr | **$1.20 / hr** |
| TPU v6e-1 (32 GB HBM) | ~$1.35 / hr | **$2.70 / hr** |
| TPU v6e-4 (128 GB HBM) | ~$5.40 / hr | ~$10.80 / hr |
| GCE 1× NVIDIA L4 (24 GB) | ~$0.30 / hr (spot) | ~$1.00 / hr |
| GCE 8× NVIDIA L4 (192 GB) | ~$2.40 / hr (spot) | ~$8.00 / hr |
| Cloud Run 1× L4 (serverless) | — | ~$1.20–1.50 / hr active |
| AWS EC2 `g6.xlarge` (1× L4) | ~$0.35 / hr (spot) | ~$0.97 / hr |
| Azure `NVadsA10v5` (1× A10G) | ~$0.35 / hr (spot) | ~$1.05 / hr |

Measured from the Cloud Billing Catalog for `us-west4` (Las Vegas) on 2026-08-06, per chip-hour:

| Provisioning model | SKU | Rate |
| --- | --- | --- |
| On-demand | `TpuV5e running in Las Vegas` | $1.2000 |
| Flex-start | `DWS Defined Duration V5e running in Las Vegas` | $0.6000 |
| Spot | `TpuV5e attached to Spot Preemptible VMs running in Las Vegas` | $0.5779 |
| Reserved (calendar) | `Reserved V5e TPU in Las Vegas in Calendar Mode` | $0.8400 |
| 1yr / 3yr commitment | `Commitment v1: TpuV5e …` | $0.8400 / $0.5400 |

Spot lands just under flex-start — a 3.7% gap, not the large discount its reputation suggests — and buys that with preemption at 30 seconds' notice instead of flex-start's bounded run duration. Note the catalog calls flex-start "DWS Defined Duration" (Dynamic Workload Scheduler).

Rates vary by region and change, so don't quote the table above as current: `estimate_deployment_cost` in `server.py` now reads these live from the Billing Catalog API and reports nothing at all when a SKU is missing. It used to carry a hardcoded `"v5e": 0.12`, off by 10x against the $1.20 list rate — which is the argument against hardcoded rate tables generally, including any transcribed from here.

A published price is not an offer of capacity: `europe-west4` quotes a flex-start v5e rate while the API still refuses to provision `v5litepod-1` there.

**Cost per million output tokens**, computed directly from the measured sweep:

| Concurrency | Context | Output tok/s | Flex ($0.60/hr) | On-demand ($1.20/hr) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 126.9 | $1.314 / M | $2.627 / M |
| 4 | 128 | 432.5 | $0.385 / M | $0.771 / M |
| 16 | 128 | 1,076.9 | $0.155 / M | $0.310 / M |
| 64 | 128 | 1,877.7 | **$0.089 / M** | $0.178 / M |
| 16 | 1,024 | 938.8 | $0.178 / M | $0.355 / M |
| 64 | 1,024 | 1,474.9 | $0.113 / M | $0.226 / M |
| 16 | 4,096 | 612.8 | $0.272 / M | $0.544 / M |
| 64 | 4,096 | 822.9 | $0.203 / M | $0.405 / M |

The headline: **$0.089 per million output tokens** at peak batching on flex-start.

The more useful headline is the spread. The *same chip*, *same model*, *same hour of billing*, run at concurrency 1 instead of 64, costs **14.8× more per token**. Idle TPU is the entire cost story. An agent rig that sits at concurrency 1 is paying $1.31/M for tokens it could be getting at $0.09/M.

Which leads to the real cost question for a devops agent: your workload probably *is* concurrency 1. An SRE assistant answering one operator's question is a single stream, and no amount of tuning turns one operator into a batch of 64.

In that regime **the cost control is run duration, not batching.** This is why the rig sets `--max-run-duration=4h` rather than accepting the seven-day flex-start default: an agent box that auto-expires costs a bounded amount, and a forgotten one costs $0.60/hr forever while serving nobody. Provision it for the session, let it expire. Do not leave a TPU idling at concurrency 1 and then compare $/M against a hosted API — you will lose that comparison badly, and you will deserve to.

The counterweight, and the reason not to set the cap too aggressively: re-provisioning is not instant. Flex-start requests sit in `WAITING_FOR_RESOURCES` until capacity appears, with **no documented SLA** on how long that takes. On this project it has run to a couple of hours. A 4-hour cap on a demo box is a reasonable trade; a 1-hour cap on something you actually need would not be.

#### Comparing 2B serving: v5e-1 vs v6e-1

Here I have to be careful, and I'd rather flag it than quietly paper over it.

The v5e-1 numbers above come from `vllm bench serve` with a **128-token output** per request. The v6e-1 Gemma 4 2B grid in this repo came from a different harness with a much **shorter output window**. Request-per-second figures between those two runs are **not directly comparable** — a shorter output inflates req/s roughly linearly, and the v6e grid's peak of ~140 req/s is largely measuring prefill-and-stop, not sustained generation.

So the honest comparison is on axes where the measurement method doesn't dominate:

| Dimension | v5e-1 (measured) | v6e-1 | Verdict |
| --- | --- | --- | --- |
| Resident weights (E2B) | ~8.97 GiB | ~8.97 GiB | Model property, identical |
| HBM left for KV | **~5.0 GiB** (derived) | ~19.8 GiB (measured) | v6e ~4× the pool |
| KV capacity | ~290K tok (derived) | 1,151,744 tok (measured) | v6e ~4× |
| Sustained decode, 1 stream | 90–127 tok/s | ~214 tok/s | v6e ~1.7–1.8× |
| TTFT, 1 stream, short ctx | 22.5 ms (on-host) | ~210 ms (over network) | Not comparable — different vantage points |
| Grid failures | **0 / 20** | 27 / 156 | v5e ran the whole grid |
| Max usable context @ conc 16 | ~8K | 65K configured | v6e, decisively |
| Price (flex) | ~$0.60/hr | ~$1.35/hr | v5e 2.25× cheaper |

**Single-stream decode: v6e is roughly 1.7–1.8× faster** (~214 vs ~120 tok/s), for 2.25× the price. On a per-dollar basis for a single interactive user, **v5e is ahead**. That tracks the hardware — decode is bandwidth-bound, v6e has exactly 2× the bandwidth, and it converts about 85% of that into real tokens.

**Memory is where v6e earns its premium, not compute.** The 4× KV pool is the difference between a model that batches 8K context at 16 concurrent and one that starts preempting. If your workload is short-context chat, you are paying for headroom you never touch. If it's RAG over long documents at any real concurrency, v5e's ~5 GiB pool is the binding constraint and no amount of tuning fixes it.

**The FLOPS gap barely shows up.** v6e has 4.7× the BF16 throughput and delivers ~1.8× the decode tokens. That gap is the clearest possible signal that serving a 2B model is a memory-bandwidth problem, not a matrix-multiply problem. It also means the v6e-4 topology — 4 chips, 128 GB — is badly over-provisioned for a 2B model; the earlier v6e-4 sweep found the 2B model became bounded by *host CPU dispatch overhead*, with the TPU processing tokens faster than the host could schedule them.

#### Which chip for the 2B model?

| If your workload is... | Use | Because |
| --- | --- | --- |
| Single-agent SRE assistant, one operator | **v5e-1 flex** | Best tokens-per-dollar at concurrency 1; $0.60/hr |
| Chat serving, short context, high concurrency | **v5e-1 flex** | 14.7 req/s at $0.089/M output tokens |
| Tool-calling agent, ≤4K context, ≤16 concurrent | **v5e-1 flex** | Comfortably inside the KV envelope, p99 under 1 s |
| RAG over 8K+ documents at concurrency >16 | **v6e-1** | v5e's ~5 GiB KV pool preempts; throughput inverts |
| Long context (15K+) at any real concurrency | **v6e-1** | v5e p99 TTFT hits 8–16 s |
| Anything 2B at 4-chip scale | **Neither** | Host-bound; spend the money on a bigger model instead |

The summary in one line: **for Gemma 4 2B, v5e-1 is the right default and v6e-1 is a memory upgrade, not a speed upgrade.**

#### Verifying the deployment

Once the queued resource goes ACTIVE, endpoint discovery is dynamic — `discover_vllm_url()` lists queued resources in the zone, takes the first `ACTIVE` one, resolves its node and external IP, and builds `http://{ip}:8000`. Never hardcode it; the IP changes every time the resource is recreated.

```plaintext
> verify_model_health

  The health check passed successfully!

  • Status: PASSED ✅
  • Response: "Hello! Yes, the model is working. I..."
  • Latency: 0.31 seconds
```

```plaintext
> query_queued_gemma4_with_stats what is a TPU?

  ### 📊 Performance Stats

  • Time to First Token (TTFT): 0.022s
  • Throughput: 126.9 tokens/s
  • Mean TPOT: 7.76 ms
```

One last trap: **raw `/v1/completions` returns an empty completion on `-it` models.** `make query` and `benchmarking_suite.py` both use it, so an empty result there is expected, not a broken deploy. `server.py` uses `/v1/chat/completions` throughout. Keep new code on the chat endpoint; raw completions are only useful for prefill-only benchmarks.

#### The tools

```plaintext
> make tools
31 tools registered.
```

`GemmaTools.md` and the `get_help` tool both build their tool list from `mcp.list_tools()`, so they cannot drift from the `@mcp.tool()` decorators. Add a tool, run `make tools`, the doc regenerates. Source of truth either way is `grep -n "^@mcp.tool" server.py`.

One split worth preserving if you fork this: `create_tpu_queued_resource` is **non-destructive** — it touches only the id it was given, which is what makes `find_tpu`'s zone sweep safe to run. `manage_queued_resource` is **destructive** — it deletes every queued resource in the zone that isn't the named primary. Two verbs, two blast radii, no overlap.

And: don't destroy a queued resource unless you mean it. Teardown is not part of routine debugging, and there is no documented SLA on how long a flex-start request waits for capacity — on this project it has run to a couple of hours.

#### Summary

A single TPU v5e chip serves Gemma 4 2B at **90–127 tok/s single-stream** and **1,878 tok/s batched**, for **$0.60/hr flex-start** — about **$0.089 per million output tokens** at peak batching, or **$1.31/M** if you run it at concurrency 1 like most agent workloads actually do.

Against v6e-1 at 2.25× the price, v5e gives up roughly **1.8× decode speed** and **4× KV cache pool**. The decode gap is a bandwidth story and v5e wins on a per-dollar basis. The KV gap is real and unfixable, and it is the actual reason to reach for v6e: not because v6e is faster, but because 16 GB runs out.

The three things that cost the most time were all configuration, not capacity: flex-start `v5litepod-1` is only accepted in `us-west4-a`; a nonexistent VPC in the create flags looked exactly like a global capacity shortage; and `ct6e-standard-1t` in a note labelled "v5e" quietly doubled every memory number downstream. Quota is not availability, availability is not the provisioning model, and a resource that fails in every zone is never the zones' fault.

#### Further reading

- [Debugging deployments with Gemma 4 4B, TPU v6e-1, MCP and Antigravity CLI](https://xbill999.medium.com/debugging-deployments-with-gemma-4b-tpu-v6e-1-mcp-and-antigravity-cli-c9846231237a) — the v6e-1 predecessor to this article, and the source of the v6e numbers used above.
- [TPU v5e vs T4 GPU: Best Budget AI Accelerator for 2026](https://deploybase.ai/articles/v5e-1-tpu-vs-t4-gpu) — the other budget comparison, against NVIDIA's T4 rather than against a bigger TPU. Its throughput and $/M-token figures come from a different model and harness than anything measured here, so treat them as directional, not as a like-for-like extension of the tables above.
- [Cloud TPU v5e documentation](https://docs.cloud.google.com/tpu/docs/v5e) and [Dynamic Workload Scheduler pricing](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/dws) — the sources for the spec table and the flex-start discount.
