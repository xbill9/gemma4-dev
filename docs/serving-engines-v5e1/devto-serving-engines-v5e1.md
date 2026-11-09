---
title: "Six Serving Engines on One Cloud TPU v5e Chip: Gemma 4 E2B"
published: false
description: "vLLM, SGLang-JAX, a pure-JAX engine, a Rust model over PJRT, and TorchTPU in Python and Rust serve Gemma 4 E2B on one flex-start v5e-1, with the same prompts, the same 1-to-64-request sweep and the same GSM8K questions. vLLM is the engine that serves out of the box; SGLang-JAX with an E2B patch is the fastest at every load from 8 requests up."
tags: googlecloud, tpu, vllm, jax
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/docs/serving-engines-v5e1/devto-cover.7c2d5345.jpg
---

This article provides a step by step comparison of six inference serving engines on one Cloud TPU v5e chip: vLLM, SGLang-JAX, a pure-JAX engine, a Rust model of Gemma 4 over PJRT, and TorchTPU in Python and in Rust, each serving Gemma 4 E2B from a flex-start `v5litepod-1`.

https://github.com/xbill9/gemma4-dev

The question is which engine to put on a v5e-1 when one small model has the chip to itself. Each engine gets the same six chat prompts, the same 1-to-64-request load sweep where it can batch, and the same 1,319 GSM8K questions. Every engine ran alone on its own v5e-1, of the same type and zone, and each VM ran its jobs unattended and deleted itself when done.

---

#### The Results in Brief

Answer quality is level across the engines that finished GSM8K: 0.8969 to 0.9143, and every engine's paired difference against vLLM includes zero except SGLang-JAX's, which is ahead.

For one request, four engines are within 12% of each other: SGLang-JAX 159.9 tokens per second end to end, the Rust model over PJRT 145.2, vLLM 143.9 and the JAX engine 134.3. TorchTPU runs at 6.6 (Python) and 13.4 (Rust).

Under load two engines batch. vLLM on its int4 build reaches 1,689.1 tokens per second at 64 requests of 128 tokens in and 512 out. SGLang-JAX, with a patch that adds what E2B needs, reaches 4,249, 2.52x vLLM, and leads from 8 requests up on both shapes. The JAX, Rust and TorchTPU servers serve one request at a time.

vLLM is the engine that serves Gemma 4 E2B on a v5e-1 as shipped, with tool calling and the int4 checkpoint. SGLang-JAX is the fastest, once patched.

---

#### What Changes and What Stays

| Engine | Version | Weights | Installed with |
| :--- | :--- | :--- | :--- |
| vLLM | `0.29.1rc1.dev468+g0b7f11a1e` | W4A16, int4 tables | the pinned `vllm-tpu` image, patched |
| SGLang-JAX | `b3e3e16b2` | bf16 | pip, Python 3.13, one E2B patch |
| JAX engine | this repo's | bf16, and W4A16 | pip `jax[tpu]==0.12.0.1` |
| Rust model over PJRT | this repo's | bf16 | Rust nightly, PJRT |
| TorchTPU, Python | 0.2.0 | bf16 | pip, Python 3.14 |
| TorchTPU, Rust | 0.2.0 | W4A16, int4 tables | PyO3 crate built on the VM |

The same for all: one TPU v5e chip (`v5litepod-1`, 15.75 GiB of HBM as vLLM reports it) on a flex-start queued resource in `us-west4-a`, runtime `v2-alpha-tpuv5-lite`, Gemma 4 E2B, greedy decoding, thinking off, and the system Python with no virtual environment. vLLM's checkpoint is `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4`: Google's QAT 4-bit values with the vocabulary tables in int4 too. The bf16 engines serve `google/gemma-4-E2B-it`.

llama.cpp is absent because it has no TPU backend: its `ggml/src` lists blas, cann, cpu, cuda, et, hexagon, hip, metal, musa, opencl, openvino, rpc, sycl, virtgpu, vulkan, webgpu, zdnn and zendnn.

---

#### At This Point You Should Have…

- A Google Cloud project with TPU v5e quota in `us-west4-a`
- A Cloud Storage bucket for results
- `gcloud` with the `alpha` component
- A clone: `git clone https://github.com/xbill9/gemma4-dev`

---

#### Step 1 — Get a v5e-1 on Flex-Start

Flex-start `v5litepod-1` is accepted in `us-west4-a`. Each engine got its own queued resource, created like this:

```text
gcloud alpha compute tpus queued-resources create tpu-sglangjax-v5e1-2b \
  --node-id tpu-sglangjax-v5e1-2b-node --zone us-west4-a --project aisprint-491218 \
  --accelerator-type v5litepod-1 --runtime-version v2-alpha-tpuv5-lite \
  --provisioning-model=flex-start --max-run-duration=24h --valid-until-duration=2h --async
```

The node's chip, as vLLM sees it:

```text
INFO 11-06 21:46:48 [__init__.py:75] TPU info: node_name=tpu-vllm-v5e1-2b-q4w4a16emb4-node | tpu_type=v5litepod-1 | worker_id=0 | num_chips=1 | num_cores_per_chip=1
(EngineCore pid=187) INFO 11-06 21:51:03 [tpu_runner.py:1300] Init model | hbm=[(2.84, 15.75)]GiB
```

The int4 build takes 2.84 GiB of the chip's 15.75 GiB.

🔎 **Tip: `--max-run-duration=24h` is the backstop.** A flex-start VM is deleted at the end of its run duration whatever it is doing, so a run that hangs costs at most a day.

---

#### Step 2 — One Chip per Engine, Unattended

`flight/ladder.py` runs each engine's jobs against one server start: `one` (six prompts), `sweep` (1 to 64 requests) and `acc` (GSM8K, and BFCL where the server returns tool calls). It runs as a systemd unit, and `finish.sh` waits for it, uploads the run directory, reads the upload back, compares SHA-256 and deletes the queued resource:

```text
2026-11-09 02:59:59 waiting for none
2026-11-09 03:00:00 none ended: success
2026-11-09 03:00:08 uploaded gs://aisprint-491218-ladder-v5e1/2026-11-08-first-light-v5e1/tpu-sglangjax-v5e1-2b/ladder-run.tgz sha256 2c35d88e65a7b8190f90ba1cc6b44638c013d047d1376ef94d054cd159a6a57c (29 entries)
2026-11-09 03:00:08 deleting queued resource tpu-sglangjax-v5e1-2b
2026-11-09 03:00:10 delete requested (rc=0)
```

Four VMs carried the six engines: one for vLLM, one for TorchTPU in Python and the bf16 JAX engine, one for TorchTPU in Rust, the Rust model and the int4 JAX engine, and one for SGLang-JAX.

---

#### Step 3 — What Each Engine Needs on a v5e

- **vLLM** serves from its pinned TPU image with this repo's patches, which add an int4 method for the vocabulary tables and an int4 `lm_head` to `tpu_inference`. It reserves 80% of the chip and holds 568,256 tokens of cache at a 16,384-token context.
- **SGLang-JAX** implements the Gemma 4 31B and 26B A4B decoders only. Gemma 4 E2B adds per-layer embeddings, a double-wide MLP on its last 20 layers and K/V shared from earlier layers, and its config leaves `num_global_key_value_heads` null. A 190-line patch adds all four; a separate article covers it.
- **The JAX engine and the Rust model** serve one request at a time at a 4,096-token context.
- **TorchTPU in Python** serves bf16. Its int4 path does not compile: `torch.compile` fails tracing compressed-tensors' int4 unpack (`index_put_(): cannot record EventSnapshot during an FX trace`).

🔎 **Tip: SGLang-JAX needs `--page-size 128` for E2B.** At the default page size of 1 its FlashAttention backend caps the server at 8 running requests at a 16,384-token context; at 128 it runs 64.

---

#### Step 4 — Start-Up Time

Seconds from launch to the first answer:

| Engine | Start s |
| :--- | :--- |
| Rust model over PJRT | 🥇 22 |
| TorchTPU, Python | 31 |
| JAX engine, bf16 | 42 |
| TorchTPU, Rust | 45 |
| JAX engine, W4A16 | 46 |
| SGLang-JAX | 270 |
| vLLM, int4 tables | 2,055 |

vLLM compiles every request shape before it serves, for 31 to 34 minutes on its two builds. SGLang-JAX's 270 s came with a compile cache already on the VM; on a fresh VM its 512-wide attention kernel compiles each prefill shape for about five minutes:

```text
[EXTEND] PRECOMPILE:  14%|█▍| 1/7 [05:06<30:36, 306.02s/it, bs=8, tokens=64]
[EXTEND] PRECOMPILE:  29%|██▊| 2/7 [10:08<25:19, 303.90s/it, bs=8, tokens=128]
```

🔎 **Tip: give SGLang-JAX `--watchdog-timeout 7200` and a JAX compile cache.** Its default 300 s watchdog stops the scheduler during a compile that long, and `JAX_COMPILATION_CACHE_DIR` makes a second start fast.

---

#### Step 5 — One Request at a Time

Six chat prompts, 256 tokens each, three repeats; the geometric mean of each prompt's median, end to end. Then 4 requests at once:

```text
arm            e2e tok/s  decode TTFT ms 4 at once    x1 repeat start s x vllm
vllm               143.9   145.1    15.4     477.2  3.32    6/6    2055  1.000
vllm-text          134.2   135.2    16.2     449.8  3.35    6/6    1860  0.932
jax                104.1       -    15.9     122.1  1.17    6/6      46  0.723
jax-bf16           134.3   131.7    11.3     150.1  1.12    6/6      42  0.933
jaxrust            145.2   145.5     9.4     145.4  1.00    6/6      22  1.009
torchtpu-bf16        6.6       -   153.6       6.7  1.00    6/6      31  0.046
torchtpu-rust       13.4    13.4    81.3      13.6  1.02    6/6      45  0.093
```

```text
arm            e2e  decode  ttft ms 4 at once  x vllm repeat
sglang       159.9   161.7     20.2     631.5   1.111 6/6
```

| Engine | 1 request | 4 at once | x vLLM |
| :--- | :--- | :--- | :--- |
| SGLang-JAX | 🥇 159.9 | 🥇 631.5 | 1.11x |
| Rust model over PJRT | 145.2 | 145.4 | 1.009x |
| vLLM | 143.9 | 477.2 | 1.000x |
| JAX engine, bf16 | 134.3 | 150.1 | 0.933x |
| JAX engine, W4A16 | 104.1 | 122.1 | 0.723x |
| TorchTPU, Rust | 13.4 | 13.6 | 0.093x |
| TorchTPU, Python | 6.6 | 6.7 | 0.046x |

The Rust model has the shortest time to first token, 9.4 ms. Every engine repeats its own greedy text on all six prompts.

---

#### Step 6 — 1 to 64 Requests

`sweep.py` runs two shapes, 512 tokens in / 128 out and 128 in / 512 out, at 1 to 64 requests at once, five repeats, a unique prompt per request, output timed from the stream. Aggregate output tokens per second, 128 in / 512 out:

| Requests | SGLang-JAX | vLLM | vLLM, bf16 tables |
| :--- | :--- | :--- | :--- |
| 1 | 🥇 160.3 | 144.5 | 134.4 |
| 8 | 🥇 943.1 | 778.4 | 737.3 |
| 16 | 🥇 1,671.9 | 1,112.8 | 1,084.7 |
| 32 | 🥇 2,550.2 | 1,462.5 | 1,424.9 |
| 64 | 🥇 4,249.4 | 1,689.1 | 1,660.1 |

512 in / 128 out:

| Requests | SGLang-JAX | vLLM | vLLM, bf16 tables |
| :--- | :--- | :--- | :--- |
| 1 | 🥇 154.2 | 140.6 | 131.3 |
| 8 | 🥇 915.6 | 682.2 | 652.5 |
| 16 | 🥇 1,663.0 | 937.9 | 906.8 |
| 32 | 🥇 2,358.4 | 1,156.8 | 1,125.9 |
| 64 | 🥇 2,758.9 | 1,288.4 | 1,266.0 |

SGLang-JAX is 2.52x vLLM at 64 requests on long replies and 2.14x on long prompts. At 2 to 4 requests it trails on two of four levels: 117.4 against 257.4 at 2 requests of 512 in / 128 out. Its five repeats spread 49 to 205% from slowest to fastest, against at most 5.0% for vLLM, so the medians above are its typical figure and a single run can land well below them.

The int4 vocabulary tables are worth 1.07x to vLLM for one request and 1.02x at 64.

🔎 **Tip: one v5e chip against one NVIDIA L4.** Against vLLM on an EC2 G6 with the same checkpoint, the v5e-1 runs at 1.06x and 1.04x for one request and 0.81x and 0.46x at 64 requests (512 in / 128 out, 128 in / 512 out).

---

#### Step 7 — Answer Quality

Each engine answered all 1,319 GSM8K test questions (2,048-token limit), one request at a time, greedy, thinking off. Paired question by question against vLLM, with a 95% range from 2,000 resamples:

| Engine | GSM8K | Minus vLLM [95%] | Time s |
| :--- | :--- | :--- | :--- |
| SGLang-JAX, bf16 | 🥇 0.9143 | +0.0136 [+0.0023, +0.0258] | 3,748 |
| JAX engine, bf16 | 0.9128 | +0.0121 [−0.0008, +0.0250] | 3,946 |
| Rust model over PJRT | 0.9113 | +0.0106 [−0.0008, +0.0227] | 3,594 |
| vLLM | 0.9007 | — | 3,630 |
| vLLM, bf16 tables | 0.8992 | −0.0015 [−0.0099, +0.0068] | 3,948 |
| JAX engine, W4A16 | 0.8969 | −0.0038 [−0.0114, +0.0038] | 5,183 |

SGLang-JAX against the bf16 JAX engine it was ported from is +0.0015 [−0.0091, +0.0121], level. The three bf16 engines lead the int4 builds by about one point, and only SGLang-JAX's lead has a range above zero.

BFCL simple (400 function-calling questions) scores 0.920 on both vLLM builds, with the same answer on all 400. The other engines return no tool calls, so BFCL ran on vLLM only.

TorchTPU's GSM8K runs were stopped: the Rust build answered at about 29 s a question.

---

#### Step 8 — What Does Not Fit on One Chip

vLLM with Google's `gemma-4-E2B-it-assistant` drafter at 4 tokens ran out of HBM during warmup at `--gpu-memory-utilization` 0.70 and again at 0.60:

```text
(EngineCore pid=291) ERROR 11-07 12:46:38 [core.py:1434] Error allocating device buffer: Attempting to allocate 512.00M. That was not possible. There are 450.17M free.; (0x0x0_HBM0)
```

Lowering the setting shrank the cache and left about the same memory free. A smaller request limit or a fixed cache size is untested.

---

#### Step 9 — Tear Down

Each VM deleted its own queued resource after its upload checked out. Only the bucket remains, and listing the zone shows no ladder VMs:

```text
gcloud alpha compute tpus queued-resources list --zone us-west4-a --project aisprint-491218
NAME              ZONE        NODE_COUNT  ACCELERATOR_TYPE  TYPE  TOPOLOGY  STATE
tpu-vllm-v5e1-2b  us-west4-a  1           v5litepod-1                       SUSPENDED
```

The one entry is an older rig's queued resource, unrelated to this run.

---

#### Compare and Contrast

| | vLLM | SGLang-JAX | Rust model over PJRT |
| :--- | :--- | :--- | :--- |
| 1 request | 143.9 | 🥇 159.9 | 145.2 |
| 64 requests, 128/512 | 1,689 | 🥇 4,249 | one at a time |
| GSM8K | 🟢 0.9007 | 🟢 0.9143 | 🟢 0.9113 |
| Tool calls (BFCL) | 🟢 0.920 | ❌ no parser | ❌ none |
| Start-up | ⚠️ 34 min | ⚠️ 5 min per shape fresh | 🥇 22 s |
| Runs as installed | 🟢 pinned image | ❌ E2B patch | 🟢 yes |
| Repeat-to-repeat spread | 🟢 ≤ 5% | ⚠️ up to 205% | 🟢 one request |

| | JAX engine | TorchTPU, Python | TorchTPU, Rust |
| :--- | :--- | :--- | :--- |
| 1 request | 134.3 | 6.6 | 13.4 |
| GSM8K | 🟢 0.9128 | ❌ stopped | ❌ stopped |
| int4 weights | 🟢 serves, 104.1 tok/s | ❌ does not compile | 🟢 serves |

---

#### So, Which One?

To serve Gemma 4 E2B on a v5e-1 today, run vLLM from its pinned TPU image on the int4 build: it serves 1 to 64 requests (1,689.1 tokens per second at 64), answers GSM8K level with every bf16 engine, returns tool calls, and needs nothing beyond the image and its patches. Plan for a 34-minute start. For the most throughput, SGLang-JAX with the E2B patch is 2.14 to 2.52x vLLM at 64 requests and the fastest for one request, at the cost of a patch, no tool calls and wide run-to-run spread. For one user who restarts often, the Rust model over PJRT starts in 22 s and matches vLLM for one request. TorchTPU is too slow on this chip to serve.

---

#### Summary

The goal of this article was to find the inference serving engine to run for Gemma 4 E2B on one Cloud TPU v5e chip. The key was holding everything else fixed: one chip per engine of the same type and zone, the same prompts, load sweep and GSM8K questions, unattended runs that delete their own VM. The six-engine results were:

- 🟢 Every engine that finished GSM8K scores 0.8969 to 0.9143; SGLang-JAX leads vLLM by 0.0136 with a range above zero.
- 🟢 vLLM serves as shipped from its pinned image, batches to 64 requests and is the only engine with tool calls (BFCL 0.920).
- 🟢 SGLang-JAX with the E2B patch is the fastest: 159.9 tokens per second for one request and 4,249 at 64, 2.52x vLLM.
- 🟢 The Rust model over PJRT starts in 22 s and is level with vLLM for one request.
- ⚠️ vLLM takes 31 to 34 minutes to start; SGLang-JAX about five minutes per prefill shape on a fresh VM.
- ⚠️ SGLang-JAX's repeats spread up to 205% at a level; vLLM's at most 5%.
- ❌ vLLM with Google's drafter runs out of HBM in warmup on one chip.
- ❌ TorchTPU runs at 6.6 and 13.4 tokens per second, and its int4 path does not compile in Python.

Scope: four flex-start `v5litepod-1` VMs in `us-west4-a`, one engine at a time on each, Gemma 4 E2B, greedy decoding, thinking off. vLLM serves the int4 build and SGLang-JAX and the JAX engines bf16, so the load comparison is engine and weights together; no bf16 vLLM sweep was run. The JAX servers ran at a 4,096-token context and vLLM and SGLang-JAX at 16,384. SGLang-JAX ran on 2026-11-09 on jax 0.11.1, the others on 2026-11-06 and 07 on jax 0.12.0.1. One-request speed is six prompts of 256 tokens, three repeats; load is two synthetic shapes at 1 to 64 requests, five repeats each; accuracy is one pass of GSM8K (1,319) and, for vLLM, BFCL simple (400). TPU memory in use and cost were not measured.

The strategy for comparing inference serving engines on Cloud TPU v5e was validated with an incremental step by step approach.

---

#### References

- [gemma4-dev on GitHub](https://github.com/xbill9/gemma4-dev)
- [The v5e-1 ladder run: 2026-11-06-ladder-v5e1](https://github.com/xbill9/gemma4-dev/blob/main/tpu-vllm-v5e1-2b-q4w4a16emb4/benchmarks/runs/2026-11-06-ladder-v5e1/REPORT.md)
- [SGLang-JAX first light: 2026-11-08-first-light-v5e1](https://github.com/xbill9/gemma4-dev/blob/main/tpu-sglangjax-v5e1-2b/benchmarks/runs/2026-11-08-first-light-v5e1/REPORT.md)
- [Six Serving Engines on One EC2 G6](https://dev.to/aws-builders/six-serving-engines-on-one-ec2-g6-gemma-4-e2b-on-an-nvidia-l4-3ff3)
- [Cloud TPU v5e](https://docs.cloud.google.com/tpu/docs/v5e)
- [vLLM](https://github.com/vllm-project/vllm)
- [tpu-inference](https://github.com/vllm-project/tpu-inference)
- [SGLang-JAX](https://github.com/sgl-project/sglang-jax)
- [JAX](https://github.com/jax-ml/jax)
- [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)
- [xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4)
- [GSM8K](https://github.com/openai/grade-school-math)
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
