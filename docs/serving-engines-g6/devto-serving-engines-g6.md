---
title: "Six Serving Engines on One EC2 G6: Gemma 4 E2B on an NVIDIA L4"
published: false
description: "llama.cpp, vLLM, SGLang, mistral.rs, a plain PyTorch engine and LiteRT-LM serve the same 4-bit Gemma 4 E2B on one g6.4xlarge, measured with the same prompts, the same 1-to-64-request sweep and the same GSM8K and BFCL questions. Answers are level on every engine; vLLM with Google's drafter is the fastest at every load, up to 4,516 output tokens per second."
tags: aws, ec2, vllm, nvidia
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/docs/serving-engines-g6/devto-cover.d8dbacaf.jpg
---

This article provides a step by step comparison of six inference serving engines on one Amazon EC2 G6 instance: llama.cpp, vLLM, SGLang, mistral.rs, a plain PyTorch engine and Google's LiteRT-LM, each serving Gemma 4 E2B on Google's QAT 4-bit weights from one NVIDIA L4.

https://github.com/xbill9/gemma4-dev

The question is which engine to put on a G6 when one small model has the whole GPU. Each engine gets the same weights, the same prompts, the same 1-to-64-request load sweep and the same 1,319 GSM8K and 400 BFCL questions, and each is measured with and without Google's E2B drafter wherever the engine supports one. Every engine ran on one instance, one at a time, with the card to itself.

---

#### The Results in Brief

Answer quality barely moves between engines. GSM8K lands between 0.8954 and 0.9014 on the five CUDA engines with the drafter on or off, and every paired difference against llama.cpp is within the run-to-run range. LiteRT-LM scores level on GSM8K and lower on BFCL.

For one request, five engines are within 8% of each other without a drafter (134.7 to 144.6 tokens per second end to end) and LiteRT-LM runs at 0.62 to 0.73x llama.cpp. The drafter decides the order: it gives vLLM 1.96x, llama.cpp 1.53x, SGLang 1.39x and LiteRT-LM 1.15x, and it slows mistral.rs to 0.95x.

Under load the batching engines separate. At 64 requests of 128 tokens in and 512 out, vLLM reaches 3,716 tokens per second without a drafter and 4,516 with it, SGLang 2,967, mistral.rs 1,825 and llama.cpp 1,056. PyTorch and LiteRT-LM serve one request at a time.

With each engine at its best setting, vLLM is the fastest at every level from 1 to 64 requests, on both shapes, and its drafter never loses. At the g6.4xlarge on-demand price that is $0.08 per million output tokens at full load.

---

#### What Changes and What Stays

| Engine | Version | Weights | Installed with |
| :--- | :--- | :--- | :--- |
| llama.cpp | `7182ef71e` | exact Q4_0 GGUF | CMake, CUDA 13.1, sm_89 |
| vLLM | 0.33.0 | W4A16, int4 tables | pip, its own `torch 2.14.0+cu130` |
| SGLang | 0.5.23 | W4A16, int4 tables | pip, cp313 wheel on Python 3.14, two patches |
| mistral.rs | v0.9.5 | exact Q4_0 GGUF | `cargo install`, CUDA 13.1, sm_89 |
| PyTorch | this repo's int4 engine | W4A16 | the DLAMI's torch 2.13 |
| LiteRT-LM | 0.19.0 | exact Q4_0, and Google's int2/4/8 file | pip, GPU through WebGPU on Vulkan |

The same for all six: one g6.4xlarge on-demand in us-east-1d (16 vCPUs, 64 GiB, one NVIDIA L4 with 23,034 MiB, driver 595.112.07), the PyTorch 2.13 / Ubuntu 26.04 Deep Learning AMI, Gemma 4 E2B on Google's QAT Q4_0 grid in every container, greedy decoding, and the system Python 3.14 with no virtual environment. The exact Q4_0 GGUF and the W4A16 checkpoint carry the same 4-bit values: `xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf` for llama.cpp and mistral.rs, `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4` for vLLM and SGLang.

---

#### At This Point You Should Have…

- An AWS account with G-family vCPU quota in us-east-1 and an instance profile with SSM access
- An S3 bucket for moving files to and from the instance
- The system `python3` with `boto3`, and the MCP server in `gpu-pytorch-g6-2b-q4w4a16/`
- A clone: `git clone https://github.com/xbill9/gemma4-dev`

---

#### Step 1 — Find a G6 With Capacity

The rigs name a g6.2xlarge (8 vCPUs, one L4). On 2026-11-05 it had no spot or on-demand capacity in any us-east-1 zone, and neither did g6.4xlarge spot. The MCP tool `create_g6_instance` walks every subnet, spot first and then on-demand:

```text
subnet-0c2872fe4182b9ec1 spot=True: ❌ AWS InsufficientInstanceCapacity: There is no Spot capacity available that matches your request.
subnet-09e0f13c0a5b43092 spot=True: ❌ AWS InsufficientInstanceCapacity: There is no Spot capacity available that matches your request.
subnet-0c2872fe4182b9ec1 spot=False: ❌ AWS InsufficientInstanceCapacity: We currently do not have sufficient g6.4xlarge capacity in the Availability Zone you requested (us-east-
subnet-0d2b2038965e8d0c0 spot=False: ✅ Launching `i-040ed957b67c2d514` (g6.4xlarge, on-demand, 1x L4) in `us-east-1`.
AMI: `ami-0f5139ed9ed52b805`
```

Both sizes carry the same L4. The g6.4xlarge doubles the vCPUs and memory, which only llama.cpp's prompt threads and the build steps use.

🔎 **Tip: all access is SSM and S3.** The instance has no SSH key and its role has no S3 access. Commands go through SSM Run Command, which returns at most 24,000 characters, so files larger than that move through presigned S3 URLs generated on the workstation.

---

#### Step 2 — One Rig Directory per Engine

Each engine has its own directory with a `tpu.env` that decides every serving flag, a Makefile run on the instance, and its measurements under `benchmarks/runs/`:

```text
gpu-llamacpp-g6-2b-q4_0/
gpu-vllm-g6-2b-q4w4a16emb4/
gpu-sglang-g6-2b-q4w4a16emb4/
gpu-mistralrs-g6-2b-q4_0/
gpu-pytorch-g6-2b-q4w4a16/
gpu-litertlm-g6-2b-int248/
gpu-litertlm-g6-2b-q4w4a16emb4/
```

`gpu-vllm-g6-2b-q4w4a16emb4/flight/setup.sh` installs every engine in order and logs each step, so a failed step reruns alone:

```text
GEMMA4_DEV=/opt/gemma4-dev MODEL_S3_URI=s3://... bash setup.sh llamacpp vllm sglang mistralrs litert
18:48:44 START llamacpp
18:56:56 END llamacpp
18:56:56 START vllm
18:59:18 END vllm
18:59:18 START sglang
18:59:57 END sglang
```

---

#### Step 3 — What the Deep Learning AMI Needs

The AMI ships the NVIDIA driver and a PyTorch 2.13 environment, and no CUDA toolkit. Four things differ from a desktop install:

- **vLLM brings its own torch.** The system Python has no torch, so `pip install vllm==0.33.0` pulls its pinned wheel from PyPI, built for CUDA 13.0, and it runs on driver 595.112.07:

```text
torch 2.14.0+cu130 cuda 13.0 NVIDIA L4 (8, 9)
```

- **llama.cpp and mistral.rs build against CUDA 13.1** from NVIDIA's apt repository, for sm_89. mistral.rs also links cuRAND, which is a separate package: `libcurand-dev-13-1`.
- **LiteRT-LM finds the L4 through Vulkan.** The AMI lists the L4 as a Vulkan device, and LiteRT-LM's WebGPU backend serves on it.
- **pip puts scripts in `~/.local/bin`.** SSM runs commands as root with `HOME=/root`, and that directory is off its `PATH`, so `vllm`, `litert-lm` and SGLang's `ninja` are found only once it is added.

🔎 **Tip: SGLang 0.5.23 needs two patches for this checkpoint.** Its Gemma 4 attention shim runs only for multimodal configs, and it leaves the int4 `embed_tokens` and per-layer table as uninitialized bf16. The rig's `patches/` keep both tables packed on the card.

---

#### Step 4 — Size Each Engine for the Whole Card

The card serves one engine at a time and nothing else, so each engine gets the most the L4 will hold:

- **vLLM** reserves 85% of the card (`--gpu-memory-utilization 0.85`), 64 running requests, 16,384 tokens each.
- **SGLang** reserves 80% (`--mem-fraction-static 0.80`), 64 running requests, decode CUDA graphs up to 64.
- **llama.cpp** runs 64 slots over a 65,536-token cache for the load sweep, and one slot of 131,072 for accuracy.
- **mistral.rs** runs 32 requests over a fixed 16,384-token cache.

🔎 **Tip: mistral.rs plans activation memory per running request.** It sets aside about 395 MiB for each, so 64 requests do not load on the L4's 23 GB:

```text
Error: Primary device cuda[0] needs 28292 MiB (2976 MiB of fixed model components, 25316 MiB of activations, 0 MiB of post-load reservations), but only 21192 MiB is usable under the memory goal.
```

At 32 requests a 65,536-token cache leaves too little room; `--max-seqs 32 --kv-cache 16384tokens --max-num-batched-tokens 4096` serves in 5,286 MiB.

---

#### Step 5 — One Request at a Time

`flight/engines.py` sends six chat prompts, 256 tokens each, three times per engine, and takes the geometric mean of each prompt's median, end to end. Then 4 requests at once:

```text
arm              e2e tok/s vs llama  decode TTFT ms 4 at once  x one    memory ready s repeat
llama                135.7    1.000   136.9      29     437.6   3.23  2930 MiB     6.1   6/6
llama-d2             207.7    1.531   211.4      30     373.2   1.80  3004 MiB     4.1   6/6
mistralrs            144.6    1.066   144.4      11     500.0   3.46  4664 MiB    24.1   6/6
pytorch              134.7    0.993       -       -     134.5   1.00  3248 MiB   236.2   6/6
litert-int248         98.4    0.725   125.2     512     100.5   1.02  1489 MiB     9.4   6/6
vllm                 140.4    1.035   140.7      16     537.2   3.83 18228 MiB   167.3   6/6
vllm-d4              275.3    2.030   281.2      19     711.3   2.58 18918 MiB   132.5   6/6
sglang               136.5    1.006   139.7      50     475.4   3.48 18345 MiB   307.2   6/6
sglang-d4            189.6    1.398   198.1      55     508.3   2.68 15191 MiB    61.1   6/6
```

| Engine | No drafter | Best drafter | Drafter gain |
| :--- | :--- | :--- | :--- |
| vLLM | 140.4 | 🥇 275.3 (4 tokens) | 1.96x |
| llama.cpp | 135.7 | 207.7 (2 tokens) | 1.53x |
| SGLang | 136.5 | 189.6 (4 tokens) | 1.39x |
| mistral.rs | 144.6 | 137.6 (1 token) | 0.95x |
| PyTorch | 134.7 | — | — |
| LiteRT-LM | 98.4 | 112.8 (its own) | 1.15x |

LiteRT-LM's exact Q4_0 build runs at 84.4 tokens per second, and LiteRT-LM takes about half a second to its first token against 11 to 55 ms for the others. Every engine repeats its own greedy text on all six prompts.

---

#### Step 6 — 1 to 64 Requests

`sweep.py` runs two shapes, 512 tokens in / 128 out and 128 in / 512 out, at 1 to 64 requests at once, five repeats, a unique prompt per request so no prefix cache can hit, output timed from the stream. `flight/suite.py` runs every sweep and every accuracy job one server at a time, and marks each finished job so a rerun skips it:

```text
python3 suite.py /opt/flight-out/2026-11-06-suite-g6 sweep:vllm,sweep:vllm-d1,sweep:vllm-d4,sweep:sglang,...
2026-11-06 04:06:20 START sweep:vllm ready_s=39.1 gpu=52, 2040 MHz, 19296 MiB
2026-11-06 04:07:33   sweep:vllm in512-out128 rc=0
2026-11-06 04:10:07   sweep:vllm in128-out512 rc=0
2026-11-06 04:10:08 END sweep:vllm ok=True
```

Aggregate output tokens per second without a drafter, 128 in / 512 out:

| Requests | vLLM | SGLang | mistral.rs | llama.cpp |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 138.8 | 134.6 | 133.9 | 133.8 |
| 8 | 991.0 | 967.4 | 626.5 | 597.5 |
| 64 | 🥇 3,715.7 | 2,967.1 | 1,825.2 | 1,056.3 |

512 in / 128 out:

| Requests | vLLM | SGLang | mistral.rs | llama.cpp |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 132.8 | 129.6 | 133.1 | 125.4 |
| 8 | 738.3 | 708.3 | 591.2 | 428.1 |
| 64 | 🥇 1,593.7 | 1,373.0 | 1,132.0 | 630.8 |

All 126 levels completed. vLLM's output at 64 requests is 26.8x its one-request rate on long replies. SGLang reaches 0.80 to 0.86x vLLM at 64, mistral.rs 0.49 to 0.71x and llama.cpp 0.28 to 0.40x.

---

#### Step 7 — Each Engine at Its Best

`best_per_level.py` picks each engine's fastest measured setting at each level, 128 in / 512 out:

| Requests | vLLM | SGLang | llama.cpp | mistral.rs |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 🥇 453 (4 drafted) | 283 (4 drafted) | 271 (2 drafted) | 134 (none) |
| 2 | 🥇 856 (4 drafted) | 552 (4 drafted) | 430 (2 drafted) | 254 (none) |
| 4 | 🥇 1,483 (4 drafted) | 875 (4 drafted) | 596 (2 drafted) | 468 (none) |
| 8 | 🥇 2,399 (4 drafted) | 1,223 (4 drafted) | 791 (2 drafted) | 627 (none) |
| 16 | 🥇 2,823 (4 drafted) | 1,673 (none) | 1,021 (2 drafted) | 1,339 (none) |
| 32 | 🥇 3,665 (1 drafted) | 2,257 (none) | 1,132 (2 drafted) | 1,843 (none) |
| 64 | 🥇 4,516 (4 drafted) | 2,967 (none) | 1,170 (2 drafted) | 1,825 (none) |

512 in / 128 out:

| Requests | vLLM | SGLang | llama.cpp | mistral.rs |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 🥇 345 (4 drafted) | 214 (4 drafted) | 224 (2 drafted) | 133 (none) |
| 8 | 🥇 1,217 (4 drafted) | 711 (4 drafted) | 494 (2 drafted) | 591 (none) |
| 16 | 🥇 1,389 (1 drafted) | 1,026 (none) | 572 (2 drafted) | 1,121 (none) |
| 64 | 🥇 1,710 (1 drafted) | 1,373 (none) | 631 (none) | 1,132 (none) |

vLLM's drafter pays at every level: with 4 drafted tokens 2.60x / 3.26x no drafter for one request, still 1.02x / 1.22x at 64. SGLang's drafter pays up to 4 to 8 requests and loses from 16 (0.73x / 0.69x at 64). llama.cpp's 2-token drafter pays at every level on long replies (1.11x at 64) and up to 16 requests on long prompts.

---

#### Step 8 — Answer Quality

`gen_eval.py` asks each engine all 1,319 GSM8K test questions (2,048-token limit) and the 400 BFCL simple function-calling questions (512), one request at a time, greedy, thinking off:

| Engine | GSM8K | BFCL | GSM8K with drafter | BFCL with drafter |
| :--- | :--- | :--- | :--- | :--- |
| llama.cpp | 0.8984 | 0.9250 | 0.8976 | 0.9200 |
| vLLM | 0.8969 | 0.9225 | 0.8992 | 0.9225 |
| SGLang | 0.8954 | 0.9150 | 0.8992 | 0.9150 |
| mistral.rs | 0.9014 | 0.9200 | — | — |
| PyTorch | 0.8961 | 0.9225 | — | — |
| LiteRT-LM, exact Q4_0 | 0.8999 | 0.8600 | — | — |
| LiteRT-LM, Google's file | 0.8908 | 0.8250 | — | — |

Paired question by question, every engine is level with llama.cpp on GSM8K, and the drafter changes no score. On BFCL, SGLang is 4 questions behind llama.cpp, a gap bigger than the run-to-run range, and LiteRT-LM is 26 and 40 behind.

The drafter cuts the time to finish GSM8K:

| Engine | No drafter | With drafter | Speedup |
| :--- | :--- | :--- | :--- |
| vLLM | 3,918 s | 🥇 1,270 s | 3.08x |
| SGLang | 4,035 s | 1,956 s | 2.06x |
| llama.cpp | 3,971 s | 2,007 s | 1.98x |
| mistral.rs | 3,918 s | — | — |
| PyTorch | 3,991 s | — | — |
| LiteRT-LM | 5,018 s | — | — |

🔎 **Tip: send thinking off explicitly to SGLang and mistral.rs.** `"chat_template_kwargs": {"enable_thinking": false}` keeps replies short on every engine; mistral.rs thinks by default on Gemma 4.

---

#### Step 9 — Cost

At the g6.4xlarge on-demand price, $1.3232 an hour in us-east-1 from the AWS Price List API, the measured throughput works out to (arithmetic, from `cost.py`):

| Setting | 128 in / 512 out | 512 in / 128 out |
| :--- | :--- | :--- |
| vLLM + drafter, 1 request | $0.81 per M tokens | $1.06 per M tokens |
| vLLM + drafter, 64 requests | 🥇 $0.08 | 🥇 $0.21 (1 drafted) |
| SGLang, 64 requests | $0.12 | $0.27 |
| mistral.rs, 64 requests | $0.20 | $0.32 |
| llama.cpp + drafter, 64 requests | $0.31 | $0.59 |

The whole sweep and accuracy suite ran 12.2 hours, $16.14 on demand. A g6.2xlarge carries the same L4 at $0.9776 an hour; it was not measured here.

---

#### Step 10 — Tear Down

Results come off the instance as one tarball through a presigned S3 PUT, and then the instance is terminated:

```text
python3 -c "import boto3; print(boto3.client('ec2', region_name='us-east-1').terminate_instances(InstanceIds=['i-040ed957b67c2d514'])['TerminatingInstances'][0]['CurrentState']['Name'])"
shutting-down
```

---

#### Compare and Contrast

| | vLLM | SGLang | llama.cpp | mistral.rs | PyTorch | LiteRT-LM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 request, best | 🥇 275.3 | 189.6 | 207.7 | 144.6 | 134.7 | 112.8 |
| 64 requests, 128/512, best | 🥇 4,516 | 2,967 | 1,170 | 1,825 | one at a time | one at a time |
| Drafter under load | 🟢 pays at every level | ❌ 0.69x at 64 | 🟢 to 16–64 | ❌ slower | — | — |
| GSM8K level | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 |
| BFCL level | 🟢 | ⚠️ 4 behind | 🟢 | 🟢 | 🟢 | ❌ 26–40 behind |
| Runs as installed | 🟢 | ❌ two patches | 🟢 | ⚠️ needs cuRAND, fixed cache | 🟢 | ⚠️ Vulkan |

---

#### So, Which One?

On a G6 serving one small model, run vLLM with Google's W4A16 drafter at 4 tokens: it is the fastest engine for one request (275.3 tokens per second) and at every load to 64 requests, it finishes GSM8K 3.08x sooner than without the drafter at the same score, and at full load it costs $0.08 per million output tokens on demand. SGLang without a drafter reaches 0.66x of that at 64 requests and needs two patches. llama.cpp with its 2-token drafter is the simplest to build and serves one user at 207.7 tokens per second, but tops out near 1,170 tokens per second. mistral.rs, PyTorch and LiteRT-LM suit one user at a time.

---

#### Summary

The goal of this article was to find the inference serving engine to run for Gemma 4 E2B on an Amazon EC2 G6. The key was holding everything else fixed: one instance, the same 4-bit values in every container, the same prompts, load sweep and question sets, and each engine sized for the whole L4. The six-engine results were:

- 🟢 All six engines give the same GSM8K answers (0.8908 to 0.9014) with the drafter on or off.
- 🟢 vLLM with its drafter is the fastest engine at every level from 1 to 64 requests: 275.3 tokens per second for one request and 4,516 at 64 (128 in / 512 out).
- 🟢 vLLM's drafter pays at every load on the L4, and finishes GSM8K 3.08x sooner.
- 🟢 At full load vLLM serves a million output tokens for $0.08 on demand.
- ⚠️ The drafter's gain depends on the engine: 1.96x vLLM, 1.53x llama.cpp, 1.39x SGLang, 1.15x LiteRT-LM, 0.95x mistral.rs.
- ⚠️ g6.2xlarge had no capacity in us-east-1 on the day; g6.4xlarge on-demand carries the same L4.
- ❌ mistral.rs loads at most 32 requests on the L4's 23 GB.
- ❌ SGLang's drafter cuts output to 0.69 to 0.73x of running without it at 64 requests.

Scope: one g6.4xlarge on-demand in us-east-1d, one NVIDIA L4, the Deep Learning AMI with driver 595.112.07, Gemma 4 E2B on Google's QAT Q4_0 grid, one engine at a time, greedy decoding. One-request speed is six prompts of 256 tokens, three repeats; load is two synthetic shapes at 1 to 64 requests, five repeats each; accuracy is one pass of GSM8K (1,319) and BFCL simple (400). The one-request runs were on 2026-11-05 and the sweeps and accuracy on 2026-11-06, on the same instance; costs are arithmetic from the on-demand price.

The strategy for comparing inference serving engines on Amazon EC2 G6 was validated with an incremental step by step approach.

---

#### References

- [gemma4-dev on GitHub](https://github.com/xbill9/gemma4-dev)
- [One-request run: 2026-11-05-engines-g6](https://github.com/xbill9/gemma4-dev/blob/main/gpu-vllm-g6-2b-q4w4a16emb4/benchmarks/runs/2026-11-05-engines-g6/REPORT.md)
- [Sweeps and accuracy: 2026-11-06-suite-g6](https://github.com/xbill9/gemma4-dev/blob/main/gpu-vllm-g6-2b-q4w4a16emb4/benchmarks/runs/2026-11-06-suite-g6/REPORT.md)
- [Amazon EC2 G6 instances](https://aws.amazon.com/ec2/instance-types/g6/)
- [AWS Deep Learning AMIs](https://aws.amazon.com/ai/machine-learning/amis/)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)
- [mistral.rs](https://github.com/EricLBuehler/mistral.rs)
- [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)
- [xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4)
- [xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf)
- [GSM8K](https://github.com/openai/grade-school-math)
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
