---
title: "Six Serving Engines on One Tesla T4: Gemma 4 E2B on a Turing GPU"
published: false
description: "llama.cpp, vLLM, SGLang, mistral.rs, a plain PyTorch engine and LiteRT-LM serve the same 4-bit Gemma 4 E2B on one 16 GB Tesla T4 on Compute Engine, with the same prompts, load sweep and GSM8K and BFCL questions. GSM8K and BFCL are level on every CUDA engine; vLLM with its drafter leads at every load."
tags: gemma, vllm, googlecloud, nvidia
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/docs/serving-engines-t4/devto-cover.7b22d2e5.jpg
---

This article provides a step by step comparison of six inference serving engines on one Tesla T4: llama.cpp, vLLM, SGLang, mistral.rs, a plain PyTorch engine and Google's LiteRT-LM, each serving Gemma 4 E2B on Google's QAT 4-bit weights from a Compute Engine VM.

https://github.com/xbill9/gemma4-dev

The T4 is the oldest and cheapest NVIDIA data-center card most clouds still rent, and it is Turing (sm_75): no bf16, 64 KiB of shared memory per block, and no 4-bit kernels in several engines' own builds. Each engine gets the same weights, the same prompts, the same 1-to-64-request load sweep and the same 1,319 GSM8K and 400 BFCL questions as on the RTX 3050 and the L4, with Google's E2B drafter wherever the engine runs one.

---

#### The Results in Brief

Answer quality holds on Turing. The five CUDA engines score GSM8K 0.8954 to 0.9030 and BFCL 0.9175 to 0.9250, each level with its own L4 run. LiteRT-LM is the exception at GSM8K 0.8734, 0.0174 below its L4 run with a paired range that excludes zero.

For one request without a drafter, five engines run at 97.8 to 113.5 tokens per second, mistral.rs fastest and LiteRT-LM at 44.6. The drafter decides the order: vLLM with 3 drafted tokens reaches 193.0, 1.78x itself, and finishes GSM8K in 0.41x the time with the same score. llama.cpp gains 1.18x, SGLang loses (0.72x), and mistral.rs's drafter does not run on the card.

Under load vLLM leads at every level. Its 1-token drafter still pays at 64 requests (1,165.5 tokens per second at 128 in / 512 out, 1.34x without it). SGLang serves the 4-bit checkpoint only with a patch that borrows vLLM's Turing kernels, and with it matches vLLM without a drafter on long replies (908.6 against 871.8 at 64). mistral.rs holds 4 requests at a time and stays at about 238.

---

#### What Changes and What Stays

| Engine | Version | Weights on the T4 | Turing changes |
| :--- | :--- | :--- | :--- |
| llama.cpp | `7182ef71e` | exact Q4_0 GGUF | none, built for sm_75 |
| vLLM | 0.33.0 | W4A16, int4 tables | Triton attention tile clamp |
| SGLang | 0.5.23 | W4A16, int4 tables | four patches, Marlin from vLLM |
| mistral.rs | v0.9.5 | exact Q4_0 GGUF | built without flash-attn |
| PyTorch | this repo's int4 engine | W4A16 | none, fp16 prompt reads |
| LiteRT-LM | 0.19.0 | Google's int2/4/8 file | none, WebGPU on Vulkan |

The same for all six: one Tesla T4 (TU104, sm_75, 15,360 MiB, driver 620.70.06) on a GCE `n1-standard-2` (2 vCPU, 7.4 GiB RAM), Gemma 4 E2B on Google's QAT Q4_0 grid, greedy decoding, thinking off for the speed and accuracy runs, and one Python 3.13 interpreter with no virtual environment. Every engine version and checkpoint revision matches the 3050 and L4 runs.

---

#### At This Point You Should Have…

- A Compute Engine VM with one T4, a CUDA 13 toolkit and the NVIDIA Vulkan driver for LiteRT-LM
- The system `python3` with pip, and a Rust toolchain for mistral.rs
- A clone: `git clone https://github.com/xbill9/gemma4-dev`
- Google's E2B QAT assistant drafter as exact Q4_0 GGUF for llama.cpp and W4A16 for vLLM and SGLang

---

#### Step 1 — One Rig Directory per Engine

Each engine has its own directory with a `tpu.env` that decides every serving flag and a `serve.py` or Makefile that turns it into a command:

```text
gpu-llamacpp-t4-2b-q4_0/
gpu-vllm-t4-2b-q4w4a16emb4/
gpu-sglang-t4-2b-q4w4a16emb4/
gpu-mistralrs-t4-2b-q4_0/
gpu-pytorch-t4-2b-q4w4a16/
gpu-litertlm-t4-2b-int248/
```

Ports differ (llama.cpp 8080, vLLM 8010, mistral.rs 8020, SGLang 8030, PyTorch 8100, LiteRT-LM 9379), and only one engine is on the card at a time.

---

#### Step 2 — What Turing Needs From Each Engine

**vLLM** picks its Triton attention backend for Gemma 4's 256- and 512-wide heads, and that kernel's tile asks for more than sm_75's 64 KiB of shared memory. `patch_triton_turing.py` clamps the tile; vLLM then serves with `--dtype float16`, `MarlinLinearKernel` and Triton attention, and holds an 880,057-token cache at 85% of the card.

**mistral.rs** builds with `--features cuda` and `CUDA_COMPUTE_CAP=75`. Its flash-attn feature needs sm_80, so it runs without it.

**PyTorch** reads prompts in fp16 here: the T4 measured 19.51 TFLOPS fp16 against 2.82 fp32.

**LiteRT-LM** serves Google's `gemma-4-E2B-it.litertlm`. The exact Q4_0 build the 3050 used needs about 45 GB of RAM to convert, and this VM has 7.4 GiB.

🔎 **Tip: SGLang's own 4-bit kernels are empty below sm_80.** SGLang 0.5.23 compiles its Marlin kernels for Ampere and newer, so on a T4 the W4A16 checkpoint does not load and the only stock route is the QAT values stored as 16-bit, 10.00 GB on the card. The rig's `marlin-sm75-via-vllm.diff` sends SGLang's `gptq_marlin_gemm` and `gptq_marlin_repack` to vLLM 0.33's compiled Turing build when compute capability is below 8, so vLLM must be installed beside SGLang. The 4-bit weights then take 3.25 GB, and SGLang's repack and GEMM agree with an fp16 matmul on the unpacked weights to a relative error of 0.7 to 2.0e-3 at E2B's shapes.

🔎 **Tip: turn off Marlin's atomic-add reduction in SGLang on float16.** SGLang 0.5.23 hard-codes it on, and it applies only to float16 activations, which on Turing means always. With it on, greedy text repeated on only 4 of 6 prompts; the patch turns it off unless `SGLANG_MARLIN_USE_ATOMIC_ADD=1`, as vLLM does, and every prompt repeats.

---

#### Step 3 — One Request at a Time

`engines.py` sends six chat prompts, 256 tokens each, three times per engine, and takes the geometric mean of each prompt's median, end to end:

```text
arm              e2e tok/s vs llama  decode TTFT ms 4 at once  x one    memory ready s repeat
llama                104.8    1.000   103.7      45     211.8   2.02  2829 MiB     8.3   6/6
llama-d1             123.5    1.179   118.7      46     187.1   1.51  2903 MiB     6.3   6/6
vllm                 108.4    1.034   104.3      34     344.1   3.17 12771 MiB   110.3   6/6
vllm-d3              193.0    1.841   194.2      39     408.0   2.11 12187 MiB   299.8   6/6
pytorch              109.8    1.048       -       -     106.1   0.97  3141 MiB    94.3   6/6
litert-int248         44.6    0.425    79.2    2400      44.7   1.00  1240 MiB    20.6   6/6
mistralrs            113.5    1.083   112.1      18     273.1   2.41  4579 MiB    46.2   6/6
```

| Engine | No drafter | Best drafter | Drafter gain |
| :--- | :--- | :--- | :--- |
| vLLM | 108.4 | 193.0 (3 tokens) | 1.78x |
| llama.cpp | 104.8 | 123.5 (1 token) | 1.18x |
| SGLang, 4-bit | 97.8 | 70.7 (4 tokens) | 0.72x |
| mistral.rs | 113.5 | does not run | — |
| PyTorch | 109.8 | — | — |
| LiteRT-LM | 44.6 | 43.7 (its own) | 0.98x |

Every engine repeats its own greedy text on all six prompts. mistral.rs reads the prompt fastest (18 ms to the first token) and LiteRT-LM slowest (2,400 ms).

🔎 **Tip: mistral.rs's drafter needs bf16.** Google's assistant drafter loads as bf16 safetensors, and on a card that runs the target model in float16 every drafted request fails:

```text
step - Model failed with error: dtype mismatch in matmul, lhs: F16, rhs: BF16
```

---

#### Step 4 — SGLang in 4-Bit Against 16-Bit

The same QAT values, served by the same SGLang build two ways: as the W4A16 checkpoint through the Marlin patch, and as the 16-bit `google/gemma-4-E2B-it-qat-q4_0-unquantized` the stock build can load.

```text
arm              e2e tok/s vs llama  decode TTFT ms 4 at once  x one    memory ready s repeat
sglang                40.2        -    40.3      78     157.4   3.92 14151 MiB   334.0   6/6
sglang4               97.8        -    98.3      93     343.6   3.51 13427 MiB   149.4   6/6
```

The 4-bit build is 2.43x the 16-bit one for one request. Under load the gap closes as requests rise: on 128 in / 512 out it is 2.22x at one request and 1.14x at 64, and on 512 in / 128 out 1.88x falls to 0.99x.

---

#### Step 5 — 1 to 64 Requests, No Drafter

`sweep.py` runs two shapes, 512 tokens in / 128 out and 128 in / 512 out, at 1 to 64 requests at once, five repeats, a unique prompt per request so no prefix cache can hit, output timed from the stream. Aggregate output tokens per second, 128 in / 512 out:

| Requests | vLLM | SGLang 4-bit | llama.cpp | mistral.rs |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 101.5 | 88.2 | 86.9 | 96.9 |
| 8 | 491.9 | 479.3 | 240.2 | 238.6 |
| 64 | 871.8 | 908.6 | 381.3 | 237.3 |

512 in / 128 out:

| Requests | vLLM | SGLang 4-bit | llama.cpp | mistral.rs |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 85.4 | 65.3 | 91.8 | 97.2 |
| 8 | 236.5 | 152.3 | 204.2 | 218.9 |
| 64 | 295.3 | 175.0 | 281.9 | 196.7 |

SGLang is level with vLLM on long replies from 4 requests up and gives 0.59 to 0.76x on long prompts. On long prompts llama.cpp reaches 0.95x vLLM at 64 requests; on long replies 0.44x. mistral.rs peaks at 4 requests (237.7 and 224.6) and stays there, because the rig holds 4 requests with a fixed 8,192-token cache, sized for the 3050's 8 GB.

---

#### Step 6 — 1 to 64 Requests, Each Engine at Its Best

The same sweep with every drafter length, one server per length; `ladder.py` picks each engine's fastest setting at each level. 128 in / 512 out:

| Requests | vLLM | SGLang 4-bit | llama.cpp | mistral.rs |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 🥇 275.6 (4 drafted) | 88.2 (none) | 169.6 (4 drafted) | 96.9 (none) |
| 2 | 🥇 423.0 (3 drafted) | 169.6 (none) | 244.7 (4 drafted) | 168.8 (none) |
| 4 | 🥇 609.7 (3 drafted) | 305.5 (none) | 319.0 (4 drafted) | 237.7 (none) |
| 8 | 🥇 785.8 (3 drafted) | 479.3 (none) | 377.2 (3 drafted) | 238.6 (none) |
| 16 | 🥇 948.6 (1 drafted) | 663.6 (none) | 422.5 (3 drafted) | 238.0 (none) |
| 32 | 🥇 1,085.9 (1 drafted) | 815.8 (none) | 459.7 (4 drafted) | 236.9 (none) |
| 64 | 🥇 1,165.5 (1 drafted) | 908.6 (none) | 462.5 (4 drafted) | 237.3 (none) |

512 in / 128 out:

| Requests | vLLM | SGLang 4-bit | llama.cpp | mistral.rs |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 🥇 157.8 (4 drafted) | 65.3 (none) | 140.7 (4 drafted) | 97.2 (none) |
| 8 | 🥇 285.0 (1 drafted) | 152.3 (none) | 256.0 (2 drafted) | 218.9 (none) |
| 64 | 🥇 328.9 (1 drafted) | 175.0 (none) | 281.9 (none) | 196.7 (none) |

vLLM's best draft length falls from 4 tokens to 1 as requests rise, and its 1-token drafter still gives 1.34x at 64 requests on long replies and 1.11x on long prompts. On the 3050 the same drafter was level with none from 32 requests. llama.cpp's drafter pays at every level on long replies.

🔎 **Tip: the best llama.cpp draft length depends on the text.** On the six chat prompts 1 drafted token is best (1.18x); on the sweep's synthetic prompts 4 tokens is best (169.6 against 86.9 at one request). Set it from your own traffic.

---

#### Step 7 — Answer Quality

`gen_eval.py` asks each engine all 1,319 GSM8K test questions (2,048-token limit) and the 400 BFCL simple function-calling questions (512), one request at a time, greedy, thinking off. Each score is paired question by question with the same engine on the L4:

| Engine | GSM8K | BFCL | GSM8K vs L4 | GSM8K hours |
| :--- | :--- | :--- | :--- | :--- |
| llama.cpp | 0.9030 | 0.9250 | +0.0045 | 1.50 |
| vLLM | 0.8999 | 0.9225 | +0.0030 | 1.49 |
| mistral.rs | 0.8999 | 0.9175 | −0.0015 | 1.49 |
| PyTorch | 0.8961 | 0.9225 | 0.0000 | 1.41 |
| SGLang 4-bit | 0.8954 | 0.9175 | 0.0000 | 1.66 |
| LiteRT-LM | 0.8734 | 0.8800 | −0.0174 | 2.74 |
| vLLM, 3 drafted | 0.8984 | 0.9225 | −0.0015 vs none | 0.61 |
| llama.cpp, 1 drafted | 0.8984 | 0.9250 | −0.0045 vs none | 1.15 |

Every CUDA engine is level with its L4 run and with T4 llama.cpp on both tests, and the drafters change no score. vLLM with its drafter finishes GSM8K in 0.41x the time of either engine without one. LiteRT-LM is 0.0174 below its L4 run (range −0.0311 to −0.0023) and 0.0296 below T4 llama.cpp. Its BFCL ran with LiteRT-LM's constrained tool decoding off, as on the 3050 (0.890 there), and is level with that run.

🔎 **Tip: mistral.rs thinks by default on Gemma 4.** At its default it scores BFCL 0.8675 on the T4 with 18 replies cut off at 512 tokens, in 10.27x the time. Send `"chat_template_kwargs": {"enable_thinking": false}` for short replies.

---

#### Step 8 — The T4 Against the L4

The same six prompts on an EC2 G6 (one L4) with the same engine builds, end to end for one request:

```text
arm                  T4      L4  T4/L4
vllm              108.4   140.4  0.772
sglang             97.8   136.5  0.716
mistralrs         113.5   144.6  0.785
pytorch           109.8   134.7  0.815
llama             104.8   135.7  0.772
litert-int248      44.6    98.4  0.453
vllm-d4           190.0   275.3  0.690
llama-d2          121.5   207.7  0.585
sglang-d4          70.7   189.6  0.373
```

Without a drafter the CUDA engines run at 0.72 to 0.81x the L4. With one the T4 falls further behind (0.37 to 0.69x), likely because a drafted step costs more of a target step on the older card (inferred, not measured here).

---

#### Compare and Contrast

| | vLLM | SGLang 4-bit | llama.cpp | mistral.rs | PyTorch | LiteRT-LM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 request, best | 🥇 193.0 | 97.8 | 123.5 | 113.5 | 109.8 | 44.6 |
| 64 requests, 128/512, best | 🥇 1,165.5 | 908.6 | 462.5 | 237.3 | one at a time | one at a time |
| Drafter on the T4 | 🟢 pays at 64 | ❌ 0.72x | 🟢 1.18x | ❌ does not run | — | ❌ 0.98x |
| GSM8K / BFCL level | 🟢 yes | 🟢 yes | 🟢 yes | 🟢 yes | 🟢 yes | ⚠️ GSM8K −0.017 |
| Runs as installed | ⚠️ one patch | ❌ four patches | 🟢 as installed | ⚠️ no flash-attn | 🟢 as installed | ⚠️ int2/4/8 file |
| Memory, one request | 12,771 MiB | 13,427 MiB | 2,829 MiB | 4,579 MiB | 3,141 MiB | 1,240 MiB |

---

#### So, Which One?

On a T4, run vLLM with Google's W4A16 drafter: 193.0 tokens per second for one request, the fastest engine at every load from 1 to 64 requests, and the same GSM8K and BFCL scores in 0.41x the GSM8K time. Drop to 1 drafted token from about 8 requests on long prompts and 16 on long replies. llama.cpp with a 1-token drafter is the simplest engine to run and serves one user at 123.5 tokens per second in under 3 GB. SGLang matches vLLM without a drafter on long replies only after borrowing vLLM's Turing kernels, and its drafter loses here. mistral.rs, PyTorch and LiteRT-LM suit one user at a time.

---

#### Summary

The goal of this article was to find the inference serving engine to run for Gemma 4 E2B on a Tesla T4. The key was holding everything else fixed: the same 4-bit values, prompts, load sweep and question sets as the 3050 and L4 runs, and each engine given the changes Turing needs. The six-engine results were:

- 🟢 The five CUDA engines score GSM8K 0.8954 to 0.9030 and BFCL 0.9175 to 0.9250 on the T4, each level with its L4 run; drafters change no score.
- 🟢 vLLM with 3 drafted tokens serves one request at 193.0 tokens per second (1.78x) and finishes GSM8K in 0.41x the time.
- 🟢 vLLM's 1-token drafter pays at every load, 1.34x at 64 requests on long replies (1,165.5 tokens per second).
- 🟢 SGLang serves the 4-bit checkpoint through vLLM's Turing Marlin kernels at 2.43x its 16-bit build for one request, and matches vLLM without a drafter on long replies.
- ⚠️ Without a drafter five engines are within 97.8 to 113.5 tokens per second for one request, 0.72 to 0.81x the L4.
- ⚠️ mistral.rs holds 4 requests at a time with the rig's fixed cache and stays at about 238 tokens per second under load.
- ❌ LiteRT-LM serves Google's int2/4/8 file at 44.6 tokens per second and scores GSM8K 0.0174 below its L4 run.
- ❌ SGLang's drafter runs at 0.72x on the T4 and mistral.rs's does not run.

Scope: one Tesla T4 on one GCE `n1-standard-2` in `us-west2-b`, Gemma 4 E2B on Google's QAT Q4_0 grid, one engine at a time, greedy decoding. One-request speed is six prompts of 256 tokens, three repeats; load is two synthetic shapes at 1 to 64 requests, five repeats each; accuracy is one pass of GSM8K (1,319) and BFCL simple (400). The runs span 2026-11-05 to 2026-11-11 with the same engine builds throughout; the SGLang 4-bit arms ran on 2026-11-10 beside a same-day vLLM run (108.8 against 108.4 tokens per second). The L4 comparison uses the 2026-11-05 G6 runs with the same builds and prompts on a different host.

The strategy for comparing inference serving engines on one Tesla T4 was validated with an incremental step by step approach.

---

#### References

- [gemma4-dev on GitHub](https://github.com/xbill9/gemma4-dev)
- [T4 one-request runs and load-sweep tables](https://github.com/xbill9/gemma4-dev/tree/main/gpu-vllm-t4-2b-q4w4a16emb4/benchmarks/runs/2026-11-05-engines-t4)
- [T4 accuracy, all arms](https://github.com/xbill9/gemma4-dev/blob/main/gpu-vllm-t4-2b-q4w4a16emb4/benchmarks/runs/2026-11-06-accuracy-t4/ACCURACY.txt)
- [SGLang Turing patches](https://github.com/xbill9/gemma4-dev/tree/main/gpu-sglang-t4-2b-q4w4a16emb4/patches)
- [Six Serving Engines, One 8 GB GPU: Gemma 4 E2B on an RTX 3050](https://dev.to/gde/six-serving-engines-one-8-gb-gpu-gemma-4-e2b-on-an-rtx-3050-ad7)
- [Six Serving Engines on One EC2 G6: Gemma 4 E2B on an NVIDIA L4](https://dev.to/aws-builders/six-serving-engines-on-one-ec2-g6-gemma-4-e2b-on-an-nvidia-l4-3ff3)
- [Six Serving Engines on One Cloud TPU v5e Chip: Gemma 4 E2B](https://dev.to/gde/six-serving-engines-on-one-cloud-tpu-v5e-chip-gemma-4-e2b-22hn)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)
- [mistral.rs](https://github.com/EricLBuehler/mistral.rs)
- [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)
- [xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4)
- [xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf)
- [GSM8K](https://github.com/openai/grade-school-math)
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
