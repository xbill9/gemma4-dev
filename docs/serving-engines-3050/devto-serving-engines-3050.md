---
title: "Six Serving Engines, One 8 GB GPU: Gemma 4 E2B on an RTX 3050"
published: false
description: "llama.cpp, vLLM, SGLang, mistral.rs, a plain PyTorch engine and LiteRT-LM serve the same 4-bit Gemma 4 E2B on one 8 GB RTX 3050, measured with the same prompts, the same load sweep and the same GSM8K and BFCL questions. GSM8K is level on every engine; vLLM is the fastest up to 16 requests and within 3% of SGLang beyond."
tags: gemma, vllm, llamacpp, nvidia
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/docs/serving-engines-3050/devto-cover.91723b8d.jpg
---

This article provides a step by step comparison of six inference serving engines on one consumer GPU: llama.cpp, vLLM, SGLang, mistral.rs, a plain PyTorch engine and Google's LiteRT-LM, each serving Gemma 4 E2B on Google's QAT 4-bit weights from an 8 GB RTX 3050 in a desktop.

https://github.com/xbill9/gemma4-dev

The question is which engine to run when the model lives on a small local card that does nothing else. Each engine gets the same weights, the same prompts, the same 1-to-64-request load sweep and the same 1,319 GSM8K and 400 BFCL questions, and each is measured with and without Google's E2B drafter wherever the engine supports one.

---

#### The Results in Brief

Answer quality barely moves between engines. GSM8K lands between 0.8992 and 0.9030 and BFCL between 0.9125 and 0.925 on all six, with the drafter on or off, and on GSM8K every paired difference is within the run-to-run range. On BFCL two engines trail vLLM by 4 and 5 questions.

For one request, five engines are within 12% of each other without a drafter (108.8 to 121.1 tokens per second end to end) and LiteRT-LM runs at about half that. The drafter decides the order: it gives vLLM 1.90x, llama.cpp 1.35x, SGLang 1.33x and LiteRT-LM 1.10x, and it slows mistral.rs to 0.72x.

Under load the batching engines separate. At 64 requests of 128 tokens in and 512 out without a drafter, vLLM reaches 1,651 tokens per second and SGLang 1,711, against 633 for llama.cpp and 308 for mistral.rs. PyTorch and LiteRT-LM serve one request at a time.

With each engine at its best setting, vLLM is the fastest from 1 to 16 requests, and SGLang leads it by 1 to 3% at 32 and 64 requests on long replies. llama.cpp gains nothing from the rest of the card: with every table on the GPU and larger batches it uses 4.3 GB and runs at 0.996 to 1.024x its 2.6 GB speed.

---

#### What Changes and What Stays

| Engine | Version | Weights | Installed with |
| :--- | :--- | :--- | :--- |
| llama.cpp | `7182ef71e` | exact Q4_0 GGUF | CMake, CUDA, sm_86 |
| vLLM | 0.33.0 | W4A16, int4 tables | pip, `--no-deps` over torch 2.14.1 |
| SGLang | 0.5.23 | W4A16, int4 tables | pip, cp313 wheel on Python 3.14, two patches |
| mistral.rs | v0.9.5 | exact Q4_0 GGUF | `cargo install`, CUDA and flash-attn |
| PyTorch | this repo's int4 engine | W4A16 | in-repo Triton kernels |
| LiteRT-LM | 0.19.0 | exact Q4_0, and Google's int2/4/8 file | pip, GPU through WebGPU on Vulkan |

The same for all six: one RTX 3050 OEM (GA106, sm_86, 8,192 MiB, driver 615.78.08) in an i7-5820K desktop, Gemma 4 E2B on Google's QAT Q4_0 grid in every container, greedy decoding, and one Python 3.14 interpreter with no virtual environment. The exact Q4_0 GGUF and the W4A16 checkpoint carry the same 4-bit values: `xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf` for llama.cpp and mistral.rs, `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4` for vLLM and SGLang.

---

#### At This Point You Should Have…

- An NVIDIA GPU with 8 GB and a CUDA toolkit
- The system `python3` with pip, and a Rust toolchain for mistral.rs
- A clone: `git clone https://github.com/xbill9/gemma4-dev`
- Google's E2B QAT assistant drafter, as exact Q4_0 GGUF for llama.cpp, W4A16 for vLLM and SGLang, bf16 for mistral.rs

---

#### Step 1 — One Rig Directory per Engine

Each engine has its own directory with a `tpu.env` that decides every serving flag, a `serve.py` or Makefile that turns it into a command, and its measurements under `benchmarks/runs/`:

```text
local-llamacpp-3050oem-2b-q4_0/
local-vllm-3050oem-2b-q4w4a16emb4/
local-sglang-3050oem-2b-q4w4a16emb4/
local-mistralrs-3050oem-2b-q4_0/
local-pytorch-3050oem-2b-q4w4a16/
local-litertlm-3050oem-2b-int248/
```

```text
cd local-sglang-3050oem-2b-q4w4a16emb4 && make command
python3 -m sglang.launch_server --model-path xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4 --revision db715f83da4f13289761d7c45f4d19ceac84695d --host 127.0.0.1 --port 8030 --mem-fraction-static 0.80 ...
```

Only one engine serves at a time; none of them fits beside another on 8 GB.

---

#### Step 2 — Install vLLM and SGLang on the System Python

vLLM 0.33.0 pins `torch==2.14.0` and the machine has 2.14.1. Its kernels are built against libtorch's stable ABI, so `make install` installs its other requirements and then vLLM itself with `--no-deps`, and it serves on 2.14.1.

SGLang 0.5.23 ships wheels for Python 3.11 to 3.13. Its HTTP server runs on 3.14 from the cp313 wheel retagged, installed with `--no-deps`, plus only the requirements that were missing. No package already installed changed version.

🔎 **Tip: SGLang 0.5.23 needs two patches for this checkpoint.** Its Gemma 4 attention shim runs only for multimodal configs, and a text-only checkpoint fails at load on `head_dim` (Gemma 4's sliding layers are 256 wide, its full layers 512). And it reads the int4 `lm_head` from compressed-tensors but leaves the int4 `embed_tokens` and per-layer table as uninitialized bf16, which takes 6.93 GB. The rig's `patches/` keep both tables packed on the card and unpack only the rows a request uses:

```text
Load weight end. elapsed=1.15 s, type=Gemma4ForCausalLM, quant=compressed-tensors, avail mem=4.34 GB, mem usage=3.25 GB.
```

---

#### Step 3 — Size Each Engine for 8 GB

Each engine needs a setting that fits the card:

- **vLLM** reserves 85% of the card up front (`--gpu-memory-utilization 0.85`), 64 running requests.
- **SGLang** reserves 80% (`--mem-fraction-static 0.80`), 64 running requests, decode CUDA graphs captured up to 64. Prompt-reading graphs are off, because capture refuses Gemma 4's 20 layers that share another layer's cache.
- **llama.cpp** runs 64 slots of 1,024 tokens for the load sweep.
- **mistral.rs** runs 4 requests with a fixed 8,192-token cache and 1,024 tokens per step.

🔎 **Tip: size mistral.rs by tokens, never by memory share.** With `--gpu-memory-utilization` it reserves about 300 MiB of image and audio encoder memory per running request and then answers every request with HTTP 503 ("Memory pressure prevented forward admission"), even one at a time. `--kv-cache 8192tokens --max-seqs 4 --max-num-batched-tokens 1024` serves.

---

#### Step 4 — One Request at a Time

`engines.py` sends six chat prompts, 256 tokens each, three times per engine, and takes the geometric mean of each prompt's median, end to end:

```text
arm              e2e tok/s vs llama  decode TTFT ms 4 at once  x one    memory ready s repeat
pytorch              118.9    1.093       -       -     119.1   1.00  3121 MiB   236.1   6/6
litert-q4             61.9    0.569    79.5     828      63.6   1.03  1846 MiB     2.0   6/6
llama                108.8    1.000   110.5      42     277.4   2.55  2809 MiB     4.0   6/6
llama-d2             147.4    1.355   150.7      42     285.4   1.94  2885 MiB     4.0   6/6
vllm                 121.1    1.113   124.5      53     454.6   3.75  6697 MiB    48.0   6/6
vllm-d4              230.5    2.118   244.5      55     470.7   2.04  7345 MiB    66.0   6/6
```

| Engine | No drafter | Best drafter | Drafter gain |
| :--- | :--- | :--- | :--- |
| vLLM | 121.1 | 230.5 (4 tokens) | 1.90x |
| SGLang | 119.9 | 158.9 (4 tokens) | 1.33x |
| mistral.rs | 118.9 | 85.8 (1 token) | 0.72x |
| PyTorch | 118.9 | — | — |
| llama.cpp | 108.8 | 147.4 (2 tokens) | 1.35x |
| LiteRT-LM | 61.9 | 68.1 (its own) | 1.10x |

SGLang and mistral.rs ran the next day beside a same-day reference engine (vLLM for SGLang, llama.cpp for mistral.rs); each matched its reference within 1%. Every engine repeats its own greedy text on all six prompts.

---

#### Step 5 — Why the Drafter Pays More on vLLM

Every engine accepts about the same share of drafted tokens at each draft length: vLLM 0.602 / 0.482 / 0.378 / 0.310 at 1 to 4 tokens, llama.cpp 0.588 / 0.455 / 0.376 / 0.311, SGLang 0.62 / 0.45 / 0.35 / 0.29, mistral.rs 61.6% / 49.4% / 42.7% at 1 to 3. What differs is what each engine pays to check a drafted step. vLLM keeps gaining up to 4 drafted tokens; llama.cpp peaks at 2; mistral.rs loses at every length.

🔎 **Tip: drafter precision does not change acceptance.** On vLLM the W4A16 drafter is level with bf16 (1.009 to 1.013x) at the same acceptance, and on llama.cpp the exact Q4_0 drafter matches Q8_0 and beats bf16. Use the 4-bit drafter.

---

#### Step 6 — 1 to 64 Requests, No Drafter

`sweep.py` runs two shapes, 512 tokens in / 128 out and 128 in / 512 out, at 1 to 64 requests at once, five repeats, a unique prompt per request so no prefix cache can hit, thinking on, output timed from the stream:

```text
python3 sweep.py --base "$ENDPOINT/v1" --model "$MODEL_NAME" --out "$R/$3" \
  --concurrency 1,2,4,8,16,32,64 --conc-input "$1" --conc-output "$2" \
  --repeats 5 --prompt-mode unique --decode-source auto --expect-device gpu
```

Aggregate output tokens per second, 128 in / 512 out:

| Requests | vLLM | SGLang | llama.cpp | mistral.rs |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 120.0 | 119.6 | 107.1 | 111.8 |
| 8 | 777.2 | 779.7 | 312.6 | 308.7 |
| 64 | 1,651.2 | 1,711.3 | 633.3 | 308.0 |

512 in / 128 out:

| Requests | vLLM | SGLang | llama.cpp | mistral.rs |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 105.4 | 103.4 | 95.3 | 107.9 |
| 8 | 422.2 | 391.2 | 228.1 | 270.2 |
| 64 | 603.1 | 544.0 | 367.6 | 258.5 |

SGLang is level with vLLM on long replies (1.00 to 1.04x) and gives 0.90 to 0.98x on long prompts. mistral.rs leads llama.cpp up to 4 requests and then stays at 258 to 309 tokens per second, because it holds 4 requests at a time on this card.

---

#### Step 7 — 1 to 64 Requests, Each Engine at Its Best

The same sweep with every drafter length, one server per length. `best_per_level.py` picks each engine's fastest setting at each level, 128 in / 512 out:

| Requests | vLLM | SGLang | llama.cpp | mistral.rs |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 🥇 377 (4 drafted) | 228 (4 drafted) | 190 (2 drafted) | 112 (none) |
| 2 | 🥇 643 (4 drafted) | 364 (4 drafted) | 255 (2 drafted) | 221 (none) |
| 4 | 🥇 994 (3 drafted) | 435 (none) | 376 (2 drafted) | 309 (none) |
| 8 | 🥇 1,265 (3 drafted) | 780 (none) | 476 (2 drafted) | 309 (none) |
| 16 | 🥇 1,463 (1 drafted) | 1,269 (none) | 614 (2 drafted) | 308 (none) |
| 32 | 1,576 (1 drafted) | 🥇 1,587 (none) | 659 (2 drafted) | 308 (none) |
| 64 | 1,658 (1 drafted) | 🥇 1,711 (none) | 633 (none) | 308 (none) |

512 in / 128 out:

| Requests | vLLM | SGLang | llama.cpp | mistral.rs |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 🥇 235 (4 drafted) | 158 (4 drafted) | 152 (2 drafted) | 108 (none) |
| 8 | 🥇 513 (3 drafted) | 391 (none) | 291 (2 drafted) | 270 (none) |
| 64 | 🥇 603 (none) | 544 (none) | 368 (none) | 259 (none) |

vLLM's best draft length falls from 4 tokens to 1 as requests rise, and its drafter is level with none from 32 requests on. SGLang's drafter pays only at 1 and 2 requests: from 8 requests on, with any draft length, its output stays at or below 271 tokens per second on 512/128 and 518 on 128/512, 0.30 to 0.49x of running without it at 64. llama.cpp's 2-token drafter pays up to 16 or 32 requests.

---

#### Step 8 — Give llama.cpp the Whole Card

The card serves nothing but the model, so memory an engine leaves unused buys nothing. llama.cpp's sweep used 2,621 MiB, with its 1,260 MiB per-layer table in CPU RAM and default batches. Four settings, at 8 and 64 requests:

| Setting | 512/128 at 64 | 128/512 at 64 | Peak memory |
| :--- | :--- | :--- | :--- |
| as swept | 368 | 627 | 2,621 MiB |
| + per-layer table on the card | 372 | 633 | 3,881 MiB |
| + `-ub 2048 -b 2048` | 377 | 633 | 4,255 MiB |
| + one shared cache (`-kvu`) | 206 | 443 | 4,281 MiB |

The full 1-to-64 sweep with the table on the card and 2,048-token batches runs at 0.996 to 1.024x the original at every level, and with the drafter on it falls to 0.39 to 0.92x. llama.cpp's limit under load is the time per generated token: 110 ms at 64 requests on 512/128, against 33 ms for SGLang and 60 ms for vLLM on the same card.

---

#### Step 9 — Answer Quality

`gen_eval.py` asks each engine all 1,319 GSM8K test questions (2,048-token limit) and the 400 BFCL simple function-calling questions (512), one request at a time, greedy, thinking off:

| Engine | GSM8K | BFCL | GSM8K with drafter | BFCL with drafter |
| :--- | :--- | :--- | :--- | :--- |
| llama.cpp | 0.9014 | 0.9225 | 0.8969 | 0.9250 |
| vLLM | 0.9014 | 0.9250 | 0.8992 | 0.9225 |
| SGLang | 0.8992 | 0.9150 | 0.8992 | 0.9175 |
| mistral.rs | 0.9030 | 0.9150 | — | — |
| PyTorch | 0.8992 | 0.9225 | — | — |
| LiteRT-LM | 0.8992 | 0.9125 | — | — |

Paired question by question, every engine is level with every other on GSM8K, and the drafter changes no score anywhere. On BFCL two pairs fall outside the run-to-run range. SGLang is 4 questions behind vLLM on the same checkpoint (0 won, 4 lost), and two of the four are SGLang's tool-call parser returning `1e-9` and `100e-06` as strings where vLLM returns numbers. LiteRT-LM is 5 behind vLLM (0 won, 5 lost).

🔎 **Tip: mistral.rs thinks by default on Gemma 4.** Left at its default it scores BFCL 0.860 with 19 replies cut off at 512 tokens and takes 9.94x as long. Send `"chat_template_kwargs": {"enable_thinking": false}` for short replies; the other five engines default to thinking off.

---

#### Compare and Contrast

| | vLLM | SGLang | llama.cpp | mistral.rs | PyTorch | LiteRT-LM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 request, best | 🥇 230.5 | 158.9 | 147.4 | 118.9 | 118.9 | 68.1 |
| 64 requests, 128/512, best | 1,658 | 🥇 1,711 | 633 | 308 | one at a time | one at a time |
| Drafter under load | 🟢 level or ahead | ❌ 0.30x at 64 | 🟢 to 16–32 | ❌ slower | — | — |
| GSM8K / BFCL level | 🟢 yes | 🟢 yes | 🟢 yes | 🟢 yes | 🟢 yes | 🟢 yes |
| Runs as installed | ⚠️ `--no-deps` | ❌ two patches | 🟢 as installed | ⚠️ fixed cache | 🟢 as installed | ⚠️ newer protobuf |
| Peak memory | 7,091 MiB | 6,966 MiB | 2,621 MiB | 4,755 MiB | 3,121 MiB | 1,846 MiB |

---

#### So, Which One?

On a card that serves only the model, run vLLM with Google's W4A16 drafter: it is the fastest engine for one request (230.5 tokens per second) and at every load to 16 requests, within 3% of the best at 32 and 64, and its drafter costs nothing under load. SGLang matches vLLM without a drafter and slightly passes it at 32 to 64 requests on long replies, but it needs two patches for this checkpoint and its drafter is a loss beyond two requests. llama.cpp with its 2-token drafter is the simplest engine to run and serves one user at 147.4 tokens per second; under load it reaches 0.38x vLLM, and more memory does not change that. mistral.rs, PyTorch and LiteRT-LM suit one user at a time.

---

#### Summary

The goal of this article was to find the inference serving engine to run for Gemma 4 E2B on an 8 GB consumer GPU. The key was holding everything else fixed: the same 4-bit values in every container, the same prompts, load sweep and question sets, and each engine given its best setting for the whole card. The six-engine results were:

- 🟢 All six engines give the same GSM8K answers (0.8992 to 0.9030) with the drafter on or off; BFCL spans 0.9125 to 0.925, SGLang and LiteRT-LM 4 and 5 questions behind vLLM.
- 🟢 vLLM with its drafter is the fastest engine from 1 to 16 requests: 230.5 tokens per second for one request and 1,265 at 8 (128 in / 512 out), and 1,658 at 64.
- 🟢 SGLang without a drafter matches vLLM under load and leads it by 1 to 3% at 32 and 64 requests on long replies, peaking at 1,711 tokens per second.
- 🟢 llama.cpp with its drafter serves one request at 147.4 tokens per second, the fastest after vLLM and SGLang.
- ⚠️ The drafter's gain depends on the engine, at the same acceptance: 1.90x vLLM, 1.35x llama.cpp, 1.33x SGLang, 0.72x mistral.rs.
- ⚠️ llama.cpp runs at the same speed with 4.3 GB as with 2.6 GB; its limit under load is the time per generated token.
- ❌ SGLang's drafter cuts output to 0.30 to 0.49x of running without it at 64 requests.
- ❌ mistral.rs holds 4 requests at a time on 8 GB and stays at 258 to 309 tokens per second under load.

Scope: one RTX 3050 OEM (8 GB) in one desktop, Gemma 4 E2B on Google's QAT Q4_0 grid, one engine at a time, greedy decoding. One-request speed is six prompts of 256 tokens, three repeats; load is two synthetic shapes at 1 to 64 requests, five repeats each; accuracy is one pass of GSM8K (1,319) and BFCL simple (400). The engines ran on 2026-10-14 to 2026-11-04 and some on different llama.cpp or engine builds; same-day reference runs agree within 1%, and wall times from different days are indicative.

The strategy for comparing inference serving engines on one local GPU was validated with an incremental step by step approach.

---

#### References

- [gemma4-dev on GitHub](https://github.com/xbill9/gemma4-dev)
- [All six engines in one table: PAPER-TABLE.md](https://github.com/xbill9/gemma4-dev/blob/main/local-vllm-3050oem-2b-q4w4a16emb4/benchmarks/runs/2026-11-03-engines-3050oem/PAPER-TABLE.md)
- [SGLang patches for Gemma 4 text checkpoints](https://github.com/xbill9/gemma4-dev/tree/main/local-sglang-3050oem-2b-q4w4a16emb4/patches)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)
- [mistral.rs](https://github.com/EricLBuehler/mistral.rs)
- [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)
- [xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4)
- [xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf)
- [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)
- [GSM8K](https://github.com/openai/grade-school-math)
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
