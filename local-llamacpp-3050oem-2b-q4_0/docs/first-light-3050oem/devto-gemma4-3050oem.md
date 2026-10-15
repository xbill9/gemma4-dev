---
title: "Gemma 4 on a Used 8 GB RTX 3050: 116 Tokens/s for One User, 618 for 64"
published: false
description: "Gemma 4 E2B and E4B on a used RTX 3050 OEM in an old desktop, step by step: single-user speed, the full 131k context, concurrency, answer quality, and how it compares on price with laptop GPUs, ARM laptops, a T4 and a TPU v5e."
tags: gemma, llamacpp, cuda, localllm
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/local-llamacpp-3050oem-2b-q4_0/docs/first-light-3050oem/devto-cover.c84e6ee4.jpg
---

This article provides a step by step deployment guide for Gemma 4 E2B and E4B to a local, desktop hosted GPU enabled system: an RTX 3050 OEM with 8 GB, in a 2014-era Haswell-E desktop. A suite of Python MCP tools is built to simplify management of the llama.cpp hosted deployment.

https://github.com/xbill9/gemma4-dev/tree/main/local-llamacpp-3050oem-2b-q4_0

**A used RTX 3050 OEM, listed at $179.97, serves Gemma 4 E2B at 116 tokens per second for one user and 618 tokens per second across 64 users, with the model's full 131,072-token context in 3.6 GB of its 8 GB. Against an AWS T4 on demand, the card costs as much as 342 hours of rental.**

---

#### What is this project trying to Do?

Earlier rigs in this repository served Gemma 4 from rented TPUs and GPUs, and from hardware already on the desk: a 2021 laptop with a 4 GB GTX 1650 Ti, its CPU, and two Snapdragon ARM laptops. This rig asks what the cheapest dedicated card adds. The RTX 3050 OEM is an 8 GB Ampere card with tensor cores, and used ones were listed on eBay at about $145 to $210 on 2026-10-14.

The rig serves the same model file as the 1650 Ti rig, byte for byte, with the same server flags. The questions are single-user speed, how much context fits, how many users one card carries, whether the bigger E4B model is worth its cost, and what all of it costs against the alternatives.

Speculative decoding with Google's drafter models is a separate article. Every figure here was measured without it.

---

#### At This Point You Should Have…

- An Nvidia GPU with 8 GB of memory. This one is an RTX 3050 OEM: GA106, compute capability 8.6, tensor cores, 120 W power limit.
- The CUDA toolkit (13.4 here) and `cmake`.
- About 7 GB of disk for both model files.
- A clone of `xbill9/gemma4-dev`.

---

#### The Hardware

```text
power.limit [W], clocks.max.sm [MHz], clocks.max.memory [MHz]
120.00 W, 2100 MHz, 7001 MHz
```

| | |
| :--- | :--- |
| GPU | NVIDIA GeForce RTX 3050 OEM, PCI `10de:2508` (GA106), compute capability 8.6, 8192 MiB |
| Driver | 615.78.08 |
| CPU | Intel i7-5820K, 6 cores / 12 threads, AVX2 |
| Board / RAM | ASUS X99-A, 46 GiB |
| OS | Debian sid, gcc 16.2.0, CUDA 13.4 |

The OEM 3050 came in prebuilt desktops and is a different configuration from the retail RTX 3050 8 GB. The host is a 2014-era X99 board; with every layer on the card, the CPU barely matters.

---

#### Step 1 — Build llama.cpp for the Card

```shell
export PATH=/usr/local/cuda/bin:$PATH
make build
```

The target runs `cmake` with `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=86` and `GGML_NATIVE=ON`, then builds `llama-server` and `llama-bench`. It printed the commit, `77c4540ce`, which goes into `tpu.env` as `LLAMA_CPP_COMMIT`. The build took 10 minutes 24 seconds at `-j10`.

---

#### Step 2 — Download the Exact QAT GGUF

```shell
make download
```

The target downloads the file and checks it against `MODEL_SHA256` in `tpu.env`. The file is `xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf`, SHA-256 `419db9a6bf3bc15d770c85ccf9216827aa88c64fe8f3d72fbe5674f48efe2dc8`. It is Google's quantization-aware-trained (QAT) Gemma 4 E2B with every trained 4-bit value packed into Q4_0 exactly. On the 1650 Ti it measured 32x closer to bf16 by KL divergence than Google's own GGUF, and faster.

---

#### Step 3 — Configure the Server

`tpu.env` is the rig's configuration, and every value in it carries the measurement that set it:

```shell
N_GPU_LAYERS=99
CONTEXT_SIZE=131072
KV_CACHE_TYPE=f16
FLASH_ATTENTION=1
THREADS=6
PARALLEL_SLOTS=1
REASONING=off
OVERRIDE_TENSORS=per_layer_token_embd\.weight=CUDA0
```

| Setting | Why |
| :--- | :--- |
| `-ngl 99` | every layer on the card |
| `-c 131072` | the model's trained maximum; Step 8 |
| `-fa 1` | flash attention: 1.133x prefill, 1.071x decode |
| `-ctk f16 -ctv f16` | `q8_0` costs 7.4% of decode |
| `-ot per_layer_token_embd…=CUDA0` | the per-layer embedding table on the card; Step 7 |
| `--reasoning off` | answers start at once; a request can still ask for thinking |

---

#### Step 4 — Start It and Check What Is Serving

```shell
make serve
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

```text
pid, process_name, used_gpu_memory [MiB]
47321, /home/xbill/llama.cpp/build/bin/llama-server, 1510 MiB
```

That is the first-light configuration: 8,192 tokens of context and the per-layer table left in host memory. The MCP server's attestation reads the device from the running process, since a CPU-only build would answer on the same port:

```text
device=gpu · pid=47321 · -ngl 99 · mapped: ggml-cuda, libcublas, libcuda, libcudart · /home/xbill/llama.cpp/build/bin/llama-server
mismatch: None
```

One chat request at temperature 0, "In one sentence, what is a TPU?":

```text
A TPU, or Tensor Processing Unit, is a specialized hardware accelerator designed to significantly speed up the training and inference of machine learning models, particularly those involving deep learning.
```

The server's own timings for that request: 34 tokens at 102.56 tokens per second.

---

#### Step 5 — Measure Single-User Speed

`llama-bench`, a 512-token prompt and 128 generated tokens, five repetitions per cell, the card cooled to 40 °C before each cell:

```text
fa1-f16-t6 pp512=3859.96±111.73 tg128=116.14±0.25 77c4540ce
fa0-f16-t6 pp512=3408.19±110.65 tg128=108.46±0.22 77c4540ce
fa1-q8-t6 pp512=3708.48±118.09 tg128=107.57±0.23 77c4540ce
fa1-f16-t4 pp512=3838.78±111.16 tg128=115.87±0.22 77c4540ce
fa1-f16-t6-rep pp512=3835.54±102.85 tg128=115.90±0.22 77c4540ce
```

**116.14 tokens per second generated, 3,860 tokens per second of prompt read.** The repeated last cell lands within 0.6% of the first, and the card held 1890 to 1905 MHz without passing 44 °C.

On the 1650 Ti the same file generates at about 81.7 tokens per second and reads prompts at about 345. That makes this card about 1.42x faster at generating and 11x faster at reading. Generating one token reads every weight once, so it follows memory speed, where the two cards are closer. Reading a prompt is matrix work, which this card runs on tensor cores and the 1650 Ti has none of.

---

#### Step 6 — Keep the Defaults That Measure Faster

| Setting | Prompt | Generate | Kept |
| :--- | ---: | ---: | :--- |
| flash attention off | 0.883x | 0.934x | ❌ on |
| `q8_0` KV cache | 0.961x | 0.926x | ❌ f16 |
| 4 CPU threads in place of 6 | 0.995x | 0.998x | ⚠️ either |

The KV cache for E2B is small, so quantizing it saves memory the card has plenty of and costs speed. Thread count makes no difference because the CPU does no model work.

---

#### Step 7 — Move the Per-Layer Embedding Table onto the Card

The `E` in E2B is a per-layer embedding table, `per_layer_token_embd`, 1.32 GB and 51% of the file. llama.cpp leaves it memory-mapped in host RAM by default and reads one row per token. With 6 GB of the card unused, the rig moves it with `-ot`. Card against host, mean of two passes each, order host, card, card, host:

| test | host | card | card / host |
| :--- | ---: | ---: | ---: |
| pp512 | 3845.18 | 3982.67 | 1.036 |
| pp2048 | 3759.36 | 3848.42 | 1.024 |
| tg128 | 115.91 | 115.98 | 1.001 |

A 2.4 to 3.6% gain on prompts, nothing on generation, for 1,260 MiB. Both card passes beat both host passes at both prompt lengths. The 1650 Ti measured no gain from the same move: this card reads prompts 11x faster, so fetching rows from host memory is a bigger share of its work.

---

#### Step 8 — Serve the Full 131k Context

Gemma 4 E2B was trained to 131,072 tokens. Only its 3 full-attention layers keep a cache that grows with context; the 12 sliding-window layers stay at 1,024 cells.

| `-c` | KV, full-attention | KV, sliding | process on the card |
| ---: | ---: | ---: | ---: |
| 8,192 | 48 MiB | 12 MiB | 2770 MiB |
| 32,768 | 192 MiB | 12 MiB | 2938 MiB |
| 131,072 | 768 MiB | 12 MiB | **3610 MiB** |

The full context costs 840 MiB over 8,192 and leaves 4.5 GB of the card free. Short chats do not notice: 256 generated tokens ran at 108.30 tokens per second at 8,192 and 107.99 at 131,072.

Long conversations do slow down, because every new token attends to everything before it:

| tokens already in the conversation | prompt t/s | generate tok/s | vs empty |
| ---: | ---: | ---: | ---: |
| 0 | 3959.63 | 116.02 | 1.000 |
| 8,192 | 2898.48 | 107.06 | 0.923 |
| 32,768 | 1690.19 | 90.68 | 0.782 |
| 65,536 | 1079.92 | 75.02 | 0.647 |
| 120,000 | 669.25 | 58.14 | 0.501 |

Generation stays over 100 tokens per second to about 8k tokens of conversation and halves by 120k.

---

#### Step 9 — Serve Many Users at Once

For the concurrency sweep the server ran 64 slots of 1,024 tokens (`-c 65536 --parallel 64`), 2612 MiB on the card. `sweep.py` sent 1 to 64 simultaneous requests, each with a unique prompt, five repeats per level, median reported:

```text
c=1      93.28 tok/s aggregate  TTFT    212.7 ms  TPOT   9.34 ms  per-stream 107.03
c=4     196.09 tok/s aggregate  TTFT    669.1 ms  TPOT  15.64 ms  per-stream  63.95
c=16    316.07 tok/s aggregate  TTFT   2702.4 ms  TPOT  30.12 ms  per-stream  33.20
c=64    358.05 tok/s aggregate  TTFT   8516.4 ms  TPOT 113.20 ms  per-stream   8.84
```

That is 512 tokens in and 128 out. With 128 in and 512 out, chat-shaped traffic:

```text
c=1     104.74 tok/s aggregate  TTFT    134.9 ms  TPOT   9.34 ms  per-stream 107.02
c=4     259.73 tok/s aggregate  TTFT    330.3 ms  TPOT  14.74 ms  per-stream  67.83
c=16    505.42 tok/s aggregate  TTFT   1114.3 ms  TPOT  29.32 ms  per-stream  34.10
c=64    618.12 tok/s aggregate  TTFT   3890.0 ms  TPOT  95.33 ms  per-stream  10.49
```

**3.84x and 5.90x the single-user output at 64 users.** Past 16 users the gain flattens while each user's speed keeps falling, so 16 is the practical ceiling: 34 tokens per second each, faster than anyone reads.

llama.cpp still reads prompts one request at a time, so time to first token about doubles with each doubling of users. The 1650 Ti ran into that wall: at 512 in and 128 out its output reached only 1.47x single-user. This card reads prompts fast enough that batched generation shows through anyway.

The card drew at most 110.93 W of its 120 W limit, held 1890 MHz in every busy sample and peaked at 68 °C. The server's prompt cache answered 0 tokens, so every prompt was read in full.

---

#### Step 10 — Bring Up E4B on the Same Card

`local-llamacpp-3050oem-4b-q4_0` is the E4B sister rig: same card, binary and flags, with the exact Q4_0 E4B file (SHA-256 `5462cc10…`, 4.20 GB).

| at 131,072 context | E2B | E4B |
| :--- | ---: | ---: |
| Process on the card | 3610 MiB | 6430 MiB |
| KV cache | 780 MiB | 2,088 MiB |
| Generate, empty context | 116.02 tok/s | 60.41 tok/s |
| Prompt, empty context | 3959.63 t/s | 2230.50 t/s |
| Generate, 120k in context | 58.14 tok/s | 30.04 tok/s |

E4B generates at 0.52x E2B at every conversation length and fits with 1.7 GB to spare. The two rigs share port 8080 and do not fit on the card together. On the 1650 Ti, E4B fits at 8,192 context with the per-layer table in host memory and generates at about 39.8 tokens per second; this card runs it 1.52x faster with 16x the context.

---

#### Step 11 — Measure Answer Quality

Both files served one at a time with identical flags, no drafter, and three task sets: a 13-subset classification suite (3,880 questions), GSM8K math (1,319 problems) and BFCL simple tool calling (400 records), all greedy.

| task | E2B | E4B | E4B − E2B | 95% range |
| :--- | ---: | ---: | ---: | :--- |
| Classification suite | 0.683 | **0.730** 🥇 | +0.048 | +0.035 to +0.061 |
| GSM8K | 0.898 | **0.933** 🥇 | +0.035 | +0.020 to +0.050 |
| BFCL simple | **0.925** | 0.910 | −0.015 | −0.043 to +0.013 |

E4B is better at reading comprehension and math, and level on tool calling, where the range includes zero. Both 4-bit files score within the error ranges of their bf16 references measured on Cloud TPU: E2B suite 0.683 against bf16's 0.683, E4B suite level with its bf16 run. The quantization costs nothing measurable; the gap between the two is the model.

The whole suite took 6.4 minutes for E2B and 11.2 for E4B on this card.

---

#### 🔎 Tip: Spend the Spare Memory on Context

On an 8 GB card E2B leaves more than half the memory free even at full context. Spend it on the 131k context and on the per-layer table, both measured gains or free. Quantizing the KV cache to `q8_0` saves memory the card does not need and costs 7.4% of generation.

---

#### 🔎 Tip: Size `--parallel` Against `-c`

llama.cpp splits `--ctx-size` across `--parallel` slots. `-c 131072 --parallel 16` gives each user 8,192 tokens; `-c 16384 --parallel 16` gives each 1,024. Set both together, from what each user needs.

---

#### Compare and Contrast: Local Hardware

Gemma 4 E2B, the same exact Q4_0 file on every row except the last two, single user, `llama-bench` generation speed:

| Machine | Device | Generate tok/s | vs 3050 |
| :--- | :--- | ---: | ---: |
| Desktop, RTX 3050 OEM 8 GB | GPU, 120 W limit | **116.14** 🥇 | 1.00x |
| Lenovo Yoga 9, GTX 1650 Ti Max-Q 4 GB | GPU, 40 W limit | 81.75 🥈 | 0.70x |
| Lenovo Yoga 9, i7-10750H | CPU, 6 threads | 18.84 🥉 | 0.16x |
| Samsung Galaxy Book Go, Snapdragon 7c Gen 2 | CPU, 2 cores | 8.33 | 0.07x |
| Lenovo Yoga C630, Snapdragon 850 | CPU, 4 cores | 5.00 | 0.04x |

| Capability | RTX 3050 OEM | GTX 1650 Ti |
| :--- | :--- | :--- |
| Prompt read, 512 tokens | 3,860 t/s 🥇 | about 345 t/s |
| Full 131k context, E2B | 🟢 3.6 GB of 8 | ⚠️ not measured |
| E4B | 🟢 131k context, 60 tok/s | ⚠️ 8k context, 40 tok/s |
| 512 in / 128 out, best aggregate | 🟢 358 tok/s, 3.84x | ❌ 1.47x single user |
| Power under load | 110.93 W peak | 40 W cap |

---

#### Compare and Contrast: Cloud

The two cheapest cloud deployments of Gemma 4 E2B measured in this repository serve bf16 weights through vLLM: an AWS `g4dn.xlarge` with a T4, and one Google Cloud TPU v5e chip.

| Operating point | RTX 3050 OEM | T4, `g4dn.xlarge` | TPU v5e-1 |
| :--- | ---: | ---: | ---: |
| One user, output tok/s | 93.28 🥈 | 42.36 🥉 | 120 🥇 |
| 8 users, 512 in / 128 out | 224.18 | 242.47 | — |
| Best aggregate measured | 618.12 🥈 | 243.93 🥉 | 1,382 🥇 |
| Price | $179.97 once | $0.526/hour | $0.5779/hour, spot |

The one-user row compares like shapes for the 3050 and T4 (512 in, 128 out); the TPU figure is 128 in, 128 out. The TPU is faster at every point and serves 2.2x the 3050's best aggregate. The 3050 generates 2.2x the T4's speed for one user and lands within 8% of it at 8 users.

---

#### And Price/Performance?

All of this is arithmetic from the measured speeds, the cloud rates recorded in those runs, a used-card listing captured on 2026-10-14, and the US Energy Information Administration's 2026 residential electricity forecast of 18.2 cents per kWh.

**The card pays for itself in two to three weeks of continuous rental.**

| Against | Rate | Hours to $179.97 | Days |
| :--- | ---: | ---: | ---: |
| TPU v5e-1, spot | $0.5779/h | 311 | 13.0 |
| T4 `g4dn.xlarge`, on-demand | $0.526/h | 342 | 14.3 |
| T4 `g4dn.xlarge`, spot | $0.3559/h | 506 | 21.1 |

**Running it costs about two cents an hour.** At 110.93 W, the most the card drew in any sample, electricity is $0.0202 per hour against $0.526 for the T4.

| Per million output tokens | Cost |
| :--- | ---: |
| RTX 3050, 64 users, 128 in / 512 out, electricity at peak draw | **$0.009** 🥇 |
| RTX 3050, 8 users, 512 in / 128 out, electricity at peak draw | $0.025 🥈 |
| RTX 3050, one user, electricity at 80 W | $0.039 🥉 |
| TPU v5e-1, 64 users, spot | $0.116 |
| T4, 8 users, on-demand | $0.603 |
| TPU v5e-1, one user, spot | $1.338 |
| T4, one user, on-demand | $3.449 |

The local rows leave out the card's purchase and the host it sits in; the cloud rows leave out storage and data transfer. A card already paid for wins by an order of magnitude on running cost. The TPU is the faster machine and the better buy only for a burst of work too big for one 3050, or for anyone without a desktop to put the card in.

---

#### So, Which One?

For a single person or a small team running Gemma 4 E2B or E4B all day, the used RTX 3050 OEM. It is the only local card here with tensor cores and 8 GB, and both together are what it takes: E4B at full context fits, prompts read at thousands of tokens per second, and 16 users each still get 34 tokens per second.

The 1650 Ti laptop remains the right answer for one user who already owns it. The ARM laptops serve E2B, slowly. Rent the TPU for bursts of work.

Between the two models, E4B buys 4.8 points on the classification suite and 3.5 on math for half the generation speed. For tool calling, E2B is level and twice as fast.

---

#### Summary

The goal of this article was to bring Gemma 4 E2B and E4B up on a used RTX 3050 OEM and to measure what it delivers for the money against the other local and cloud deployments in this repository. The key to the solution was the exact Q4_0 QAT files, which leave enough of the 8 GB card free for the full context and for many users at once. The measured results were:

- 🟢 **116.14 tok/s generated and 3,860 t/s of prompt read** for one user, E2B, 1.42x and 11x the 1650 Ti.
- 🟢 **The full 131,072-token context in 3610 MiB** for E2B and 6430 MiB for E4B, with no cost to short chats.
- 🟢 **618.12 tok/s aggregate at 64 users** (128 in / 512 out) and 358.05 at 512 in / 128 out; 34 tok/s per user at 16.
- 🟢 **4-bit quality matches bf16**: E2B suite 0.683 against 0.683; E4B 0.730.
- 🟢 **Pays for itself in 311 to 506 hours** against the cheapest cloud options, and runs for about $0.02 an hour in electricity (arithmetic).
- ⚠️ **E4B generates at 0.52x E2B** for +4.8 suite and +3.5 GSM8K points.
- ⚠️ **Long conversations halve generation speed** by 120k tokens of context.
- ❌ **A `q8_0` KV cache and flash attention off are both slower** on this card.

Scope: one RTX 3050 OEM in an i7-5820K desktop, llama.cpp `77c4540ce` with CUDA 13.4, measured 2026-10-13, without speculative decoding. The 1650 Ti, i7-10750H and Snapdragon figures come from other hosts, llama.cpp commits and days with the same model file; the T4 and TPU figures come from vLLM serving bf16 weights under a different benchmark tool; so every cross-machine ratio is indicative. The card price is one used listing's asking price on 2026-10-14, and the electricity figures use the highest power draw sampled or the single-user benchmark's end-of-run reading.

The strategy for running Gemma 4 locally on a used RTX 3050 OEM was validated with an incremental step by step approach.

---

#### References

* [local-llamacpp-3050oem-2b-q4_0 | GitHub](https://github.com/xbill9/gemma4-dev/tree/main/local-llamacpp-3050oem-2b-q4_0)
* [local-llamacpp-3050oem-4b-q4_0 | GitHub](https://github.com/xbill9/gemma4-dev/tree/main/local-llamacpp-3050oem-4b-q4_0)
* [xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf | Hugging Face](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf)
* [xbill9/gemma-4-E4B-it-qat-q4_0-exact-gguf | Hugging Face](https://huggingface.co/xbill9/gemma-4-E4B-it-qat-q4_0-exact-gguf)
* [Short-Term Energy Outlook, electricity | U.S. EIA](https://www.eia.gov/outlooks/steo/report/elec_coal_renew.php)
* [llama.cpp | GitHub](https://github.com/ggml-org/llama.cpp)
