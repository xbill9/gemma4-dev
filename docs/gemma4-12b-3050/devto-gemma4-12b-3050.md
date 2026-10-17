---
title: "Can a QAT 12B Fit an 8 GB GPU? Gemma 4 12B on an RTX 3050"
published: false
description: "Yes. An exact Q4_0 rebuild of Google's Gemma 4 12B QAT GGUF runs with every layer on an 8 GB RTX 3050 at up to 40,960 tokens of context, and Google's 12B drafter, rebuilt the same way, doubles its generation speed to 56 tok/s with llama.cpp."
tags: gemma, llamacpp, cuda, benchmarking
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/docs/gemma4-12b-3050/devto-cover.2bba6341.jpg
---

This article provides a step by step guide to serving Gemma 4 12B, Google's quantization-aware trained (QAT) build, on an 8 GB NVIDIA RTX 3050 with llama.cpp, with every layer on the GPU and Google's multi-token prediction (MTP) drafter for speculative decoding. A suite of Python MCP tools is built to simplify management of the llama.cpp hosted deployment.

This is the third part of a short series. [Part 1, Gemma 4's MTP Drafter, Explained](https://dev.to/gde/gemma-4s-mtp-drafter-explained-speculative-decoding-and-a-4-bit-drafter-at-60-of-the-size-1oc0), covers how the drafter works and how the exact Q4_0 repack is built. [Part 2, Do Drafters Work on Small Models?](https://dev.to/gde/do-drafters-work-on-small-models-gemma-4-e2b-and-e4b-on-two-small-gpus-1b4g), measures E2B and E4B on this card and a GTX 1650 Ti. This part moves up to the 12B.

https://github.com/xbill9/gemma4-dev

**Can a QAT 12B fit an 8 GB GPU? Yes**, with every layer on the card and room for a working context:

| RTX 3050 OEM, 8 GB | Google's 12B Q4_0 GGUF | exact Q4_0 rebuild |
| :--- | ---: | ---: |
| File size | 6.98 GB | **6.72 GB** |
| Most context that loads | 28,672 tokens | **40,960 tokens** |
| Generation | 27.7 tok/s | **28.6 tok/s** |
| With the 12B drafter | — | **56.4 tok/s** at 24,576 tokens of context |

The rebuild stores every weight on the 4-bit grid QAT trained, including the embedding table that Google's file keeps at 6 bits. That one table frees 247.5 MiB of the card, which buys 12,288 more tokens of context. Google's 12B drafter, rebuilt the same way, doubles generation.

---

#### What Is Being Measured?

Gemma 4 12B is a dense model with 48 layers and 11.9 billion parameters. Google publishes it already trained for 4-bit weights, as safetensors (`google/gemma-4-12B-it-qat-q4_0-unquantized`) and as a llama.cpp GGUF (`google/gemma-4-12B-it-qat-q4_0-gguf`). At 4 bits its weights take about 6.7 GB, which leaves an 8 GB card very little for anything else.

The questions are whether the whole model fits on the card, how much conversation fits beside it, how fast it generates, and what Google's 12B drafter adds. The card is the same RTX 3050 OEM that Part 2 measured with E2B and E4B.

---

#### At This Point You Should Have…

- An NVIDIA GPU with 8 GB and a CUDA driver — here an RTX 3050 OEM, driver 615.78.08
- The CUDA toolkit, 13.4 here, and `cmake`
- llama.cpp at a commit with Gemma 4 and `--spec-type draft-mtp` — every number here is from `7182ef71e`
- Python 3.10 or newer, the system `python3`, no virtualenv, with `numpy`
- About 40 GB of free disk for the source checkpoint and the build
- The repository cloned, with the rig `local-llamacpp-3050oem-12b-q4_0/`
- Claude Code, or any MCP client that speaks stdio

---

#### How the 12B Differs From E2B and E4B

| | E4B | 12B |
| :--- | ---: | ---: |
| Layers | 42 | 48 |
| Hidden size | 2,560 | 3,840 |
| Per-layer embedding table | yes, read from disk | **none** |
| Layers that own a KV cache | 24 of 42 | all 48 |
| Sliding window | 512 | 1,024 |
| Generation on the 3050, no drafter | 61.2 tok/s | 28.6 tok/s |

E2B and E4B carry a large per-layer embedding table that llama.cpp reads a few rows at a time from the memory-mapped file, so much of their file never reaches the card. The 12B has no such table: almost the whole file goes to the GPU. Its embedding table doubles as the output layer, so it sits on the card as well.

The 12B generates at 0.47x E4B's speed on the same card and llama.cpp build.

---

#### Step 1 — Size the Context Cache Before Downloading

The context cache (KV cache) decides how much conversation fits. Gemma 4 12B mixes two kinds of attention layers:

- **8 full-attention layers**, each with one key/value head of width 512. At 16-bit precision that is 8 × 512 × 2 (key and value) × 2 bytes = **16 KiB per token**.
- **40 sliding-window layers**, each with 8 key/value heads of width 256, which only look back 1,024 tokens. llama.cpp holds a fixed 1,536 positions for them (the window plus one 512-token batch): **480 MiB**, whatever the context.

So 40,960 tokens of context cost 640 + 480 MiB. Those are the figures llama.cpp's allocation log prints:

```
I load_tensors:        CUDA0 model buffer size =  6390.19 MiB
I llama_kv_cache:      CUDA0 KV buffer size =   640.00 MiB
I llama_kv_cache:      CUDA0 KV buffer size =   480.00 MiB
I sched_reserve:      CUDA0 compute buffer size =   161.52 MiB
```

With 6.4 GB of weights on an 8 GB card, every 247 MiB saved on the weights is about 15,800 tokens of context (arithmetic, at 16 KiB per token).

---

#### Step 2 — Rebuild the GGUF on Its Trained Grid

Google's 12B GGUF is 6.131 GB of Q4_0 and 0.826 GB of Q6_K. The Q6_K is a single tensor, `token_embd`, the 262,144 × 3,840 embedding table, at 787.5 MiB.

QAT trained that table on a 4-bit grid like everything else: in each block of 32 values, every value is a block step times an integer from −8 to 7. Q4_0 stores one fp16 step and a 4-bit level per value, the same shape. `gguf_exact.py` reads each block's trained step off its own values (the largest magnitude divided by 8, 7, … 1, whichever puts every value on an integer level, refined by least squares) and writes every weight matrix as Q4_0. It copies Google's metadata, tokenizer and chat template byte for byte.

```
hf download google/gemma-4-12B-it-qat-q4_0-unquantized --revision b6ed862 --local-dir src
hf download google/gemma-4-12B-it-qat-q4_0-gguf gemma-4-12b-it-qat-q4_0.gguf --revision 29d0977 --local-dir gguf
python3 gguf_exact.py src gguf/gemma-4-12b-it-qat-q4_0.gguf gemma-4-12B-it-q4_0-exact.gguf
```

It is the same script that built the E2B and E4B files, unchanged: the 12B's tensor names already match. It rebuilt 329 tensors in 17 minutes, and every block landed on the grid. Per tensor kind, 96.77–96.88% of values match Google's bf16 weights bit for bit once rounded back to bf16; the rest differ by the fp16 rounding of the step.

| | Google's GGUF | exact rebuild |
| :--- | ---: | ---: |
| File | 6,975,879,296 B | **6,716,356,736 B** |
| `token_embd` | Q6_K, 787.5 MiB | **Q4_0, 540.0 MiB** |
| Weights on the card | 6,637.69 MiB | **6,390.19 MiB** |

---

#### 🔎 Tip: Google's 12B GGUF Is Already Text Only

The vision encoder ships as a separate file, `mmproj-gemma-4-12b-it-qat-q4_0.gguf` (167 MiB). Leave it out and the model is text only, as the rebuild is. On an 8 GB card those 167 MiB are another 10,700 tokens of context (arithmetic).

---

#### Step 3 — Fetch the Published Files

Both rebuilds are on Hugging Face:

- [xbill9/gemma-4-12B-it-qat-q4_0-exact-gguf](https://huggingface.co/xbill9/gemma-4-12B-it-qat-q4_0-exact-gguf), file `gemma-4-12B-it-q4_0-exact.gguf`, SHA-256 `038337ee0375f602…`
- [xbill9/gemma-4-12B-it-qat-assistant-q4_0-exact-gguf](https://huggingface.co/xbill9/gemma-4-12B-it-qat-assistant-q4_0-exact-gguf), the drafter, file `gemma-4-12B-it-qat-assistant-q4_0-exact.gguf`, SHA-256 `f22f6366ca76102a…`

The rig's `make download` and `make draft` fetch them and check the hashes in `tpu.env`:

```
make download
make draft
/home/xbill/models/gemma-4-12B-it-qat-assistant/gemma-4-12B-it-qat-assistant-q4_0-exact.gguf: OK
```

---

#### Step 4 — Find the Most Context That Fits

`fit.sh` starts `llama-server -ngl 99 -fa 1` at rising context sizes and records the largest that loads, with the card otherwise idle:

| RTX 3050, 8,192 MiB | 16-bit cache | 8-bit cache (`-ctk q8_0 -ctv q8_0`) |
| :--- | ---: | ---: |
| Google's GGUF | 28,672 tokens, 7,814 MiB | not tested |
| exact rebuild | **40,960 tokens, 7,772 MiB** | **65,536 tokens, 7,550 MiB** |
| exact rebuild + drafter | **24,576 tokens, 7,820 MiB** | **49,152 tokens, 7,790 MiB** |

Google's file fails at 32,768 tokens; the rebuild loads there with 556 MiB to spare and fails at 49,152. The 8-bit cache halves every cache figure, sliding layers included, so it doubles the context at each setting. Its speed cost is unmeasured on the 12B.

The model's own limit is 131,072 tokens. On this card the 12B reaches a third of that with the 16-bit cache, where E4B fits the full 131,072 in 6,662 MiB.

---

#### Step 5 — Measure Speed Against Google's File

`bench.sh` runs `llama-bench`, five repetitions, in four passes that alternate which file goes first, with the card cooled to 40 °C before each:

| pass | file | prompt, 512 tokens | generation, 128 tokens |
| :--- | :--- | ---: | ---: |
| 1 | Google | 981.57 tok/s | 27.68 tok/s |
| 2 | exact | 987.49 tok/s | 28.64 tok/s |
| 3 | exact | 986.97 tok/s | 28.64 tok/s |
| 4 | Google | 977.83 tok/s | 27.68 tok/s |

**The rebuild generates 1.035x faster**, the same in both passes of each file, and reads prompts at the same speed. Every generated token reads the output layer in full, and the rebuild's output layer is 247.5 MiB smaller. The gain is smaller than E4B's 1.06x and E2B's 1.11x because the output layer is a smaller share of what a 12B reads per token (inferred, not measured).

Generation slows as the conversation grows:

| tokens already in context | 0 | 8,192 | 16,384 |
| :--- | ---: | ---: | ---: |
| prompt, 512 tokens | 981.09 tok/s | 801.58 tok/s | 692.32 tok/s |
| generation | 28.63 tok/s | 26.81 tok/s | 26.33 tok/s |

---

#### Step 6 — Check How Close It Stays to bf16

`run_kld.sh` scores the same text with Google's bf16 weights and with each 4-bit file, and compares the predicted next-token distributions (`llama-perplexity --kl-divergence`, wikitext-2, 16 chunks of 512 tokens, on the CPU):

| against bf16 | Google's GGUF | exact rebuild |
| :--- | ---: | ---: |
| Mean KL divergence | 0.543 | **0.125** |
| 99th percentile KL divergence | 6.217 | **1.915** |
| Same top token | 71.77% | **89.14%** |
| Perplexity, relative to bf16 | 1.386 | **1.257** |

**The rebuild is 4.4x closer to bf16 by mean divergence**, and picks the same top token 89% of the time against 72%.

Compare the two columns with each other only. The 12B instruction-tuned model reads raw Wikipedia text poorly, at a bf16 perplexity of 312 against E4B's 40 on the same test, and every divergence is larger in that regime. E4B's rebuild scores 0.0010 on this test.

---

#### Step 7 — Rebuild the 12B Drafter

Google's drafter for this model is [google/gemma-4-12B-it-qat-q4_0-unquantized-assistant](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized-assistant): 4 layers of width 1,024 with queries only, reading the 12B's own KV cache, and a tied 262,144-token embedding. It ships as bf16 on a 4-bit grid.

llama.cpp's converter writes a Q8_0 file, which the drafter version of `gguf_exact.py` uses for metadata:

```
python3 convert_hf_to_gguf.py hf --outtype q8_0 --outfile gemma-4-12B-it-qat-assistant-q8_0.gguf
python3 gguf_exact.py hf gemma-4-12B-it-qat-assistant-q8_0.gguf gemma-4-12B-it-qat-assistant-q4_0-exact.gguf
```

Every block landed on the grid. The rebuild is 253,708,960 bytes against 465,128,096 for Q8_0, and its weights take 226.90 MiB on the card against 428.53 MiB. This drafter is 4.7x the size of the E2B and E4B drafters.

---

#### Step 8 — Measure the Drafter

The rig serves with four extra flags:

```
llama-server -m gemma-4-12B-it-q4_0-exact.gguf -ngl 99 -c 24576 -fa 1 --reasoning off \
  -md gemma-4-12B-it-qat-assistant-q4_0-exact.gguf --spec-type draft-mtp -ngld all --spec-draft-n-max 2
```

`spec_ab.py` runs six prompts (code, advice, a story, an explanation, a list, a summary), 256 tokens each, three greedy repeats plus one sampled pass, one server per setting at `-c 8192`, the card cooled before each. Serving without a drafter ran first and last, at 28.1 tok/s both times.

| greedy, geometric mean of 6 prompts | tok/s | gain | drafted tokens kept | memory |
| :--- | ---: | ---: | ---: | ---: |
| no drafter | 28.1 | — | — | 7,255 MiB |
| Q8_0 drafter, 2 tokens | 53.4 | 1.90x | 0.661 | 7,771 MiB |
| **exact Q4_0 drafter, 2 tokens** | **56.4** | **2.01x** | **0.658** | **7,569 MiB** |
| exact Q4_0 drafter, 1 token | 45.6 | 1.62x | 0.759 | 7,569 MiB |
| exact Q4_0 drafter, 3 tokens | 56.7 | 2.02x | 0.561 | 7,569 MiB |

Per prompt at 2 tokens: code 2.40x, a list of facts 2.30x, advice 1.94x, an explanation 1.93x, a summary 1.82x, a story 1.75x. Sampled at temperature 0.8: 1.90x.

**The drafter doubles the 12B's generation speed.** The gain is larger than E4B's 1.72x on this card in Part 2: the 12B keeps 66% of drafted tokens against E4B's 48%.

---

#### 🔎 Tip: For the 12B, the 4-Bit Drafter Is Faster Than the 8-Bit One

For E2B and E4B the Q4_0 and Q8_0 drafters ran level. For the 12B the Q4_0 drafter is 1.06x faster (1.04–1.07x on each prompt), at the same share of tokens kept, and uses 202 MiB less memory. This drafter is 4.7x larger, so its own weight reads are likely a larger share of each drafted step (inferred, not measured). On an 8 GB card the 202 MiB is also about 12,900 tokens of context (arithmetic).

---

#### 🔎 Tip: Greedy Text Changes With the Drafter

With the drafter on, greedy replies repeat exactly from run to run, but differ from the replies without a drafter in all 18 at 2 drafted tokens and 9 of 18 at 1. Checking several tokens in one pass runs different GPU kernels, which can change a near-tie word. Step 9 checks whether the answers change.

---

#### Step 9 — Check the Answers

The rig's own settings, one request at a time: GSM8K (1,319 grade-school math questions, up to 2,048 tokens per reply) and BFCL simple (400 single function calls), with the drafter and without. E4B's scores are from Part 2's run of the same tests on the same card.

| one request at a time | 12B, drafter | 12B, no drafter | E4B, drafter | E4B, no drafter |
| :--- | ---: | ---: | ---: | ---: |
| GSM8K | **0.960** | **0.963** | 0.930 | 0.931 |
| BFCL simple | **0.950** | **0.948** | 0.913 | 0.913 |

**The 12B answers more questions than E4B on both tests.** Paired question by question, it is 3.0 points ahead on GSM8K (95% range 1.8 to 4.3) and 3.8 points ahead on BFCL (1.0 to 6.5), both with the drafter.

**The drafter changes no score.** On GSM8K the runs with and without it differ by 0.3 points (95% range −0.8 to +0.2), on BFCL by one question of 400. The greedy text differs, as in Step 8: 332 of 1,319 GSM8K replies match word for word. Without the drafter GSM8K took 2.34x as long, 4 h 16 min against 1 h 49 min.

---

#### Compare and Contrast

| RTX 3050 OEM, 8 GB | 🥇 Gemma 4 12B | 🥈 Gemma 4 E4B |
| :--- | :--- | :--- |
| File | 6.72 GB | 4.22 GB |
| Generation, no drafter | 28.6 tok/s | 61.2 tok/s |
| Generation, 2-token drafter | 56.4 tok/s | 100.0 tok/s (Part 2) |
| Drafter gain | 2.01x | 1.72x |
| Context with the drafter, 16-bit cache | 24,576 tokens | 131,072 tokens |
| GSM8K, with the drafter | 0.960 | 0.930 |
| BFCL simple, with the drafter | 0.950 | 0.913 |

E4B's numbers are from llama.cpp `77c4540ce` (Part 2) except the no-drafter generation, measured beside the 12B on `7182ef71e`.

---

#### So, Which One?

For answers, the 12B. It scores 3 to 4 points higher than E4B on math and on function calls, and with the drafter it generates at 56.4 tok/s, near E4B's 61.2 without one.

For long conversations or speed, E4B. It holds the full 131,072-token context on the same card and generates at 100 tok/s with its drafter, against the 12B's 24,576 tokens and 56.4 tok/s.

Either way, serve with the drafter: it doubles the 12B's speed and costs no answers.

---

#### What Comes Next

Google publishes the same QAT drafters for 26B-A4B and 31B, both too large for this card. The next measurements are the 12B under several requests at once, where a 24,576-token context leaves little room to share, and the 12B on a 16 GB card with its full context.

---

#### Summary

The goal of this article was to find out whether Gemma 4 12B, in Google's QAT build, runs on an 8 GB RTX 3050 with every layer on the GPU. It does. The key to the solution was rebuilding Google's GGUF with every weight on the 4-bit grid QAT trained, which frees the memory a working context needs, and rebuilding the 12B drafter the same way. The results were:

- 🟢 The exact Q4_0 rebuild fits with every layer on the card at up to 40,960 tokens of context, against 28,672 for Google's file
- 🟢 It generates 1.035x faster than Google's file and stays 4.4x closer to bf16 by mean divergence
- 🟢 Google's 12B drafter, rebuilt as exact Q4_0, doubles generation to 56.4 tok/s, and runs 1.06x faster than the Q8_0 conversion
- 🟢 The 12B answers more questions than E4B on the same card: GSM8K 0.960 against 0.930, BFCL 0.950 against 0.913, and the drafter changes neither score
- ⚠️ With the drafter, the context that fits drops to 24,576 tokens at 16-bit, or 49,152 with an 8-bit cache
- ⚠️ The 12B generates at 0.47x E4B's speed on the same card
- ❌ The full 131,072-token context does not fit on 8 GB

Scope: Gemma 4 12B-it QAT only, text only, one RTX 3050 OEM (8 GB) in an i7-5820K desktop, llama.cpp `7182ef71e`, Google's 12B Q4_0 GGUF against the exact Q4_0 rebuild, Google's 12B QAT drafter as exact Q4_0 and Q8_0, six short prompts for speed, divergence on 8,192 tokens of wikitext-2 on the CPU, GSM8K and BFCL simple one request at a time, one request at a time throughout. E4B comparisons from Part 2 used llama.cpp `77c4540ce`.

The strategy for using MCP for local accelerator comparison was validated with an incremental step by step approach.

---

#### References

- Part 1, Gemma 4's MTP Drafter, Explained: https://dev.to/gde/gemma-4s-mtp-drafter-explained-speculative-decoding-and-a-4-bit-drafter-at-60-of-the-size-1oc0
- Part 2, Do Drafters Work on Small Models?: https://dev.to/gde/do-drafters-work-on-small-models-gemma-4-e2b-and-e4b-on-two-small-gpus-1b4g
- Repository: https://github.com/xbill9/gemma4-dev
- 12B rig: https://github.com/xbill9/gemma4-dev/tree/main/local-llamacpp-3050oem-12b-q4_0
- The rebuild and divergence: https://github.com/xbill9/gemma4-dev/blob/main/local-llamacpp-3050oem-12b-q4_0/benchmarks/runs/2026-10-16-exact-gguf-12b-3050oem/REPORT.md
- Memory and speed: https://github.com/xbill9/gemma4-dev/blob/main/local-llamacpp-3050oem-12b-q4_0/benchmarks/runs/2026-10-16-first-light-3050oem/REPORT.md
- The drafter: https://github.com/xbill9/gemma4-dev/blob/main/local-llamacpp-3050oem-12b-q4_0/benchmarks/runs/2026-10-16-drafter-q4exact-12b-3050oem/REPORT.md
- Answers: https://github.com/xbill9/gemma4-dev/blob/main/local-llamacpp-3050oem-12b-q4_0/benchmarks/runs/2026-10-16-accuracy-12b-3050oem/REPORT.md
- 12B GGUF: https://huggingface.co/xbill9/gemma-4-12B-it-qat-q4_0-exact-gguf
- 12B drafter: https://huggingface.co/xbill9/gemma-4-12B-it-qat-assistant-q4_0-exact-gguf
- Google's 12B QAT GGUF: https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf
- llama.cpp: https://github.com/ggml-org/llama.cpp
