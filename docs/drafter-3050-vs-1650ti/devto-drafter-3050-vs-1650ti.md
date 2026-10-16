---
title: "Gemma 4's Drafter on Two Small GPUs: 1.72x on an RTX 3050, 1.49x on a GTX 1650 Ti"
published: false
description: "Google's QAT assistant drafters for Gemma 4 E2B and E4B, served by llama.cpp with speculative decoding on an 8 GB RTX 3050 and a 4 GB laptop GTX 1650 Ti, same build, same prompts, same drafter files. Both cards keep the same share of drafted tokens; the older card gets less speed from them."
tags: gemma, llamacpp, cuda, benchmarking
---

This article provides a step by step guide to serving Gemma 4 E2B and E4B with Google's QAT assistant drafters through llama.cpp's speculative decoding, and measuring what the drafter is worth on two small NVIDIA cards: a desktop RTX 3050 OEM and a laptop GTX 1650 Ti. A suite of Python MCP tools is built to simplify management of the llama.cpp hosted deployment.

https://github.com/xbill9/gemma4-dev

One request at a time, the drafter speeds up E4B **1.72x** on the RTX 3050 and **1.49x** on the GTX 1650 Ti, and E2B **1.40x** and **1.18x**. Both cards keep the same share of the drafted tokens, within 0.03, so the difference comes from the cards. Answer quality is unchanged on both. With several requests at once the gain shrinks on both cards, and on the 1650 Ti it turns into a loss from 4 requests (E4B) or 8 (E2B).

---

#### What Is Being Measured?

A drafter is a small model that guesses the next few tokens. The main model then checks all the guesses in one step and keeps the ones it would have produced itself. When most guesses are kept, one step of the main model yields several tokens instead of one.

Google publishes an assistant drafter for each Gemma 4 QAT checkpoint. Both cards here serve the exact Q4_0 GGUFs of `gemma-4-E2B-it` and `gemma-4-E4B-it` with the drafters stored back as Q4_0 on their trained 4-bit grid, 59 MB each. The question is what the drafter is worth on each card, and why the two cards differ.

---

#### At This Point You Should Have…

- An NVIDIA GPU with a CUDA driver — here an RTX 3050 OEM (driver 615.78.08) and a GTX 1650 Ti with Max-Q Design (driver 615.71.09)
- The CUDA toolkit, 13.4 here, and `cmake`
- llama.cpp at a commit with `--spec-type draft-mtp` — both cards ran `77c4540ce` for every comparison below
- Python 3.10 or newer, the system `python3`, no virtualenv
- The repository cloned, with the four rigs `local-llamacpp-{3050oem,1650ti}-{2b,4b}-q4_0/`
- Claude Code, or any MCP client that speaks stdio

---

#### The Two Cards

| | RTX 3050 OEM | GTX 1650 Ti Max-Q |
| :--- | :--- | :--- |
| Host | i7-5820K desktop, 46 GiB | i7-10750H laptop (Lenovo Yoga 9), 15 GiB |
| Memory | 8192 MiB | 4096 MiB |
| Architecture | Ampere, compute capability 8.6, tensor cores | Turing TU117, compute capability 7.5, no tensor cores |
| Power limit | 120 W | about 40 W |
| E2B, no drafter | 108.2 tok/s | 76.5 tok/s |
| E4B, no drafter | 58.3 tok/s | 38.0 tok/s |

Without a drafter the 3050 generates 1.41x faster on E2B and 1.53x on E4B. Different hosts, so treat ratios between the cards as indicative.

---

#### Step 1 — Repack the Drafter on Its Trained Grid

Google publishes the QAT assistant drafters as [google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant) and [google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant). Each is a 4-layer model that reads the main model's KV cache, stored as bf16 safetensors.

Quantization-aware training already put the weights on a 4-bit grid: in every group of 32 values there are at most 16 distinct levels, evenly spaced. bf16 spends 16 bits storing each of those levels.

llama.cpp's `convert_hf_to_gguf.py` has no way to know that, so it writes Q8_0: 8.5 bits per weight, re-rounded onto an 8-bit grid of its own. The file shrinks, and the values move. Only 11.7% of the drafter's weights come back bit-identical.

Q4_0 is 4.5 bits per weight: one fp16 step per group of 32, and a 4-bit level for each value. That is enough to hold the trained grid, provided each group's step is read off its own values. `gguf_exact.py` does this for each group:

```
amax = np.abs(g).max(-1, keepdims=True)
for m in range(1, 9):                      # try steps amax/1 .. amax/8
    cand = amax / m
    q = np.clip(np.rint(g / cand), -8, 7)
    good = np.abs(g - q * cand).max(-1, keepdims=True) <= amax * 2**-7
```

The first step that puts every value in the group on an integer level wins. A least-squares fit then refines it, and it is rounded to fp16, Q4_0's step type. A group that fits no step stops the build, since nothing is re-rounded. On both drafters no group was off the grid.

The script takes Google's safetensors and the converter's Q8_0 file, copies the Q8_0 file's metadata byte for byte (tokenizer, chat template, settings), and writes every weight matrix as Q4_0. That covers 23 tensors per drafter, including the token embedding, which holds 86–87% of the values rewritten:

```
python3 gguf_exact.py <google-snapshot-dir> gemma-4-E2B-it-qat-assistant-q8_0.gguf gemma-4-E2B-it-qat-assistant-q4_0-exact.gguf
python3 gguf_exact.py <google-snapshot-dir> gemma-4-E2B-it-qat-assistant-q8_0.gguf gemma-4-E2B-it-qat-assistant-bf16.gguf --bf16
```

`--bf16` writes Google's weights unchanged into the same layout, so all three formats can be compared with identical metadata.

| E2B drafter | Q8_0 (converter) | exact Q4_0 (repack) | bf16 (Google's weights) |
| :--- | ---: | ---: | ---: |
| File size | 97.8 MB | **59.2 MB** | 170.2 MB |
| Bits per weight | 8.5 | **4.5** | 16 |
| Error against Google's weights | 0.00494 | **0.00138** | 0 |
| Values bit-identical, read back in fp32 | 11.7% | **29.4%** | 100% |
| Values bit-identical, rounded to bf16 | — | **96.6–97.0% per tensor** | 100% |

The repack is 60% of the Q8_0 file and its error is 3.6x lower. Rounded back to bf16, the format Google stored, 97% of its values match Google's bit for bit. The E4B drafter gives the same picture: 59.7 MB, error 0.00138, 96.5–97.1% per tensor.

---

#### 🔎 Tip: The Converter Drops the Masked Embedding

llama.cpp's converter leaves out the checkpoint's `masked_embedding` tensors, and the repack keeps its tensor list. All three drafter files compared here lack them, so the comparison is fair. The files work with `--spec-type draft-mtp` as they are.

---

#### Step 2 — Fetch the Published Drafters

Both repacks are on Hugging Face:

- [xbill9/gemma-4-E2B-it-qat-assistant-q4_0-exact-gguf](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-assistant-q4_0-exact-gguf), file `gemma-4-E2B-it-qat-assistant-q4_0-exact.gguf`, SHA-256 `060f5a0158f8120d…`
- [xbill9/gemma-4-E4B-it-qat-assistant-q4_0-exact-gguf](https://huggingface.co/xbill9/gemma-4-E4B-it-qat-assistant-q4_0-exact-gguf), file `gemma-4-E4B-it-qat-assistant-q4_0-exact.gguf`, SHA-256 `dc965cb963e56b86…`

They pair with the main models' exact Q4_0 GGUFs, built the same way: [xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf) and [xbill9/gemma-4-E4B-it-qat-q4_0-exact-gguf](https://huggingface.co/xbill9/gemma-4-E4B-it-qat-q4_0-exact-gguf).

Each rig's `make draft` downloads its drafter at a pinned revision and checks the hash from `tpu.env`:

```
curl -fL -o $(SPEC_DRAFT_MODEL).part https://huggingface.co/$(SPEC_DRAFT_HF)/resolve/$(SPEC_DRAFT_HF_REVISION)/$(notdir $(SPEC_DRAFT_MODEL))
echo "$(SPEC_DRAFT_SHA256)  $(SPEC_DRAFT_MODEL)" | sha256sum -c -
```

Both cards loaded the same two files, checked by hash.

---

#### How the Repack Compares on Both Cards

Each format was served on both cards at 2 drafted tokens, six prompts, greedy. Gain against no drafter:

| 2 drafted tokens | 3050 E2B | 1650 Ti E2B | 3050 E4B | 1650 Ti E4B |
| :--- | ---: | ---: | ---: | ---: |
| Q8_0 | 1.39x | 1.09x | 1.72x | 1.40x |
| **exact Q4_0** | **1.40x** | **1.11x** | **1.72x** | **1.42x** |
| bf16 | 1.23x | 0.96x | 1.59x | 1.30x |

Memory with each drafter loaded, at `-c 8192`:

| MiB | no drafter | Q8_0 | exact Q4_0 | bf16 |
| :--- | ---: | ---: | ---: | ---: |
| 3050 E2B | 2,795 | 2,915 | **2,875** | 2,983 |
| 3050 E4B | 4,415 | 4,537 | **4,499** | 4,607 |
| 1650 Ti E2B | 1,485 | 1,595 | **1,557** | 1,663 |
| 1650 Ti E4B | 2,853 | 2,965 | **2,927** | 3,035 |

- **The repack is level with Q8_0 on speed on both cards** and uses 38–40 MiB less memory.
- **bf16 is slowest everywhere.** On the 1650 Ti it makes E2B slower than serving without a drafter.
- **The share of drafted tokens kept is the same for all three formats**, within 0.012 on both cards. The drafter's precision changes what each drafted step costs, never which tokens get kept.
- **Greedy text depends on drafting, never on the drafter's format.** At a given draft length, Q8_0, the repack and bf16 give the same 18 greedy replies, on the 3050 for both models and on the 1650 Ti for E2B (the 1650 Ti's E4B text does not repeat on `f95b0d9`, so it cannot be compared there).

Answer quality on the 3050, GSM8K (1,319 problems) and BFCL simple (400 function calls), 8 requests at a time:

| | no drafter | Q8_0 | exact Q4_0 |
| :--- | ---: | ---: | ---: |
| E2B GSM8K | 0.8984 | 0.8992 | 0.9022 |
| E2B BFCL | 0.9250 | 0.9225 | 0.9225 |
| E4B GSM8K | 0.9340 | 0.9310 | 0.9325 |
| E4B BFCL | 0.9100 | 0.9150 | 0.9150 |

Every drafted score is level with the same day's no-drafter run, and the repack is level with Q8_0: E2B GSM8K +0.0030 (95% range −0.0061 to +0.0129), identical BFCL answers on E2B.

Google also publishes non-QAT drafters, [google/gemma-4-E2B-it-assistant](https://huggingface.co/google/gemma-4-E2B-it-assistant) and [google/gemma-4-E4B-it-assistant](https://huggingface.co/google/gemma-4-E4B-it-assistant). Their weights are on no 4-bit grid, so they were converted to Q8_0. Against the QAT targets they keep fewer drafted tokens on E2B and are level or behind at every draft length on both models: E2B 1.34x against the repack's 1.40x on the 3050, and 1.16x against 1.18x on the 1650 Ti.

The 1650 Ti's format run used llama.cpp `f95b0d9`, the build its rigs serve; every other comparison here uses `77c4540ce` on both cards.

---

#### Step 3 — Serve With the Drafter

The drafter adds four flags to the rig's normal command line:

```
llama-server -m gemma-4-E4B-it-q4_0-exact.gguf --host 127.0.0.1 --port 8080 \
  -ngl 99 -c 8192 -ctk f16 -ctv f16 -fa 1 --reasoning off --metrics \
  -md gemma-4-E4B-it-qat-assistant-q4_0-exact.gguf --spec-type draft-mtp -ngld all --spec-draft-n-max 2
```

`--spec-draft-n-max` is the number of tokens drafted per step. Each rig's `make serve` builds this line from `tpu.env`, and an empty `SPEC_DRAFT_MODEL` serves without the drafter. The drafter costs 72–84 MiB of memory on either card.

---

#### Step 4 — Measure One Request at a Time

`speed.py` runs six prompts (code, advice, a story, an explanation, a list, a summary), 256 tokens each, three greedy repeats plus one sampled pass at temperature 0.8, with the prompt cache off. Each draft length gets its own server, the card cools before each one, and the first drafted setting runs again at the end to check for drift. The same script, prompts and llama.cpp build ran on both cards.

```
python3 speed.py <rig> <out>
```

---

#### The Result

Greedy, geometric mean over the six prompts. "Kept" is the share of drafted tokens the main model accepted.

| | 3050 tok/s | 3050 gain | 3050 kept | 1650 Ti tok/s | 1650 Ti gain | 1650 Ti kept |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| E2B, no drafter | 108.2 | — | — | 76.5 | — | — |
| E2B, 1 token | 148.9 | 1.38x | 0.609 | **90.3** | **1.18x** | 0.602 |
| E2B, 2 tokens | **151.2** | **1.40x** | 0.475 | 83.6 | 1.09x | 0.469 |
| E2B, 3 tokens | 143.0 | 1.32x | 0.384 | 77.3 | 1.01x | 0.378 |
| E4B, no drafter | 58.3 | — | — | 38.0 | — | — |
| E4B, 1 token | 89.8 | 1.54x | 0.624 | **56.5** | **1.49x** | 0.635 |
| E4B, 2 tokens | **100.0** | **1.72x** | 0.481 | 54.5 | 1.44x | 0.506 |
| E4B, 3 tokens | 99.3 | 1.70x | 0.398 | 50.9 | 1.34x | 0.414 |

The share of tokens kept matches across the cards at every length. What differs is how much each extra drafted token costs: on the 3050 E4B gains from 1 token to 2 and barely loses at 3, while on the 1650 Ti every extra token makes it slower. The likely reason is that each drafted step costs a larger share of a main-model step on the older card (inferred, not measured).

The repeated setting landed within 0.3% (3050) and 2.0% (1650 Ti) of its first run.

---

#### 🔎 Tip: Pick the Draft Length Per Card

The best draft length is a property of the card as much as the model. The 3050 rigs serve 2 tokens for both models. On the 1650 Ti, 1 token leads for both models on this build, and E2B at 3 tokens is no faster than serving without a drafter. A setting copied from a faster card loses most of the gain on a slower one.

---

#### Step 5 — Measure With Long Text Ahead

`longctx.py` puts the first 0 to 120,000 tokens of the NVIDIA driver README ahead of a story, a code request or a summary of the text, and measures generation speed alone. Same document, by hash, on both cards. The 1650 Ti's E4B stops at 32,768 tokens, the most that fits in 4 GiB with a drafter loaded.

Gain at 1 drafted token, mean over the three requests:

| tokens ahead | 3050 E2B | 1650 Ti E2B | 3050 E4B | 1650 Ti E4B |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.41x | 1.27x | 1.61x | 1.52x |
| 8,192 | 1.38x | 1.18x | 1.55x | 1.53x |
| 32,768 | 1.33x | 1.03x | 1.48x | 1.45x |
| 65,536 | 1.27x | 0.97x | 1.46x | |
| 120,000 | 1.23x | 0.89x | 1.41x | |

On the 3050 the drafter pays at every depth for both models; at 2 tokens it does better still, E4B 1.86x with nothing ahead and 1.55x at 120k. On the 1650 Ti E2B's gain is gone by 32k tokens and turns into a loss beyond. The share of tokens kept holds steady as the text grows on both cards, so the drop comes from each checking step getting dearer with a long context.

---

#### Step 6 — Measure Several Requests at Once

`sweep.py` sends 1 to 16 requests at once, each with a unique prompt, in two shapes: 512 tokens in and 128 out, and 128 in and 512 out. Both cards ran it with one shared KV cache (`-kvu`); the 3050 at `-c 65536 --parallel 64`, the 1650 Ti at `-c 16384 --parallel 16`.

Total output with the drafter divided by without:

| requests at once | 1 | 2 | 4 | 8 | 16 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 3050 E2B, 512 / 128 | 1.56 | 1.17 | 1.28 | 1.28 | 1.05 |
| 1650 Ti E2B, 512 / 128 | 1.15 | 1.04 | 1.03 | 0.95 | 0.97 |
| 3050 E2B, 128 / 512 | 1.79 | 1.28 | 1.43 | 1.54 | 1.18 |
| 1650 Ti E2B, 128 / 512 | 1.33 | 1.12 | 1.09 | 0.91 | 0.96 |
| 3050 E4B, 512 / 128 | 1.86 | 1.38 | 1.32 | 1.38 | 1.14 |
| 1650 Ti E4B, 512 / 128 | 1.28 | 1.07 | 0.90 | 0.93 | 0.92 |
| 3050 E4B, 128 / 512 | 2.05 | 1.44 | 1.41 | 1.60 | 1.26 |
| 1650 Ti E4B, 128 / 512 | 1.65 | 1.18 | 0.79 | 0.83 | 0.83 |

`sweep.py`'s prompt repeats one paragraph, which the drafter guesses unusually well, so these ratios run higher than on everyday chat. On mixed chat prompts the 3050 measured E4B 1.08–1.24x and E2B 0.92–1.17x at 2–8 requests, and a loss at 32–64 for both. Read the table for the trend: the 3050 keeps a gain through 16 requests here, and the 1650 Ti loses from 4 or 8.

---

#### 🔎 Tip: The Shared KV Cache Has a Price on Both Cards

llama-server gives each slot its own KV cache by default. On llama.cpp `77c4540ce` that layout halved generation speed at random with several requests in flight, on both cards: per-slot repeats spread 28–36% at 2 requests on the 1650 Ti. `-kvu` gives all slots one shared cache and the spread drops to about 1%. It also costs output as the total text held grows. At 16 requests in `sweep.py` the shared cache delivered 0.86–0.94x of per-slot output on the 3050 and 0.59–0.81x on the 1650 Ti. The drafter's ratio is nearly the same under either layout. All four rigs serve one request at a time, where the two layouts are identical.

---

#### Step 7 — Check the Answers

GSM8K (1,319 problems) and BFCL simple (400 function calls), one request at a time, with and without the drafter:

| | 3050 none | 3050 drafter | 1650 Ti none | 1650 Ti drafter |
| :--- | ---: | ---: | ---: | ---: |
| E2B GSM8K | 0.9014 | 0.8969 | 0.8939 | 0.8969 |
| E2B BFCL | 0.9225 | 0.9250 | 0.9250 | 0.9225 |
| E4B GSM8K | 0.9310 | 0.9295 | 0.9363 | 0.9310 |
| E4B BFCL | 0.9125 | 0.9125 | 0.9125 | 0.9125 |

Every difference has a 95% range that includes zero. The drafter finishes GSM8K 1.76x (E2B) and 2.23x (E4B) faster on the 3050 and 1.34x and 1.78x faster on the 1650 Ti. The 3050 ran at `-c 131072` with 2 drafted tokens for both models; the 1650 Ti at `-c 8192` with each rig's setting, 1 token for E2B and 2 for E4B.

---

#### 🔎 Tip: Greedy Text Can Change With the Drafter

At temperature 0 a drafted reply can differ from the undrafted one at a near-tie word, because checking several tokens at once runs different GPU kernels. On both cards E2B's drafted text differs from its undrafted text on every prompt at 2 tokens. E4B at 1 token matches on all 18 short replies on both cards. On the 3050 E4B also matches at 2 tokens for short prompts and starts to differ from about 2,000 tokens of prompt. The prompt cache changes greedy text by itself as well. A test that needs identical text must serve without the drafter and send `cache_prompt: false`. The scores above show that the changed words do not change the answers.

---

#### Compare and Contrast

| | 🥇 RTX 3050 OEM | 🥈 GTX 1650 Ti Max-Q |
| :--- | :--- | :--- |
| Best gain, E2B, one request | 1.40x at 2 tokens | 1.18x at 1 token |
| Best gain, E4B, one request | 1.72x at 2 tokens | 1.49x at 1 token |
| Fastest drafted E4B | 100.0 tok/s | 56.5 tok/s |
| Share of tokens kept, 1 token | 0.61–0.62 | 0.60–0.64 |
| E2B gain with 120k tokens ahead | 1.23x | 0.89x |
| Gain at 8 requests at once (`sweep.py`) | 1.28–1.60x | 0.83–0.95x |
| Answer quality | unchanged | unchanged |
| GSM8K faster with the drafter | 1.76x / 2.23x | 1.34x / 1.78x |

---

#### So, Which One?

Turn the drafter on for both cards when they serve one user. It costs no answers and 72–84 MiB of memory, and E4B gains the most on both: 1.72x on the 3050 and 1.49x on the 1650 Ti. Set the draft length per card, 2 tokens on the 3050 and 1 on the 1650 Ti.

On the 1650 Ti, leave E2B's drafter off for long conversations past about 32k tokens and for several users at once. On the 3050, the drafter keeps paying with long text ahead, and stops paying only once many users share the card.

---

#### Summary

The goal of this article was to measure what Google's Gemma 4 assistant drafters are worth on two small NVIDIA cards served by llama.cpp. The key to the solution was running the same llama.cpp build, scripts, prompts and drafter files on both cards, one setting per server, and checking answer quality alongside speed. The results were:

- 🟢 One request at a time the drafter speeds up E4B 1.72x on the RTX 3050 and 1.49x on the GTX 1650 Ti, and E2B 1.40x and 1.18x
- 🟢 Both cards keep the same share of drafted tokens, within 0.03 at every draft length
- 🟢 GSM8K and BFCL scores are unchanged with the drafter on both cards
- 🟢 The exact Q4_0 repack of the drafter is 60% of the Q8_0 file and level with it on speed on both cards; Google's bf16 weights are slowest on both
- ⚠️ The best draft length differs: 2 tokens on the 3050, 1 on the 1650 Ti
- ⚠️ With long text ahead, E2B's gain on the 1650 Ti is gone by 32k tokens; on the 3050 it holds at 1.23x at 120k
- ❌ With 4–8 or more requests at once the drafter costs output on the 1650 Ti

Scope: two cards on two hosts, llama.cpp `77c4540ce` on both, the exact Q4_0 E2B and E4B GGUFs with Google's QAT assistant drafters as exact Q4_0 (plus Q8_0 and bf16 for the format comparison), six short prompts for speed, one 120k-token document for long text, `sweep.py` at 1–16 requests with a shared KV cache, GSM8K and BFCL simple one request at a time, each setting run once with a repeat check.

The strategy for using MCP for local accelerator comparison was validated with an incremental step by step approach.

---

#### References

- Repository: https://github.com/xbill9/gemma4-dev
- RTX 3050 rigs: https://github.com/xbill9/gemma4-dev/tree/main/local-llamacpp-3050oem-2b-q4_0 and https://github.com/xbill9/gemma4-dev/tree/main/local-llamacpp-3050oem-4b-q4_0
- GTX 1650 Ti rigs: https://github.com/xbill9/gemma4-dev/tree/main/local-llamacpp-1650ti-2b-q4_0 and https://github.com/xbill9/gemma4-dev/tree/main/local-llamacpp-1650ti-4b-q4_0
- Drafter repack, 3050: https://github.com/xbill9/gemma4-dev/blob/main/local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-drafter-q4exact-3050oem/REPORT.md
- Drafter formats, 1650 Ti: https://github.com/xbill9/gemma4-dev/blob/main/local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-10-14-drafter-formats-1650ti/REPORT.md
- 3050 draft length and long text: https://github.com/xbill9/gemma4-dev/blob/main/local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-drafter-gaps-3050oem/REPORT.md
- 1650 Ti draft length and long text: https://github.com/xbill9/gemma4-dev/blob/main/local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-10-14-drafter-gaps-1650ti/REPORT.md
- 3050 several requests at once: https://github.com/xbill9/gemma4-dev/blob/main/local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-15-concurrency-drafter-kvu-3050oem/REPORT.md
- 1650 Ti several requests at once and answers: https://github.com/xbill9/gemma4-dev/blob/main/local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-10-15-drafter-load-1650ti/REPORT.md
- 3050 answers: https://github.com/xbill9/gemma4-dev/blob/main/local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-accuracy-fullctx-3050oem/REPORT.md
- Drafters: https://huggingface.co/xbill9/gemma-4-E2B-it-qat-assistant-q4_0-exact-gguf and https://huggingface.co/xbill9/gemma-4-E4B-it-qat-assistant-q4_0-exact-gguf
- llama.cpp: https://github.com/ggml-org/llama.cpp
