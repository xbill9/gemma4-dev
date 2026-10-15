---
title: "Gemma 4's MTP Drafter, Explained: Speculative Decoding and a 4-Bit Drafter at 60% of the Size"
published: false
description: "What speculative decoding is, how Gemma 4's multi-token prediction drafter rides on the main model, why you would add one, and an exact Q4_0 repack of Google's QAT drafters on Hugging Face that cuts the file to 60% of the size and the memory cost by a third."
tags: gemma, llamacpp, huggingface, ai
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/local-llamacpp-3050oem-2b-q4_0/docs/mtp-drafter-intro/devto-cover.f5d57346.jpg
---

This article provides a step by step introduction to speculative decoding with Gemma 4's multi-token prediction (MTP) drafter, served by llama.cpp on a local, desktop hosted GPU enabled system. It is the first of a short series: this part explains what the drafter is and why you would use it, and the next part has the measurements.

https://github.com/xbill9/gemma4-dev/tree/main/local-llamacpp-3050oem-2b-q4_0

**Gemma 4 ships a small companion model for every size, a drafter that guesses the next few tokens so the main model can check them in one pass. Google's QAT drafters are trained onto a 4-bit grid; stored that way they shrink from 97.8 MB to 59.2 MB and are published on Hugging Face.**

---

#### What is this project trying to Do?

The rigs in this repository serve Gemma 4 E2B and E4B from local hardware: an RTX 3050 OEM in a desktop and a GTX 1650 Ti in a laptop. Both already run the exact 4-bit QAT builds of the main models. Google also publishes a drafter for each Gemma 4 model, and llama.cpp can now run it.

This part covers three things: how speculative decoding works, how Gemma 4's drafter differs from the usual draft model, and the exact Q4_0 repacks of Google's QAT drafters that these rigs serve.

---

#### Why Is Generating One Token at a Time Slow?

A language model writes one token per forward pass, and every pass reads every weight. For one user on a small GPU, the arithmetic is light and the memory traffic is heavy: the card spends most of each step moving weights, not computing with them.

That leaves spare compute on every step. Checking several tokens in one pass reads the same weights once and does a little more arithmetic, so it takes little more time than producing one. Speculative decoding exists to use that spare capacity.

---

#### How Does Speculative Decoding Work?

Each step has two parts.

1. **Draft.** A small, fast model guesses the next few tokens.
2. **Verify.** The main model, called the target, runs one forward pass over all the guesses at once and checks each one against what it would have produced.

The target accepts guesses from the front until the first one it disagrees with. It discards the rest and adds one token of its own at that point. A step therefore always produces at least one token, and as many as the number drafted plus one when every guess is right.

The output is the target's: every token either matches what the target would have chosen or is the target's own replacement. The drafter only changes how many tokens each expensive pass produces.

Google's model card puts it this way:

> When used in a Speculative Decoding pipeline, the draft model predicts several tokens ahead, which the target model then verifies in parallel.

---

#### What Makes Gemma 4's Drafter Different?

A classic draft model is a smaller copy of the same family, with its own layers, its own KV cache and its own pass over the whole conversation. Gemma 4's drafter is an add-on to the target. Google calls the approach multi-token prediction and describes the drafter as a lightweight 4-layer model that uses the target model's activations and KV cache.

The E2B drafter's `config.json` on Hugging Face shows how small it is:

```json
"architectures": ["Gemma4AssistantForCausalLM"],
"backbone_hidden_size": 1536,
"hidden_size": 256,
"num_hidden_layers": 4,
"num_kv_shared_layers": 4,
"layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
"vocab_size": 262144,
"tie_word_embeddings": true
```

| Property | What it means |
| :--- | :--- |
| `backbone_hidden_size: 1536` | the drafter reads the E2B target's 1536-wide hidden state (2560 for E4B) |
| `hidden_size: 256`, 4 layers | a tiny transformer of its own |
| `num_kv_shared_layers: 4` | all four layers reuse the target's KV cache; the drafter computes queries only |
| `vocab_size: 262144` | the same tokenizer as the target, so its guesses are the target's tokens |

Two consequences follow. The drafter keeps no separate cache of the conversation, so its memory cost stays its weights and a small working buffer. And it starts each guess from what the target already worked out for the current token, which is why so small a model can guess well.

The checkpoint also carries a clustered shortcut for its output layer (`num_centroids: 2048`). llama.cpp's converter leaves that out and scores the full vocabulary.

---

#### Why Would You Use a Drafter?

- 🟢 **Faster answers for one user.** On a GPU waiting on memory, each pass that verifies two or three tokens does the work of two or three passes. Google's card claims up to 3x.
- 🟢 **No retraining and no change to the main model.** The target file, its settings and its context stay as they are; the drafter is one more file and two server flags.
- 🟢 **Small memory cost.** The drafter shares the target's KV cache, so it costs its weights and a small working buffer.
- ⚠️ **The gain depends on the text.** Predictable output such as code is guessed well; free-form stories are guessed less often, and each rejected guess is wasted work.
- ⚠️ **The gain depends on load.** With many users at once the GPU is already busy, and spare capacity for checking guesses shrinks. Part 2 measures where it stops paying.
- ⚠️ **Greedy text can change at near-ties.** Where two words score almost the same, checking several tokens in one batched pass can tip the choice the other way; batched arithmetic rounding differently is the likely cause, untested here. The answers stay the target's; part 2 shows the scores.

---

#### Which Drafter Goes With Which Model?

Google publishes two drafters per Gemma 4 size: `-it-assistant` for the bf16 model and `-it-qat-q4_0-unquantized-assistant` for the QAT builds. Google's QAT model card is explicit about pairing them:

> When using multi-token prediction (speculative decoding) with an assistant model alongside a QAT target model, the assistant model must also be a QAT checkpoint with the same precision to ensure compatibility.

These rigs serve the exact Q4_0 QAT builds of E2B and E4B, so they take the QAT drafters.

---

#### Step 1 — Repack the QAT Drafter Exactly

Google ships the QAT drafter as bf16, but the values already lie on a 4-bit grid: in every weight matrix, each block of 32 holds at most 16 distinct values. That is Q4_0's own layout, so the weights can be stored at 4 bits without rounding them again.

llama.cpp's converter writes the drafter as Q8_0. `gguf_exact.py`, the same builder behind the exact main-model GGUFs, finds each block's trained step and stores the drafter as Q4_0 instead. No block was off the grid.

| E2B drafter | File | Error vs Google's bf16 |
| :--- | ---: | ---: |
| bf16, Google's weights | 170.2 MB | 0 |
| Q8_0, llama.cpp converter | 97.8 MB | 0.00494 |
| **Exact Q4_0** | **59.2 MB** 🥇 | **0.00138** 🥇 |

The exact Q4_0 file is 39% smaller than Q8_0 and 3.6x closer to Google's weights by relative error (arithmetic from the build reports). The E4B drafter comes out the same: 59.7 MB against 98.7 MB.

---

#### Step 2 — Download It From Hugging Face

Both drafters are published, with their build script and evidence:

- `xbill9/gemma-4-E2B-it-qat-assistant-q4_0-exact-gguf`
- `xbill9/gemma-4-E4B-it-qat-assistant-q4_0-exact-gguf`

In the rig, `make draft` downloads the pinned revision and checks its hash:

```text
060f5a0158f8120d25137866b90e04e21cd0d2d23de1db7a9cc3b7f00e60bd11  gemma-4-E2B-it-qat-assistant-q4_0-exact.gguf
dc965cb963e56b86fbc04db8efec83163886085e88514574650a57fa55a2ee24  gemma-4-E4B-it-qat-assistant-q4_0-exact.gguf
```

---

#### Step 3 — Serve the Target With the Drafter

llama.cpp runs Gemma 4's drafter as its own speculative type, `draft-mtp`. Two flags added to the usual server command:

```shell
llama-server -m gemma-4-E2B-it-q4_0-exact.gguf -ngl 99 -fa 1 \
  -md gemma-4-E2B-it-qat-assistant-q4_0-exact.gguf \
  --spec-type draft-mtp -ngld all --spec-draft-n-max 2
```

| Flag | Meaning |
| :--- | :--- |
| `-md` | the drafter file |
| `--spec-type draft-mtp` | run it as a Gemma 4 MTP drafter on the target's state |
| `-ngld all` | put the drafter on the GPU with the target |
| `--spec-draft-n-max 2` | guess at most two tokens per step |

The server log confirms the drafter loaded and reports, per request, how many guesses the target accepted:

```text
I common_speculative_init_result: loading draft model '/home/xbill/models/gemma-4-E2B-it-qat-assistant/gemma-4-E2B-it-qat-assistant-q4_0-exact.gguf'
I slot print_timing: id  0 | task 8 | draft acceptance = 0.84656 (  160 accepted /   189 generated), mean len =  2.68
```

That request was a coding prompt: 160 of 189 guesses accepted, and each target pass produced 2.68 tokens on average.

---

#### How Much Memory Does the Drafter Cost?

Measured on the RTX 3050 at 8,192 tokens of context, with the E2B target alone at 2,795 MiB:

| Drafter added | GPU memory | Cost of the drafter |
| :--- | ---: | ---: |
| none | 2,795 MiB | — |
| bf16 | 2,983 MiB | +188 MiB |
| Q8_0 | 2,915 MiB | +120 MiB |
| **Exact Q4_0** | **2,875 MiB** 🥇 | **+80 MiB** 🥇 |

The exact drafter costs a third less memory than the Q8_0 conversion. With E4B the pattern holds: +84 MiB against +122. On a 4 GB laptop card every one of those megabytes is context or headroom.

The smaller file also leaves the guesses unchanged: Q4_0, Q8_0 and Google's bf16 drafter had the target accept the same share of drafted tokens. Part 2 covers what that means for speed.

---

#### 🔎 Tip: Where the Drafter Runs

llama.cpp's `draft-mtp` is the route measured here. Two others exist:

- **Hugging Face Transformers**: pass `assistant_model=` to `generate()`, which is the route Google documents.
- **vLLM**: `--speculative-config '{"method":"mtp", ...}'`, measured in this repository on a Cloud TPU v5e.

---

#### 🔎 Tip: Start With Two Drafted Tokens

Every extra guess must be right for the one after it to count. Two drafted tokens per step was the best setting on the RTX 3050 for both models, and one on the GTX 1650 Ti laptop card. Raise it only for highly predictable output such as code.

---

#### Compare and Contrast

| | Classic draft model | Gemma 4 MTP drafter |
| :--- | :--- | :--- |
| What it reads | the conversation, from scratch | the target's hidden state and KV cache |
| Own KV cache | yes, grows with context | none |
| Size | a smaller model of the family | 4 layers, 256 wide |
| Tokenizer | must match | same 262,144-token vocabulary |

| E2B QAT drafter | File | GPU memory added |
| :--- | ---: | ---: |
| bf16 | 170.2 MB | +188 MiB |
| Q8_0 | 97.8 MB | +120 MiB |
| Exact Q4_0 | **59.2 MB** 🥇 | **+80 MiB** 🥇 |

---

#### So, Which One?

For a Gemma 4 QAT target served by llama.cpp, the exact Q4_0 QAT drafter. It matches the target's precision as Google requires, it is the smallest file, the closest to Google's weights and the cheapest in memory, and the target accepts its guesses as often as it accepts the larger files'.

Turn it on for one user, or a few, on a GPU that waits on memory. The next part measures how much faster that is, on which kinds of text, and where adding users makes the drafter stop paying.

---

#### Summary

The goal of this article was to explain speculative decoding with Gemma 4's MTP drafter and to introduce exact 4-bit repacks of Google's QAT drafters. The key to the solution was Google's drafter design, a 4-layer model that reuses the target's state and KV cache, trained onto the same 4-bit grid as the QAT target. The results were:

- 🟢 **A drafter guesses, the target verifies** several tokens in one pass, and every token is still the target's.
- 🟢 **Gemma 4's drafter keeps no KV cache of its own**: 4 layers of width 256 that read the target's hidden state.
- 🟢 **Exact Q4_0 QAT drafters on Hugging Face**: 59.2 MB for E2B and 59.7 MB for E4B, 39–40% smaller than llama.cpp's Q8_0 conversion and 3.6x closer to Google's weights.
- 🟢 **+80 MiB of GPU memory for the E2B drafter**, against +120 MiB for Q8_0 and +188 MiB for bf16.
- ⚠️ **The gain depends on the text and on the number of users**; part 2 has the measurements.
- ⚠️ **Greedy text can change at near-ties** when verification runs in batches.

Scope: Google's QAT drafters for Gemma 4 E2B and E4B, repacked with `gguf_exact.py` and served by llama.cpp `77c4540ce` on one RTX 3050 OEM (8 GB) at 8,192 tokens of context on 2026-10-14. The descriptions of the drafter come from Google's model cards, its MTP documentation and the checkpoints' `config.json`.

The strategy for adding Gemma 4's MTP drafter to a local llama.cpp deployment was validated with an incremental step by step approach.

---

#### References

* [xbill9/gemma-4-E2B-it-qat-assistant-q4_0-exact-gguf | Hugging Face](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-assistant-q4_0-exact-gguf)
* [xbill9/gemma-4-E4B-it-qat-assistant-q4_0-exact-gguf | Hugging Face](https://huggingface.co/xbill9/gemma-4-E4B-it-qat-assistant-q4_0-exact-gguf)
* [google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant | Hugging Face](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant)
* [google/gemma-4-E2B-it-assistant | Hugging Face](https://huggingface.co/google/gemma-4-E2B-it-assistant)
* [xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf | Hugging Face](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf)
* [local-llamacpp-3050oem-2b-q4_0 | GitHub](https://github.com/xbill9/gemma4-dev/tree/main/local-llamacpp-3050oem-2b-q4_0)
* [Gemma 4 on a Used 8 GB RTX 3050 | dev.to](https://dev.to/gde/gemma-4-on-a-used-8-gb-rtx-3050-116-tokenss-for-one-user-618-for-64-k6h)
* [llama.cpp | GitHub](https://github.com/ggml-org/llama.cpp)
