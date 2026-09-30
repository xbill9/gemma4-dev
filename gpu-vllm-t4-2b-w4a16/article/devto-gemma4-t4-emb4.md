---
title: "Gemma 4 on a Tesla T4, Part 3: Int4 Embeddings Serve E2B in 2.86 GiB at 2.30x bf16"
published: false
description: "Google's QAT Gemma 4 E2B keeps its embedding tables in bf16, and on a Tesla T4 they are most of the model. Packing them to int4 on the grid QAT trained them onto cuts model loading from 6.33 to 2.86 GiB, with every greedy test output token-identical, and raises vLLM's output throughput 11-37% over Google's own W4A16 export."
tags: gemma, vllm, cuda, machinelearning
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/gpu-vllm-t4-2b-w4a16/article/devto-t4-emb4-cover.71f9359a.jpg
---

This article provides a step by step guide to shrinking Google's quantization-aware-trained (QAT) Gemma 4 E2B to 4-bit weights end to end, embedding tables included, and serving it with vLLM on one Tesla T4 attached to a Compute Engine VM. It compares the result with the bf16 reference and with Google's own W4A16 export on the same card, with the same prompts.

The int4-embedding build loads in 2.86 GiB against 9.8 GiB for bf16, holds 1,099,587 tokens of KV cache against 315,974, and serves 85.28 output tokens per second to a single 512-token request against bf16's 37.04. All eight greedy test prompts produce the same tokens as the build with bf16 embeddings.

https://github.com/xbill9/gemma4-dev/tree/main/gpu-vllm-t4-2b-w4a16

---

#### Why Measure This?

[Part 1](https://dev.to/gde/gemma-4-on-a-tesla-t4-qat-weights-decode-179x-faster-than-bf16-2fi4) served Gemma 4 E2B on a T4 and found Google's QAT W4A16 checkpoint decodes 1.79x faster than bf16 for one user. [Part 2](https://dev.to/gde/gemma-4-on-a-tesla-t4-part-2-the-minimum-gce-vm-and-a-script-to-drive-it-3gk1) built the smallest VM that runs it and the script that drives it.

The QAT checkpoint quantizes the linear layers. Its embedding tables stay bf16, and E2B has two:

- `embed_tokens_per_layer`, the per-layer embedding (PLE) table that gives the E-series models their small active size: **4.375 GiB**
- `embed_tokens`, the token embedding, which Gemma ties to the output layer: **0.750 GiB**

Together they are 5.1 of the 6.11 GiB text-only checkpoint. The output layer also runs once per generated token over the whole 262,144-token vocabulary, so at bf16 it is one of the largest reads in every decode step.

QAT trained those tables too. Every sampled group of 32 values in both tables sits on the same 4-bit grid as the linear layers, and no group does in the bf16 base model. So the tables can be stored as int4 with no new quantization error.

---

#### At This Point You Should Have…

- The Compute Engine VM from Part 2: `n1-standard-2` (2 vCPU, 7.80 GB RAM), one Tesla T4, `us-west2-b`, a 16 GB swapfile on the data disk
- vLLM 0.29.0 with the Turing attention patch from Part 1 applied and verified
- Python 3 with `numpy` for the repack; no PyTorch is needed for it
- `git clone https://github.com/xbill9/gemma4-dev`

---

#### Step 1 — Repack the Linear Layers and Drop the Towers

`google/gemma-4-E2B-it-qat-q4_0-unquantized` stores bf16 values that already sit on a 4-bit grid in groups of 32. The repack recovers each group's grid step and writes compressed-tensors W4A16. A second script drops the vision and audio towers and sets the architecture to `Gemma4ForCausalLM`, leaving every other tensor byte-identical.

```bash
cd gemma4-dev/gpu-vllm-t4-2b-w4a16
python3 repack/repack_q4_0.py repack SRC OUT
python3 repack/text_only.py OUT OUT-text
```

The intermediate is published as [xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text): 6.11 GiB, int4 linears, bf16 embeddings.

---

#### Step 2 — Pack the Embedding Tables

```bash
python3 repack/embed_int4.py OUT-text OUT-text-emb4 --embed-tokens
```

`embed_int4.py` packs the PLE table and, with `--embed-tokens`, the token embedding. It aborts on any group that is off the grid.

| Table | bf16 | int4 |
| :--- | ---: | ---: |
| `embed_tokens_per_layer` | 4.375 GiB | 1.230 GiB |
| `embed_tokens` | 0.750 GiB | 0.211 GiB |
| `lm_head` (untied) | tied | 0.211 GiB |
| Whole checkpoint | 6.11 GiB | **2.64 GiB** |

vLLM ties the output layer by copying the embedding's `.weight`, and a packed embedding has none. So the script unties them and writes the same levels and scales a second time as `lm_head`, which vLLM runs as an int4 linear. The model was trained tied, so both copies hold the trained values.

The result is [xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4).

---

#### Step 3 — Store the Scales as fp16

Each group of 32 carries one scale, the grid step. The T4 computes in fp16, and fp16 has three more mantissa bits than bf16, so an fp16 scale reproduces more of the source values exactly:

| Table | bf16 scales: bit-identical | fp16 scales: bit-identical |
| :--- | ---: | ---: |
| PLE | 73.80% | 74.26% |
| `embed_tokens` | 73.48% | 74.02% |

The remaining mismatches are within one bf16 unit in the last place for 99.9% of values. They come from the source's own bf16 rounding of step times level. fp16 is the script's default.

---

#### Step 4 — Serve It

`tpu.env` names the checkpoint and every serving flag, and `vllm-t4` reads it:

```bash
./vllm-t4 start
```

The engine log from the sweep's server start:

```plaintext
served xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4 max_model_len 16384
Model loading took 2.86 GiB memory and 19.400900 seconds
torch.compile took 1.03 s in total
GPU KV cache size: 1,099,587 tokens, Maximum concurrency for 16,384 tokens per request: 67.11x
```

vLLM loads the packed tables through `CompressedTensorsEmbeddingWNA16Int` and logs `Using MarlinLinearKernel for CompressedTensorsWNA16` for the int4 linear layers. The flags are the ones from Part 1: `--dtype float16 --gpu-memory-utilization 0.9 --max-model-len 16384 --max-num-seqs 8 --language-model-only`.

🔎 Tip: restart once after changing the model. The first start compiles from scratch and profiles a larger activation peak, which leaves a smaller KV cache: 980,210 tokens on the cold start against 1,099,362 on the warm one.

---

#### Step 5 — Check It Serves

```bash
./vllm-t4 status
curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4","messages":[{"role":"user","content":"What does the HTTP status code 418 mean? One sentence."}],"temperature":0,"max_tokens":60}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content']); print(json.dumps(d['usage']))"
```

```plaintext
✅ Serving at http://127.0.0.1:8000 (pid 18273).

VRAM 13581 MiB, 15360 MiB, 0 %

The HTTP status code 418, "I'm a teapot," is a humorous error code indicating that the server refuses to brew coffee because it is a teapot and not a coffee maker.
{"prompt_tokens": 24, "total_tokens": 65, "completion_tokens": 41, "prompt_tokens_details": null, "completion_tokens_details": null}
```

The 13,581 MiB is vLLM's reservation at `--gpu-memory-utilization 0.9`; the KV cache fills whatever the weights leave.

---

#### What the Engine Allocates

Four builds of the same model, same flags, warm compile cache:

| Build | Model loading | KV cache |
| :--- | ---: | ---: |
| bf16 reference, `google/gemma-4-E2B-it` | 9.8 GiB | 315,974 tokens |
| Google QAT W4A16, `-qat-w4a16-ct` | 8.02 GiB | 519,568 tokens |
| Repack, text only, bf16 embeddings | 6.33 GiB | 711,539 tokens |
| **Repack, text only, int4 embeddings** | **2.86 GiB** | **1,099,362 tokens** |

The two Google builds were served as multimodal checkpoints, so their figures include the vision and audio towers.

At this serving shape, eight sequences of up to 16,384 tokens, no build's KV cache binds. The freed memory is headroom for more concurrent sequences or longer contexts.

---

#### Does It Produce the Same Text?

Eight prompts, 160 tokens each, greedy decoding, against the build with bf16 embeddings:

```plaintext
PROMPT: What is the capital of France? Answer in one word.
SAME TOKENS: True
PROMPT: List the first ten prime numbers, comma separated.
SAME TOKENS: True
PROMPT: What does the HTTP status code 418 mean?
SAME TOKENS: True
```

All eight are token-identical. A build that packs only the PLE table, with bf16 scales, matched seven of eight and diverged at token 55 of the HTTP 418 answer. That is a spot check of fidelity; no accuracy benchmark was run on either build.

---

#### Single-Stream Decode

One request at a time, 256 output tokens:

| Build | Decode, c=1 |
| :--- | ---: |
| Repack, text only, bf16 embeddings | 81.6 tok/s |
| PLE table int4 only | 81.2 tok/s |
| **PLE, `embed_tokens` and `lm_head` int4** | **109.7 tok/s** |

Packing the PLE table saves 3.14 GiB and changes nothing about speed: each token looks up one row of it per layer. The speedup comes from the output layer. At one token per step, decode on the T4 is limited by how many bytes it reads, and the output layer is a full read of a 262,144-row matrix on every token, 0.75 GiB at fp16 and 0.21 GiB at int4.

---

#### The Sweep

`vllm bench serve`, random prompts of 512 and 4096 tokens, 128 output tokens, concurrency 1 to 16, three repeats per cell, greedy. Every cell uses the same prompt seeds as Part 1's bf16 and QAT runs, so all three builds answer the same prompts. The largest run-to-run variation in any cell is 2.5%.

---

#### 512-Token Prompts

| c | bf16 out tok/s | QAT out tok/s | emb4 out tok/s | emb4 vs QAT | emb4 vs bf16 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 37.04 | 62.27 | **85.28** | +37% | 2.30x |
| 4 | 109.05 | 157.49 | **183.08** | +16% | 1.68x |
| 8 | 164.62 | 215.91 | **239.64** | +11% | 1.46x |
| 16 | 164.36 | 213.79 | **237.08** | +11% | 1.44x |

The gain is largest for one user and shrinks with concurrency. At c=1 each decode step reads every weight to produce one token, so the output layer is a large share of the step. At c=8 the same read produces eight tokens and attention over eight sequences grows, so the output layer's share is smaller.

c=16 exceeds `--max-num-seqs 8`, so the extra requests wait in the queue: throughput stays at the c=8 value and time to first token absorbs the wait.

---

#### 4096-Token Prompts

| c | bf16 out tok/s | QAT out tok/s | emb4 out tok/s | emb4 vs QAT |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 12.11 | 14.00 | **15.03** | +7% |
| 4 | 15.16 | 15.82 | **16.36** | +3% |
| 8 | 15.66 | 15.95 | **16.38** | +3% |
| 16 | 15.46 | 15.75 | **16.17** | +3% |

A 4096-token prompt takes about 7.1 seconds to its first token on all three builds. That time is compute over the prompt, which int4 embeddings do not change, so long prompts gain 3-7%. The 3% at c=4 and above is larger than those cells' run-to-run variation, which is 0.2% or less.

---

#### 🔎 Tip: W8A8 Is Slower on the T4

The same benchmark ran a W8A8 build, int8 weights and activations with the same int4 embeddings, for one repeat over five cells:

| input | c | W8A8 out tok/s | emb4 out tok/s |
| ---: | ---: | ---: | ---: |
| 512 | 1 | 45.67 | 85.28 |
| 512 | 8 | 181.25 | 239.64 |
| 4096 | 1 | 12.94 | 15.03 |

W4A16 is faster in every cell. Decode on this card is bound by bytes read, and 4-bit weights read half the bytes of 8-bit ones.

---

#### 🔎 Tip: Google's Drafter for One User

Google publishes a small assistant model for speculative decoding with the QAT target, `google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant`. vLLM serves it as method `gemma4_mtp`:

```bash
--speculative-config '{"model":"google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant","num_speculative_tokens":3}'
```

Twenty real chat prompts, one at a time, median time per output token:

| Setting | Time per token | Drafts accepted |
| :--- | ---: | ---: |
| No drafter, greedy | 9.9 ms | |
| 2 draft tokens | 7.9 ms | 48% |
| **3 draft tokens** | **7.4 ms** | **38%** |
| 4 draft tokens | 7.3 ms | 32% |
| 6 draft tokens | 7.5 ms | 23% |
| No drafter, default sampling | 9.8 ms | |
| 3 draft tokens, default sampling | 8.7 ms | 37% |

With greedy decoding, three draft tokens make decode 1.35x faster, and every one of the twenty prompts is faster. At the checkpoint's own sampling settings (temperature 1.0, top-k 64, top-p 0.95) the gain drops to 1.13x at the same acceptance, because sampling with the drafter costs extra on the T4. Startup takes about 4.5 minutes instead of 90 seconds. Three of the twenty greedy answers differ in length from the plain engine's. These are one run per setting, measured at one user; the gain under concurrency is unmeasured.

---

#### Compare and Contrast

| One Tesla T4 | bf16 | Google QAT W4A16 | QAT, int4 embeddings |
| :--- | :---: | :---: | :---: |
| Model loading | 9.8 GiB | 8.02 GiB | 🥇 2.86 GiB |
| KV cache | 315,974 | 519,568 | 🥇 1,099,587 |
| Out tok/s, 512, c=1 | 37.04 | 62.27 | 🥇 85.28 |
| Out tok/s, 512, c=8 | 164.62 | 215.91 | 🥇 239.64 |
| Out tok/s, 4096, c=1 | 12.11 | 14.00 | 🥇 15.03 |
| Vision and audio | 🥇 yes | 🥇 yes | text only |
| Published by | 🥇 Google | 🥇 Google | community repack |

---

#### So, Which One?

For text serving on a T4, the int4-embedding build: it is the fastest at every cell measured, and its KV cache is 3.48x the size of bf16's. For images or audio, Google's QAT W4A16 export, which keeps the towers and is still 1.31x bf16 at eight users. bf16 is the reference to measure against.

---

#### What Does It Cost?

The VM's compute is $0.5241 an hour at us-west2 on-demand list prices: the T4, two vCPUs and 7.5 GiB of RAM. At the measured 512-token, eight-user throughput, by arithmetic:

| Build | Output tok/s | Per million output tokens |
| :--- | ---: | ---: |
| bf16 | 164.62 | $0.88 |
| Google QAT W4A16 | 215.91 | $0.67 |
| **QAT, int4 embeddings** | **239.64** | **$0.61** |

This assumes the VM runs at that load around the clock, and excludes disks, sustained-use discounts and egress.

---

#### Teardown

```bash
./vllm-t4 stop
```

```plaintext
✅ Sent SIGTERM to vLLM (pid 14977). VRAM is released on exit.
```

Stopping vLLM frees the GPU. The VM and its T4 bill by the hour while the VM runs; stop the VM to stop the charge. The swapfile stays on the data disk and `vllm-t4 start` turns it back on.

---

#### Summary

The goal of this article was to serve Gemma 4 E2B on a Tesla T4 with every weight at 4 bits. The key to the solution was that QAT trained the embedding tables onto the same 4-bit grid as the linear layers, so packing them adds no error, and untying the output layer lets vLLM run it as an int4 linear. The results were:

- 🟢 Model loading 2.86 GiB against 9.8 GiB for bf16, and 1,099,587 KV tokens against 315,974
- 🟢 85.28 output tok/s for one 512-token request, 2.30x bf16 and 1.37x Google's QAT W4A16 export
- 🟢 239.64 tok/s at eight users, 1.46x bf16 and 1.11x the QAT export
- 🟢 Eight of eight greedy test outputs token-identical to the build with bf16 embeddings
- ⚠️ 3-7% gain on 4096-token prompts, where prompt processing sets the rate
- 🟢 Google's drafter adds 1.35x for one user with greedy decoding, 1.13x with default sampling
- ❌ W8A8 is slower than W4A16 on the T4 in every cell measured
- ⚠️ Text only: the vision and audio towers are dropped

Scope: one Tesla T4 on one `n1-standard-2` VM in `us-west2-b`, vLLM 0.29.0, torch 2.13.0+cu130, three repeats per sweep cell. The bf16 and QAT cells are Part 1's 2026-09-18 run with the same host, flags, benchmark script and prompt seeds; the int4-embedding cells were measured 2026-09-29. That comparison changes three things at once: the exact-grid repack, int4 embeddings with an int4 output layer, and text only. The single-stream table isolates the embeddings. Fidelity was spot-checked on eight prompts and no accuracy benchmark was run. The W8A8 and drafter figures are one run per setting.

The strategy for using MCP for Tesla T4 deployment and benchmarking was validated with an incremental step by step approach.

---

#### References

- The int4-embedding checkpoint: https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4
- The bf16-embedding intermediate: https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text
- Code, repack scripts, sweep and logs: https://github.com/xbill9/gemma4-dev/tree/main/gpu-vllm-t4-2b-w4a16
- Google's QAT source checkpoint: https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-unquantized
- Google's QAT W4A16 export: https://huggingface.co/google/gemma-4-E2B-it-qat-w4a16-ct
- Google's drafter: https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant
- The bf16 reference: https://huggingface.co/google/gemma-4-E2B-it
- Part 1: https://dev.to/gde/gemma-4-on-a-tesla-t4-qat-weights-decode-179x-faster-than-bf16-2fi4
- Part 2: https://dev.to/gde/gemma-4-on-a-tesla-t4-part-2-the-minimum-gce-vm-and-a-script-to-drive-it-3gk1
- vLLM speculative decoding: https://docs.vllm.ai/en/latest/features/speculative_decoding/
