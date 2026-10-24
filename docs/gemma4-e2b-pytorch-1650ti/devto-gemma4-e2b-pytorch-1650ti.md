---
title: "Gemma 4 on a 4 GB GPU in Plain PyTorch: From Out of Memory to Faster Than llama.cpp"
published: false
description: "On a 4 GB GTX 1650 Ti, Gemma 4 E2B does not fit in 16-bit, and Transformers' own 4-bit path runs out of memory on its first forward pass. Kept in Google's QAT 4-bit end to end, with a few Triton kernels and a CUDA graph, plain PyTorch decodes at 88 tok/s, 1.07x llama.cpp on the same card, and holds the full 131,072-token context."
tags: gemma, pytorch, cuda, quantization
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/docs/gemma4-e2b-pytorch-1650ti/devto-cover.fa8eaefc.jpg
---

This article provides a step by step guide to running Gemma 4 E2B, Google's quantization-aware trained (QAT) build, on a 4 GB NVIDIA GTX 1650 Ti laptop GPU with plain PyTorch and Hugging Face Transformers. A small OpenAI-compatible server is built to run the result like any other local model server.

https://github.com/xbill9/gemma4-dev

**Can plain PyTorch run Gemma 4 on a 4 GB GPU? Yes, and faster than llama.cpp at chat-length context**, once the model stays 4-bit from disk to arithmetic:

| GTX 1650 Ti, 4 GB, Gemma 4 E2B | Decode | Fits the card |
| :--- | ---: | :--- |
| 16-bit weights | — | ❌ 3.58 GiB of layers alone |
| Transformers' own 4-bit path | — | ❌ out of memory on the first forward pass |
| 4-bit layers, unpacked by PyTorch one layer at a time | 0.58 tok/s | ✅ |
| **4-bit end to end, this article** | **87.8 tok/s** | ✅ **with the full 131,072-token context** |
| llama.cpp, same card, same day | 82.1 tok/s | ✅ |

Every weight stays on the 4-bit grid Google trained, and the speed comes from doing the 4-bit arithmetic in kernels that read the packed weights directly.

---

#### What Is Being Measured?

Gemma 4 E2B stores about 4.7 billion parameters (8.73 GiB in 16-bit), most of them in two vocabulary tables, and a 4 GB laptop GPU shows PyTorch 3,732 MiB. Google publishes E2B already trained for 4-bit weights. The questions are whether plain PyTorch can run it inside that memory, how fast it decodes, how much context fits beside it, and whether the answers survive, measured against llama.cpp on the same card.

The card is a GTX 1650 Ti with Max-Q design (Turing TU117, no tensor cores) in a laptop. The same card's llama.cpp results are in [Do Drafters Work on Small Models?](https://dev.to/gde/do-drafters-work-on-small-models-gemma-4-e2b-and-e4b-on-two-small-gpus-1b4g).

---

#### At This Point You Should Have…

- An NVIDIA GPU with 4 GB and a CUDA driver — here a GTX 1650 Ti Max-Q, driver 615.71.10
- PyTorch with CUDA and Triton — every number here is from torch 2.13.0+cu130 and Triton 3.7.1
- `transformers==5.16.1`, `compressed-tensors==0.17.0`, `accelerate==1.14.0`, in the system `python3`, no virtualenv
- About 9 GB of free disk for the two checkpoints
- The repository cloned, with the rig `local-pytorch-1650ti-2b-q4w4a16/`

---

#### Why Doesn't 16-bit Fit?

Unpacked to 16-bit on the CPU, the model's parts weigh:

```plaintext
{"embed_tokens": 0.75, "layers": 3.579, "embed_tokens_per_layer": 4.375, "total": 8.731}   # GiB
```

The 35 decoder layers alone are 3.58 GiB against 3.65 GiB the card gives PyTorch, before any working memory. The per-layer embedding table, 4.38 GiB, is Gemma 4's per-layer input table for the E sizes: a row of 35 × 256 values for every token in the 262,144-token vocabulary.

---

#### Step 1 — Load Google's QAT Weights Without Unpacking Them

The checkpoint is `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text`: Google's QAT E2B repacked into the compressed-tensors W4A16 format (4-bit weights in groups of 32, 16-bit activations), text only. Every 4-bit code in it equals Google's QAT release:

```plaintext
{"elements": 1876819968, "elements_differing": 187046012, "int4_codes_disagreeing": 0, "worst_diff_in_units_of_group_step": 0.062}
```

All 1,876,819,968 codes agree with `google/gemma-4-E2B-it-qat-q4_0-unquantized`. The 10% of values that differ do so by at most 6.2% of one grid step, from storing each group's step in bf16.

Loaded through Transformers with the per-layer table left in CPU memory, the packed layers take 1.73 GiB on the GPU. The first forward pass then fails. Transformers 5.16 applies compressed-tensors with `run_compressed=False` (`quantizer_compressed_tensors.py:114`), so compressed-tensors unpacks every layer to 16-bit on the first call, which needs the 3.58 GiB above. Transformers 5.20.2 does the same, and its `use_optimized_inference` covers W8A8 FP8 on SM89 and newer only.

Removing that unpack hook and unpacking each layer inside its own forward pass runs, at 0.58 tok/s:

```plaintext
{"step": "generate_warm", "prompt_tokens": 19, "new_tokens": 57, "seconds": 97.69, "tok_per_s": 0.58, "peak_cuda_mib": 1948}
```

---

#### Step 2 — Keep the Vocabulary Tables in 4-Bit

`embed_tokens` (which doubles as the output layer, `lm_head`) and the per-layer table are 16-bit in the `-text` checkpoint. Google's QAT trained them onto a 4-bit grid as well: every sampled group of 32 holds at most 16 distinct values. `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4` stores them as 4-bit codes with each group's step recovered from the grid, in 16-bit:

| Table | Values equal to the 16-bit QAT table | Worst difference |
| :--- | ---: | ---: |
| `embed_tokens` / `lm_head` | 90.9% | 5.3% of a step |
| per-layer table | 89.7% | 5.3% of a step |

The emb4 checkpoint writes `embed_tokens` and the untied `lm_head` as identical levels and scales, so one copy on the GPU serves both, as the model was trained. With every table in 4-bit, the whole model is **2,483 MiB on the GPU**, and the per-layer lookup happens on the card instead of in CPU memory.

---

#### Step 3 — Multiply Straight From the Packed Weights

Decoding one token multiplies one vector by every weight matrix, so the speed limit is how fast the card reads the weights. A Triton kernel (`w4gemv.py`) reads the packed 4-bit words, unpacks them in registers and multiplies, never writing a 16-bit copy.

Two details carry most of its speed on this card. Each 4-bit value becomes a float by OR-ing it into the bits of 2²³ and subtracting 2²³ + 8, which is exact and skips Turing's slow integer-to-float conversion. And each weight shape gets its own fixed tiling, chosen by searching tile sizes, warps and pipeline stages on this card:

```plaintext
{"summary": {"v4": {"ms_per_token": 9.3, "GBps_effective": 137.8}, "v5": {"ms_per_token": 9.12, "GBps_effective": 140.6}}, "bytes_per_token_MiB": 1222.8}
```

9.1 ms per token for 1,222.8 MiB of packed weights is 140.6 GB/s. The card reads 155.9 GB/s in a plain sum over memory, so the kernel runs at 90% of what the card can read.

---

#### Step 4 — Replay Each Decode Step as One CUDA Graph

A decode step is hundreds of small kernels, and launching them one at a time from Python costs more than running them. The step is compiled with `torch.compile` and captured once as a CUDA graph, then replayed for every token.

Transformers' static cache stands in the way. Its sliding-window layer keeps the cache length as a Python integer: a CUDA graph captures that integer as a constant, so the sliding-window mask points at the wrong position from the fifth token on, and `torch.compile` guards on it, so it recompiles at every token. The engine keeps its own cache instead (Step 7), with every position on the GPU.

---

#### Step 5 — Attention for One Token at a Time

PyTorch's default attention on this card is a memory-efficient kernel built for many query tokens. For one query token it took 13 ms per decode step. Gemma 4 E2B has one key/value head shared by eight query heads, so a Triton kernel (`w4attn.py`) reads each cached key and value once for all eight heads, split across the cache in slices, and combines the slices with an online softmax:

| Attention, per decode step | ms |
| :--- | ---: |
| PyTorch default (memory-efficient kernel) | 12.96 |
| PyTorch plain matmul and softmax (sum of its kernels) | 5.4 |
| **Triton decode kernel** | **0.85** |

The kernel avoids `tl.dot`: on a card without tensor cores, Triton's 16-bit dot product ran at 24 GB/s, against 65 GB/s for a multiply-and-sum over the eight heads.

---

#### 🔎 Tip: Merge the Small Projections

The gate and up projections of each MLP, and the query, key and value projections of each attention layer that computes its own keys and values, read the same input. Concatenated into one matrix each, they become one kernel launch instead of two or three, 65 fewer per token. On this card that was worth about 1 tok/s.

---

#### Step 6 — Read Prompts in 32-Bit on a Card Without Tensor Cores

A prompt is read in 256-token chunks, with the weights unpacked to a full matrix for each multiply. For 16-bit matrix multiplies cuBLAS picks its tensor-core kernels (`s884gemm`), and the TU117 has no tensor cores:

| `[256, 1536] x [1536, 24576]` | TFLOP/s |
| :--- | ---: |
| 16-bit | 0.38–0.42 |
| **32-bit** | **1.57** |

Unpacking to 32-bit for prompt reads doubles their speed: a 4,081-token prompt reads at 226.5 tok/s against 97.7 in 16-bit. Decode stays 16-bit.

---

#### Step 7 — Fit the Full 131,072-Token Context

Gemma 4 E2B keeps a full-length key/value cache in only 3 of its 35 layers, one head of width 512 each: 2 KiB per token per layer. Twelve more layers attend over a 512-token window, and the last 20 reuse earlier layers' keys and values. So the whole 131,072-token context costs 774 MiB:

| | MiB |
| :--- | ---: |
| Model, every weight and table in 4-bit | 2,483 |
| Key/value cache, 131,072 tokens | 774 |
| Peak, reading a 130,981-token prompt and decoding | **3,471** |
| Visible to PyTorch on a 4 GB card | 3,732 |

The engine owns that cache: full layers get one slot per position, sliding layers a 512-slot ring that records each slot's position. Decode speed with the context held, against llama.cpp on the same card:

| Context held | PyTorch | llama.cpp | |
| ---: | ---: | ---: | ---: |
| ~0 | 87.8 tok/s | 82.1 tok/s | 🥇 PyTorch 1.07x |
| ~4k | 80.4 | 76.7 | 🥇 PyTorch 1.05x |
| ~32k | 51.0 | 56.8 | 🥇 llama.cpp 1.11x |
| ~131k | 21.8 | 30.2 | 🥇 llama.cpp 1.39x |

Seven layers attend over the full-length cache every token, at 3.74 ms each at 131,072 tokens in this kernel, which is where llama.cpp pulls ahead at depth. llama.cpp also fits the full context on this card, with its per-layer table in CPU memory.

---

#### Step 8 — Check Long-Context Recall Against llama.cpp

A recall test fills the context with numbered records, each with a random six-digit code, and asks for the code of the record a quarter of the way in. The same prompts, sent to llama-server with thinking off, give the same token counts and the same answers:

| Prompt | Records | PyTorch | llama.cpp |
| ---: | ---: | :--- | :--- |
| 4,081 tokens | 135 | ✅ | ✅ |
| 32,761 tokens | 1,091 | ❌ code of record 723 | ❌ the same code |

At 32,761 tokens a second prompt with different codes was answered correctly, and at 130,981 tokens the answer came from the wrong record. Finding one of thousands of near-identical records is at the edge of what E2B does at this depth, in either engine.

---

#### Step 9 — Check the Answers: GSM8K and BFCL

GSM8K (1,319 grade-school math problems, 2,048-token limit) and BFCL simple (400 single tool calls) run through the same scoring code, question sets and chat template as the llama.cpp results on this card, greedy, and are paired question by question:

| GTX 1650 Ti, Gemma 4 E2B QAT | llama.cpp | PyTorch | Difference (95% range) |
| :--- | ---: | ---: | ---: |
| GSM8K | 0.8992 | 0.8992 | +0.000 (−0.011 to +0.011) |
| BFCL simple | 0.9225 | 0.9200 | −0.003 (−0.007 to +0.000) |

Both engines answer 1,186 GSM8K problems correctly; 26 go each way. On BFCL one question separates them. The 4-bit arithmetic, 16-bit decode and 32-bit prompt reads leave the answers where llama.cpp has them.

---

#### 🔎 Tip: Copy Keys and Values Before Masked Memory-Efficient Attention

For the sliding-window layers a prompt chunk goes to PyTorch's memory-efficient attention with an explicit mask, and the single key/value head is shared by the eight query heads. Broadcast as a view (`expand`, stride 0), those keys and values make torch 2.13's kernel return a wrong last row whenever the chunk holds 33, 65, 97 … queries, on this card:

```plaintext
n=33 Lk=545: bool+expand 1.127  bool contiguous 0.000  float mask 1.205  D=128 1.452  H=1 0.000
```

The last row of a prompt read is the token that picks the first reply token, so this shows up as about one prompt in 32 starting its reply wrong. Copying the keys and values first (`.contiguous()`, at most 768 keys on these layers) gives the right row at every chunk length. Calls with the built-in causal mask and no explicit mask, as on the full-attention layers, return the right rows with the broadcast view.

---

#### Step 10 — Serve It

`openai_server.py` puts the engine behind `/v1/chat/completions`, one request at a time, greedy. Tool definitions go through Gemma 4's chat template and tool calls come back in OpenAI's `tool_calls` field:

```bash
python3 openai_server.py --port 8100 --max-ctx 8192
curl -s localhost:8100/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "What is the weather in Paris right now?"}],
  "tools": [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
  }}]
}'
```

The reply, with its message, finish reason and timings:

```json
{"message": {"role": "assistant", "content": null, "tool_calls": [{"id": "call_5f097a4f09c24b0e8581e124", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]}, "finish_reason": "tool_calls", "usage": {"prompt_tokens": 65, "completion_tokens": 15, "total_tokens": 80}, "timings": {"prompt_ms": 366.8, "decode_ms": 160.3, "decode_tok_per_s": 81.08}}
```

A plain chat question, "In one sentence, what is a GPU?", came back in 34 tokens at 85.23 tok/s. The BFCL and GSM8K scores above were measured through this server. `--max-ctx` defaults to 131,072.

---

#### Run It on Colab

The notebook `local-pytorch-1650ti-2b-q4w4a16/colab/notebooks/01_gemma4_e2b_int4_on_4gb.ipynb` runs the same steps on a Colab T4. It caps PyTorch at the 3,732 MiB a GTX 1650 Ti shows (`torch.cuda.set_per_process_memory_fraction`), so a 16 GB T4 behaves like the 4 GB card for memory, then times 16-bit against 32-bit multiplies and reads prompts with the faster one. The same notebook, run on both cards:

| Same notebook, 3,732 MiB cap | Colab T4 | GTX 1650 Ti |
| :--- | ---: | ---: |
| 16-bit weights | ❌ out of memory | ❌ out of memory |
| Transformers' own 4-bit path | ❌ out of memory | ❌ out of memory |
| Engine, weights + 32k cache | 2,498 + 198 MiB | 2,499 + 198 MiB |
| Matrix multiply, 16-bit / 32-bit | 36.58 / 6.68 TFLOP/s | 0.42 / 1.58 TFLOP/s |
| Decode, short chat | 101.6–117.5 tok/s | 86.1–87.5 tok/s |
| Prompt read, 32,491 tokens | 1,508.2 tok/s (16-bit) | 147.2 tok/s (32-bit) |
| Decode, 32k held | 69.2 tok/s | 50.8 tok/s |
| Peak memory, 32k | 2,836 MiB | 2,908 MiB |
| Recall, 32k | ✅ | ✅ |

The T4's tensor cores make 16-bit prompt reads ten times faster than the laptop's 32-bit ones; decode, limited by how fast the card reads the weights, gains 1.2–1.4x. No Hugging Face token is needed, and every checkpoint is public.

---

#### Compare and Contrast

| GTX 1650 Ti, Gemma 4 E2B QAT | PyTorch, this article | llama.cpp `7182ef7` |
| :--- | :--- | :--- |
| Decode, short chat | 🥇 87.8 tok/s | 82.1 tok/s |
| Decode, 131k context held | 21.8 tok/s | 🥇 30.2 tok/s |
| Prompt read, 4k tokens | 226.5 tok/s, whole prompt | 🥇 279.8 tok/s, 512 tokens at 4k |
| Full 131,072-token context | ✅ every table on the GPU | ✅ per-layer table in CPU memory |
| Weights | Google's QAT grid, 4-bit end to end | Google's QAT grid, exact Q4_0 GGUF |
| Code you change | Python and Triton | C++ and CUDA |

---

#### So, Which One?

For serving E2B on this card, llama.cpp is ready today, holds long context better and reads prompts faster. Plain PyTorch matches it at chat length once the model stays 4-bit end to end, and every piece is Python and Triton you can read and change. The steps carry to any small GPU: keep QAT weights packed, multiply from the packed words, own the cache, and replay decode as a graph.

---

#### Summary

The goal of this article was to run Gemma 4 E2B in plain PyTorch on a 4 GB GTX 1650 Ti. The key to the solution was keeping Google's QAT 4-bit weights packed from disk to arithmetic, with Triton kernels that read them directly and the decode step replayed as one CUDA graph. The results were:

- 🟢 Every 4-bit code matches Google's QAT release; the whole model is 2,483 MiB on the GPU
- 🟢 Decode at 87.8 tok/s at short context, 1.07x llama.cpp on the same card the same day
- 🟢 The full 131,072-token context fits, 3,471 MiB peak of 3,732
- 🟢 Long-context recall matches llama.cpp answer for answer on the prompts both ran
- ❌ 16-bit weights and Transformers' own 4-bit path both run out of memory on this card
- ⚠️ At 32k and 131k tokens held, llama.cpp decodes 1.11x and 1.39x faster
- ⚠️ Prompts read at 226.5 tok/s for 4k tokens; llama.cpp reads faster
- 🟢 GSM8K 0.8992 and BFCL 0.920, level with llama.cpp question for question

Scope: Gemma 4 E2B-it QAT only, text only, one GTX 1650 Ti Max-Q (4 GB) in a laptop, torch 2.13.0+cu130, Triton 3.7.1, transformers 5.16.1, greedy decoding, one request at a time throughout. Decode speeds are 63 to 128 generated tokens after the prompt; llama.cpp `7182ef7` was measured the same day on the same card with `llama-bench`, 64 to 128 generated tokens at a stated depth, so the prompt-read columns are different measurements. The GSM8K and BFCL reference is llama.cpp `77c4540ce` on the same card on 2026-10-14, 8 requests at a time. Recall used four prompts, too few for a rate.

The strategy for running Gemma 4 on a 4 GB GPU in plain PyTorch was validated with an incremental step by step approach.

---

#### References

- Repository: https://github.com/xbill9/gemma4-dev
- PyTorch rig: https://github.com/xbill9/gemma4-dev/tree/main/local-pytorch-1650ti-2b-q4w4a16
- First light, the default path and the QAT check: https://github.com/xbill9/gemma4-dev/blob/main/local-pytorch-1650ti-2b-q4w4a16/benchmarks/runs/2026-10-23-first-light-1650ti/REPORT.md
- 4-bit tables and the 4-bit kernel: https://github.com/xbill9/gemma4-dev/blob/main/local-pytorch-1650ti-2b-q4w4a16/benchmarks/runs/2026-10-23-lmhead-kernel-1650ti/REPORT.md
- Decode attention and kernel tuning: https://github.com/xbill9/gemma4-dev/blob/main/local-pytorch-1650ti-2b-q4w4a16/benchmarks/runs/2026-10-23-close-gap-1650ti/REPORT.md
- Long context: https://github.com/xbill9/gemma4-dev/blob/main/local-pytorch-1650ti-2b-q4w4a16/benchmarks/runs/2026-10-23-long-context-1650ti/REPORT.md
- Colab notebook: https://github.com/xbill9/gemma4-dev/blob/main/local-pytorch-1650ti-2b-q4w4a16/colab/notebooks/01_gemma4_e2b_int4_on_4gb.ipynb
- Answer quality and the attention fix: https://github.com/xbill9/gemma4-dev/blob/main/local-pytorch-1650ti-2b-q4w4a16/benchmarks/runs/2026-10-23-accuracy-1650ti/REPORT.md
- Decoder checkpoint: https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text
- 4-bit tables checkpoint: https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4
- Google's QAT E2B: https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-unquantized
- Do Drafters Work on Small Models? (the llama.cpp side of this card): https://dev.to/gde/do-drafters-work-on-small-models-gemma-4-e2b-and-e4b-on-two-small-gpus-1b4g
- llama.cpp: https://github.com/ggml-org/llama.cpp
