---
library_name: gguf
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: text-generation
base_model:
- google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant
base_model_relation: quantized
tags:
- gemma4
- gguf
- llama.cpp
- q4_0
- qat
- speculative-decoding
- mtp
- cuda
- text-only
---

# Gemma 4 E4B-it QAT assistant (drafter), exact Q4_0 GGUF (unofficial)

**This is an unofficial build, made and published independently of Google.** It is a llama.cpp
GGUF of Google's speculative-decoding drafter for Gemma 4 E4B-it QAT,
[`google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant`](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant), with every weight matrix stored at 4 bits on the
grid quantization-aware training produced. It is 59,678,368 B, against 98,672,288 B for the
Q8_0 file llama.cpp's converter writes, and follows Google's weights more closely. Google's model
card is kept unchanged in `ORIGINAL_README.md`.

Use it with a Gemma 4 E4B-it **QAT** target, for example
[`xbill9/gemma-4-E4B-it-qat-q4_0-exact-gguf`](https://huggingface.co/xbill9/gemma-4-E4B-it-qat-q4_0-exact-gguf) or
[`google/gemma-4-E4B-it-qat-q4_0-gguf`](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf). Google's card says the drafter must
match the target's QAT precision.

```
llama-server -m <E4B QAT Q4_0 target>.gguf -ngl 99 -fa 1 \
  -md gemma-4-E4B-it-qat-assistant-q4_0-exact.gguf --spec-type draft-mtp -ngld all --spec-draft-n-max 2
```

Needs a llama.cpp with Gemma 4 MTP support (`#23398`, June 2026). Measured on builds `77c4540ce`
(RTX 3050) and `f95b0d9` (GTX 1650 Ti).

| | Q8_0 (llama.cpp converter) | This GGUF |
|---|---:|---:|
| File size | 98,672,288 B | **59,678,368 B** |
| Relative error against Google's bf16 weights | 0.00486 | **0.00138** |
| Weights bit-identical (dequantized, fp32) | 11.4% | **28.7%** |
| Generation with drafter, E4B target, RTX 3050 | 100.4 tok/s (1.72x) | **100.0 tok/s (1.72x)** |
| Drafted tokens accepted | 0.487 | 0.481 |
| VRAM, target + drafter, `-c 8192` | 4537 MiB | **4499 MiB** |

Speed is level with the Q8_0 file and acceptance is the same. The gain is size, memory and
fidelity. Google's bf16 weights, served unchanged, accept the same share of tokens
(0.484) and run slower (92.3 tok/s, 1.58x), so drafter precision does not
limit acceptance. Without a drafter the target runs at 58.2 tok/s.

Measured on an RTX 3050 OEM (8 GB), llama.cpp `77c4540ce`, target
`xbill9/gemma-4-E4B-it-qat-q4_0-exact-gguf`, flash attention, f16 KV cache: six prompts (code, chat, story, explanation,
facts, summary), 256 tokens, greedy, median of three runs, geometric mean over prompts; 2 drafted
tokens per step. Every drafter precision produced the same greedy replies.

#### On a GTX 1650 Ti (4 GB laptop GPU)

E4B target `xbill9/gemma-4-E4B-it-qat-q4_0-exact-gguf`, llama.cpp `f95b0d9`, same prompts and method,
2 drafted tokens per step (level with 1 token on this card):

| | Q8_0 (llama.cpp converter) | This GGUF | bf16 (Google's weights) |
|---|---:|---:|---:|
| Generation with drafter | 53.8 tok/s (1.40x) | **54.5 tok/s (1.42x)** | 49.8 tok/s (1.30x) |
| Drafted tokens accepted | 0.493 | 0.491 | 0.502 |
| VRAM, target + drafter, `-c 8192` | 2,965 MiB | **2,927 MiB** | 3,035 MiB |

Without a drafter the target runs at 38.4 tok/s in 2,853 MiB. With this drafter at
1 / 2 / 3 drafted tokens: 1.43x / 1.43x / 1.32x
(RTX 3050: 1.72x at 2 tokens). Acceptance is about the same on both cards, so the smaller gain likely
comes from each drafted step costing a larger share of a target step on this card (inferred, not
measured). Use `--spec-draft-n-max 2` on a card of this class.

#### Answer quality with the drafter serving

The target model checks every drafted token, so the drafter changes speed and leaves the target's
choices to the target, apart from rounding in the batched check. E4B target, llama.cpp
`77c4540ce`, 8 requests at once, greedy, 2 drafted tokens:

| | no drafter | Q8_0 drafter | this drafter |
|---|---:|---:|---:|
| GSM8K, 1,319 problems | 0.9325 | 0.9310 | 0.9325 |
| BFCL simple, 400 tool calls | 0.9100 | 0.9150 | 0.9150 |
| Drafted tokens accepted over both | — | 0.8374 | 0.8382 |

Paired on the same problems, every drafted arm's difference from no drafter has a 95% bootstrap
range that includes zero.

#### What is in it

- **Built from** [`google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant`](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant) at revision `27f8d20`:
  4 transformer layers of width 256 (3 sliding-window, 1 full attention), with queries only;
  keys and values come from the target's cache. Two projections join it to the target's
  2560-wide hidden state. The output head is the tied 262,144-token embedding.
- **Every weight matrix as Q4_0** (the embedding, 4 x attention q/o, 4 x MLP gate/up/down, both
  projections), each block's trained step recovered: largest magnitude divided by 8, 7, … 1,
  whichever puts all 32 values on an integer level, refined by least squares and stored as the
  fp16 block scale. No block was off the grid. 96.5–97.1% of values per tensor reproduce the
  bf16 source exactly once rounded back to bf16.
- **Metadata** (hyperparameters, tokenizer, chat template) is llama.cpp's converter output,
  `general.file_type` set to Q4_0. Norms and scales stay F32.
- **Left out by llama.cpp's converter:** the checkpoint's `masked_embedding` centroids and token
  ordering, a clustered shortcut for the output head. llama.cpp scores the full vocabulary instead.
- **Text only.** The drafter has no vision or audio parts.

#### Files

| file | bytes | SHA-256 |
|---|---:|---|
| `gemma-4-E4B-it-qat-assistant-q4_0-exact.gguf` | 59,678,368 | `dc965cb963e56b86fbc04db8efec83163886085e88514574650a57fa55a2ee24` |

`evidence/` holds the build script (`gguf_exact.py`), its per-tensor report, the RTX 3050 speed
run's per-prompt table (`speed-summary.txt`) and the GTX 1650 Ti tables
(`speed-summary-1650ti-precision.txt`, `speed-summary-1650ti-draft-length.txt`).

#### License

Apache 2.0, as Google's source. Gemma 4 terms: https://ai.google.dev/gemma/docs/gemma_4_license
