---
library_name: gguf
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: text-generation
base_model:
- google/gemma-4-12B-it-qat-q4_0-unquantized-assistant
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

# Gemma 4 12B-it QAT assistant (drafter), exact Q4_0 GGUF (unofficial)

**This is an unofficial build, made and published independently of Google.** It is a llama.cpp
GGUF of Google's speculative-decoding drafter for Gemma 4 12B-it QAT,
[`google/gemma-4-12B-it-qat-q4_0-unquantized-assistant`](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized-assistant),
with every weight matrix stored at 4 bits on the grid quantization-aware training produced. It is
253,708,960 B, against 465,128,096 B for the Q8_0 file llama.cpp's converter writes. Google's
model card is kept unchanged in `ORIGINAL_README.md`.

Use it with a Gemma 4 12B-it **QAT** target, for example
[`xbill9/gemma-4-12B-it-qat-q4_0-exact-gguf`](https://huggingface.co/xbill9/gemma-4-12B-it-qat-q4_0-exact-gguf)
or [`google/gemma-4-12B-it-qat-q4_0-gguf`](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf).
Google's card says the drafter must match the target's QAT precision.

```
llama-server -m gemma-4-12B-it-q4_0-exact.gguf -ngl 99 -fa 1 -c 81920 -ctk q4_0 -ctv q4_0 \
  -md gemma-4-12B-it-qat-assistant-q4_0-exact.gguf --spec-type draft-mtp -ngld all --spec-draft-n-max 2
```

Measured on llama.cpp build `7182ef71e` only.

| | Q8_0 (llama.cpp converter) | This GGUF |
|---|---:|---:|
| File size | 465,128,096 B | **253,708,960 B** |
| Drafter weights in VRAM | 428.53 MiB | **226.90 MiB** |
| Generation with drafter, 12B target, RTX 3050 | 53.4 tok/s (1.90x) | **56.4 tok/s (2.01x)** |
| Drafted tokens accepted | 0.661 | 0.658 |
| VRAM, target + drafter, `-c 8192` | 7,771 MiB | **7,569 MiB** |

Without a drafter the target runs at 28.1 tok/s in 7,255 MiB. **This file is 1.06x faster than
the Q8_0 one at the same acceptance** (1.04–1.07x on each prompt). The 12B drafter is about
4.7x the size of the E2B and E4B drafters, so its own weight reads likely weigh more in each
drafted step (inferred, not measured); for those smaller drafters Q4_0 and Q8_0 ran level.

| drafted tokens per step | 1 | 2 | 3 |
|---|---:|---:|---:|
| Generation, x no drafter | 1.62x | **2.01x** | 2.02x |
| Drafted tokens accepted | 0.759 | 0.658 | 0.561 |

Per prompt at 2 drafted tokens: code 2.40x, facts 2.30x, chat 1.94x, explanation 1.93x,
summary 1.82x, story 1.75x. Sampled at temperature 0.8: 1.90x.

Measured on an RTX 3050 OEM (8 GB), llama.cpp `7182ef71e`, target
`xbill9/gemma-4-12B-it-qat-q4_0-exact-gguf`, flash attention, f16 KV cache, `-c 8192`: six
prompts (code, chat, story, explanation, facts, summary), 256 tokens, greedy, median of three
runs, geometric mean over prompts. No drafter ran first and last, at 28.1 tok/s both times.

**On an 8 GB card the drafter costs context.** With this file the 12B target loads at up to
24,576 tokens of f16 context (7,820 MiB); without a drafter, 40,960. With a q8_0 KV cache, 49,152
fit with the drafter, and with q4_0, 81,920, with the same GSM8K and BFCL scores as f16. The drafter
still gives 1.60x with 65,536 tokens in context
([run](https://github.com/xbill9/gemma4-dev/blob/main/local-llamacpp-3050oem-12b-q4_0/benchmarks/runs/2026-10-17-kvcache-12b-3050oem/REPORT.md)).

**Greedy text changes.** With the drafter, greedy replies repeat exactly from run to run, but
they differ from the target's greedy replies without a drafter in 18 of 18 at 2 drafted tokens
(9 of 18 at 1). The target still checks every drafted token; the differences likely come from
rounding in the batched check (inferred, not measured).

**The answers do not change.** One request at a time at `-c 24576`, the 12B target scores the
same with and without this drafter, and finishes GSM8K 2.34x faster with it:

| | with this drafter | without a drafter |
| --- | ---: | ---: |
| GSM8K (1,319 questions) | 0.9598 | 0.9629 |
| BFCL simple (400) | 0.9500 | 0.9475 |

Paired by question the differences are −0.0030 (95% range −0.0083 to +0.0023) and +0.0025
(0 to +0.0075). Method and per-question records:
[2026-10-16-accuracy-12b-3050oem](https://github.com/xbill9/gemma4-dev/tree/main/local-llamacpp-3050oem-12b-q4_0/benchmarks/runs/2026-10-16-accuracy-12b-3050oem).

#### What is in it

- **Built from** [`google/gemma-4-12B-it-qat-q4_0-unquantized-assistant`](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized-assistant)
  at revision `1893406`: 4 transformer layers of width 1024 (3 sliding-window, 1 full
  attention), with queries only; keys and values come from the target's cache. Two projections
  join it to the target's 3840-wide hidden state. The output head is the tied 262,144-token
  embedding.
- **Every weight matrix as Q4_0** (the embedding, 4 x attention q/o, 4 x MLP gate/up/down, both
  projections), each block's trained step recovered: largest magnitude divided by 8, 7, … 1,
  whichever puts all 32 values on an integer level, refined by least squares and stored as the
  fp16 block scale. No block was off the grid. 96.7–97.1% of values per tensor kind reproduce
  the bf16 source exactly once rounded back to bf16.
- **Metadata** (hyperparameters, tokenizer, chat template) is llama.cpp's converter output,
  `general.file_type` set to Q4_0. Norms and scales stay F32.
- **Text only.** The drafter has no vision or audio parts.

#### Files

| file | bytes | SHA-256 |
|---|---:|---|
| `gemma-4-12B-it-qat-assistant-q4_0-exact.gguf` | 253,708,960 | `f22f6366ca76102ac1d365a7ee331e80d4b1e986ddd7f0149c6d3ca5c7a87985` |

`evidence/` holds the build script (`gguf_exact.py`), its per-tensor report (`build-report.json`),
the speed run's script and per-prompt table (`spec_ab.py`, `speed-summary.txt`) and the
converter log.

#### License

Apache 2.0, as Google's source. Gemma 4 terms: https://ai.google.dev/gemma/docs/gemma_4_license
