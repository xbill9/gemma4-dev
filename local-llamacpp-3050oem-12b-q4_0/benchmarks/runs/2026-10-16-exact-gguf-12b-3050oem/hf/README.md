---
library_name: gguf
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: text-generation
base_model:
- google/gemma-4-12B-it-qat-q4_0-unquantized
base_model_relation: quantized
tags:
- gemma4
- gguf
- llama.cpp
- q4_0
- qat
- cuda
- text-only
---

# Gemma 4 12B-it QAT, exact Q4_0 GGUF, text only (unofficial)

**This is an unofficial build, made and published independently of Google.** It is a
llama.cpp GGUF of Gemma 4 12B-it in which every 4-bit weight is the value Google's
quantization-aware training (QAT) produced. Every weight matrix is stored at 4 bits,
including the embedding table. It is 6.72 GB, against 6.98 GB for Google's
[`gemma-4-12B-it-qat-q4_0-gguf`](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf),
and follows the unquantized model more closely. Google's model card is kept unchanged in
`ORIGINAL_README.md`.

**It runs a QAT 12B on an 8 GB card with every layer on the GPU**, at up to 40,960 tokens of
context, against 28,672 for Google's file. With the 12B drafter,
[`xbill9/gemma-4-12B-it-qat-assistant-q4_0-exact-gguf`](https://huggingface.co/xbill9/gemma-4-12B-it-qat-assistant-q4_0-exact-gguf),
it generates at 56 tok/s on an RTX 3050, with up to 81,920 tokens of context in a q4_0 KV cache.

The E2B and E4B counterparts, built the same way, are
[`xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf) and
[`xbill9/gemma-4-E4B-it-qat-q4_0-exact-gguf`](https://huggingface.co/xbill9/gemma-4-E4B-it-qat-q4_0-exact-gguf).

| | Google Q4_0 GGUF | This GGUF |
|---|---|---|
| File size | 6,975,879,296 B | **6,716,356,736 B** |
| Embedding table (also the output layer) | Q6_K | Q4_0 |
| Mean KL divergence from bf16, CPU | 0.543225 ± 0.019046 | **0.124504 ± 0.009525** |
| 99th percentile KL divergence, CPU | 6.217015 | **1.914886** |
| Same top token as bf16, CPU | 71.77 ± 0.71 % | **89.14 ± 0.49 %** |
| Perplexity relative to bf16, CPU | 1.3855 ± 0.0449 | **1.2567 ± 0.0248** |
| Generation, 128 tokens, RTX 3050 | 27.68 tok/s | **28.64 tok/s** (1.035x) |
| Prompt processing, 512 tokens, RTX 3050 | 979.7 tok/s | **987.2 tok/s** (1.008x) |
| Weights in VRAM | 6637.69 MiB | **6390.19 MiB** |
| Most context that loads on 8 GB, f16 KV | 28,672 tokens | **40,960 tokens** |

Measured with llama.cpp build `7182ef71e` on an RTX 3050 OEM (8 GB, Ampere) in an i7-5820K
desktop. Divergence and perplexity: `llama-perplexity` on wikitext-2 test, 16
chunks of 512 tokens, on the CPU, against a bf16 GGUF of the same QAT weights with identical
metadata. **Compare the two columns with each other, not with other models' figures:** the 12B
instruction-tuned model reads raw Wikipedia text poorly (bf16 perplexity 312, against 40 for E4B
on the same test), and in that regime every divergence is larger. The E4B build's mean divergence
on this test is 0.0010. Speed: `llama-bench`, 5 repetitions, four passes alternating which file runs
first (Google, this, this, Google), each starting at or below 40 °C; the generation figures were
the same in both passes of each file. Context: the largest `-c` at which `llama-server -ngl 99
-fa 1` loaded. Logs are in `evidence/`.

#### Fitting it on an 8 GB card

The 12B has no per-layer embedding table, so unlike E2B and E4B almost all of the file goes to
the GPU: 6,390 MiB of weights. The embedding table is also the output layer, so it sits on the
GPU as well as being read from the memory-mapped file for input; storing it as Q4_0 saves
247.5 MiB of VRAM. What is left goes to the context cache, which costs 16 KiB per token for the
8 full-attention layers plus a fixed 480 MiB for the 40 sliding-window layers (f16).

| `llama-server -ngl 99 -fa 1`, RTX 3050 8 GB | f16 KV cache | q8_0 KV cache | q4_0 KV cache |
|---|---:|---:|---:|
| this GGUF | 40,960 tokens | 65,536 tokens | 131,072 tokens |
| this GGUF + exact Q4_0 drafter | 24,576 tokens | 49,152 tokens | 81,920 tokens (7,853 MiB) |
| Google's GGUF | 28,672 tokens | not tested | not tested |

GSM8K, BFCL simple and recall of codes spread through a long document are level across the f16,
q8_0 and q4_0 caches, with and without the drafter (paired ranges include zero). With the drafter,
the q4_0 cache generates 55.1 / 42.4 / 37.8 / 30.8 tok/s with 0 / 16k / 32k / 65k tokens in
context, 0.85x f16 at 16k
([run](https://github.com/xbill9/gemma4-dev/blob/main/local-llamacpp-3050oem-12b-q4_0/benchmarks/runs/2026-10-17-kvcache-12b-3050oem/REPORT.md)).

Generation without a drafter: 28.6 / 26.8 / 26.3 tok/s with 0 / 8,192 / 16,384 tokens already in
context. The card has to be otherwise idle: these figures leave 370–640 MiB free.

#### What is in it

- **Language model only.** Vision lives in Google's separate
  `mmproj-gemma-4-12b-it-qat-q4_0.gguf`, which is not included here.
- **Metadata copied byte for byte** from Google's GGUF at revision `29d0977`: tokenizer,
  chat template, hyperparameters and tensor order.
- **Weights rebuilt** from
  [`google/gemma-4-12B-it-qat-q4_0-unquantized`](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized)
  at revision `b6ed862`: all 328 layer matrices (attention q/k/v/o, MLP gate/up/down; the 8
  full-attention layers have no v, as in the source) and the embedding table, as Q4_0.
- **Copied unchanged:** the norms, scales and RoPE frequencies (F32).

#### Why it is smaller and closer

QAT trained these weights on a 4-bit grid: in every block of 32 values, each value is a block
scale times an integer from -8 to 7. That holds for the embedding table as well as the layers.
Google's GGUF stores the embedding table as Q6_K, which is where the 0.26 GB goes.

For the layer matrices, Google's GGUF uses llama.cpp's standard Q4_0 step, the block's largest
magnitude divided by 8. QAT often puts a block's largest value on a lower level, and in those
blocks the standard step moves values off the trained grid. This build recovers each block's
trained step instead (largest magnitude divided by 8, 7, … 1, whichever puts every value on an
integer level, then refined by least squares) and stores it as the fp16 block scale.

Every rebuilt block landed on a 4-bit grid, and 96.77–96.88% of values per tensor kind are
bit-identical to the source after bf16 rounding. The rest differ by the fp16 rounding of the
scale. Per-tensor counts are in `evidence/build_report.json`.

The generation gain is smaller than the E4B build's (1.06x) and E2B's (1.11x): the output layer
is a smaller share of what a 12B reads per token (inferred, not measured).

#### Benchmarks

One request at a time on an RTX 3050 (8 GB), llama.cpp `7182ef71e`, `-c 24576`, thinking off,
with and without the exact Q4_0 12B drafter
([`xbill9/gemma-4-12B-it-qat-assistant-q4_0-exact-gguf`](https://huggingface.co/xbill9/gemma-4-12B-it-qat-assistant-q4_0-exact-gguf)).
Gemma 4 E4B (exact Q4_0) on the same card for scale, on llama.cpp `77c4540ce`:

| | this file, drafter | this file, no drafter | E4B, drafter | E4B, no drafter |
| --- | ---: | ---: | ---: | ---: |
| GSM8K (1,319 questions) | 0.9598 | 0.9629 | 0.9295 | 0.9310 |
| BFCL simple (400) | 0.9500 | 0.9475 | 0.9125 | 0.9125 |

Paired by question, the 12B is ahead of E4B by 0.030 on GSM8K (95% range 0.018 to 0.043) and
0.0375 on BFCL (0.010 to 0.065) with the drafter. The drafter changes neither score and finishes
GSM8K 2.34x faster. Method and per-question records:
[2026-10-16-accuracy-12b-3050oem](https://github.com/xbill9/gemma4-dev/tree/main/local-llamacpp-3050oem-12b-q4_0/benchmarks/runs/2026-10-16-accuracy-12b-3050oem).

#### Running it

```bash
llama-server -m gemma-4-12B-it-q4_0-exact.gguf -ngl 99 -fa 1 -c 40960 --jinja
# with the drafter:
llama-server -m gemma-4-12B-it-q4_0-exact.gguf -ngl 99 -fa 1 -c 81920 -ctk q4_0 -ctv q4_0 --jinja \
  -md gemma-4-12B-it-qat-assistant-q4_0-exact.gguf --spec-type draft-mtp -ngld all --spec-draft-n-max 2
```

`llama-server` turns thinking on by default, so a reply lands in `reasoning_content` and a short
`max_tokens` can leave `content` empty. To get direct answers, start the server with
`--reasoning off`, or send `"chat_template_kwargs": {"enable_thinking": false}` with a request.

#### Rebuilding it

```bash
hf download google/gemma-4-12B-it-qat-q4_0-unquantized --revision b6ed862 --local-dir src
hf download google/gemma-4-12B-it-qat-q4_0-gguf gemma-4-12b-it-qat-q4_0.gguf --revision 29d0977 --local-dir gguf
python3 gguf_exact.py src gguf/gemma-4-12b-it-qat-q4_0.gguf gemma-4-12B-it-q4_0-exact.gguf
```

Needs only `numpy`; took 17 minutes and peaked at 20.5 GiB of resident memory (mostly the
memory-mapped source) on a 6-core desktop. The output is deterministic: SHA-256
`038337ee0375f602cc39eac7575b1c562cf7faaffcdbf538909c243862752731`. `--bf16` writes the reference
instead: the same file with the source bf16 weights. The script stops if any block fails to land
on a 4-bit grid. It is the E2B and E4B builds' script, unchanged apart from its docstring.

#### Limitations

- Text only.
- Tested with llama.cpp on one machine: speed, memory and three greedy chat prompts on its GPU
  (RTX 3050, CUDA), divergence on its CPU (AVX2). Other backends and runtimes were not tested,
  and divergence was not measured on CUDA.
- Benchmarks: GSM8K and BFCL simple only (below), on one machine with llama.cpp `7182ef71e`.
- The divergence figures cover 8,192 tokens of English Wikipedia text, a poor fit for this
  instruction-tuned model (see above).
- Unofficial. Report problems here, not to Google.

#### License and attribution

Gemma 4 is released by Google DeepMind under the
[Apache 2.0 license](https://ai.google.dev/gemma/docs/gemma_4_license). This repository
redistributes Google's weights in a changed storage format, under the same license. The weights,
training, tokenizer, chat template and the original model card are Google DeepMind's; this build
and card are not affiliated with or endorsed by Google.
