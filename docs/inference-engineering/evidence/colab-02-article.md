---
title: "Gemma 4 E2B in Pure JAX on a Colab TPU: Google's 4-Bit Export Against an Exact Repack"
published: false
description: "A Colab notebook for the AI GDE Marathon that loads three Gemma 4 E2B checkpoints into a pure-JAX engine on one TPU v5e chip and measures, on the reader's own chip, how far each 4-bit build sits from the weights Google trained. The repack holds the trained grid and comes out 342.6x closer to the QAT model at the same speed and a smaller download."
tags: jax, tpu, gemma, machinelearning
cover_image: https://raw.githubusercontent.com/xbill9/colab-jax/main/article/devto-cover.c35e91a0.jpg
---

This article provides a step by step guide to a Colab notebook that serves Gemma 4 E2B on a single TPU v5e chip with a pure-JAX engine and compares two 4-bit builds of the same model against the weights Google trained. Every number below was measured in the notebook on a Colab v5e-1 runtime, and the executed notebook is committed.

Google ships E2B in 4 bits as `gemma-4-E2B-it-qat-w4a16-ct`. Its export rounds every weight a second time, onto a grid the model never trained on. A repack that stores the trained grid instead lands 342.6 times closer to the original model in next-token predictions, matches its top token 99.29% of the time against 85.96%, writes the same 128-token story word for word, runs at the same speed and downloads 0.8 GB less.

https://colab.research.google.com/github/xbill9/colab-jax/blob/main/notebooks/02_e2b_repack_vs_stock_on_jax.ipynb

---

#### The Marathon

This notebook is an entry in the AI GDE Marathon: JAX on TPU Tutorial, a series of Apache 2.0 Colab notebooks covering JAX on the TPU backend. Colab's TPU runtimes are the same v5e and v6e silicon the source measurements came from, so every cell measures on the reader's chip and prints the source result beside it as context.

---

#### Why Two 4-Bit Builds of One Model?

Gemma 4 E2B was trained with quantization-aware training (QAT): during training every weight was held on a 4-bit grid, one scale per group of 32 values, so the model learned to work with exactly those values. Google publishes the result twice:

| Build | Hugging Face | What it stores |
|---|---|---|
| QAT reference | `google/gemma-4-E2B-it-qat-q4_0-unquantized` | the trained grid values, in bf16 |
| Google's export | `google/gemma-4-E2B-it-qat-w4a16-ct` | int4, each group re-rounded with step = largest weight ÷ 7.5 |
| The repack | `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct` | int4, the trained levels and the trained step |

Both 4-bit builds use the same compressed-tensors W4A16 format and the same shapes, so any loader that reads one reads the other. The difference is which numbers are inside.

---

#### At This Point You Should Have…

- A Google account with Colab access to a TPU runtime
- About 40 minutes, most of it downloading 26 GB and compiling

No Hugging Face token is needed: all three checkpoints are public and ungated, and the engine is a public GitHub repo.

---

#### Step 1 — Open the Notebook on a TPU

Open the notebook from the link above, then Runtime → Change runtime type → v5e-1 TPU, then Runtime → Run all. Cell 1.1 records the chip:

```text
JAX 0.7.2
1 device(s): TPU v5 lite (tpu)
HBM 16.91 GB   host RAM 50.5 GB   free disk 195 GB
```

Cell 1.2 clones the engine, [`tpu-jax`](https://github.com/xbill9/tpu-jax), at a pinned commit. It is a Gemma 4 E2B decoder in pure JAX with no PyTorch in the path, and its 4-bit path unpacks each weight with plain XLA operations before multiplying.

---

#### Step 2 — Download the Three Checkpoints

Each checkpoint is pinned to a Hub revision, so the bytes you download are the bytes measured here. Cell 2.2 then indexes every tensor:

```text
qat      1951 tensors     0 packed int4   lm_head stored: False
stock    2504 tensors   276 packed int4   lm_head stored: True
repack   2503 tensors   276 packed int4   lm_head stored: False

stock lm_head identical to embed_tokens: True   (0.805 GB)
stock minus repack on disk: 0.805 GB
```

Both 4-bit builds pack the same 276 linear layers. Google's export also stores `lm_head.weight`, a byte-for-byte copy of the token embedding, although the config ties the two. That one tensor is the whole 0.8 GB difference in download size.

---

#### Step 3 — Are the Weights the Trained Ones?

Cell 3.1 reads 24 tensors from the first, middle and last layers, every projection kind and 5.7 million groups of 32, from all three files. It unpacks each 4-bit build with the engine's own function and compares the result with the QAT values:

```text
all 24 tensors
  stock   groups 5,738,496   rel err 0.0667   identical 12.08%   step=max/7.5 100.00%   peak on a level 0.00%
  repack  groups 5,738,496   rel err 0.0019   identical 73.65%   step=max/7.5 0.00%   peak on a level 99.92%

  repack: level the largest weight of each group sits on, share of groups
   1: 0.0%   2: 0.0%   3: 0.0%   4: 0.0%   5: 0.0%   6: 0.0%   7: 39.5%   8: 60.5%
```

Google's export uses largest weight ÷ 7.5 as the step in every group, and its values sit 6.7% from the trained ones. The repack sits 0.19% away, three values in four are bit-for-bit the trained value, and the rest differ only by the bf16 rounding of the stored step.

The last line explains why the trained step has to be stored. If a group's largest weight sits on level *m* of the trained step *d*, the ÷ 7.5 rule gives a step of *m·d* / 7.5, which never equals *d* because *m* is a whole number from 1 to 8. The largest weight sits on level 7 in 39.5% of groups and level 8 in the other 60.5%, so a fixed ÷ 8 rule would be right for only some of them.

---

#### Step 4 — Does the Difference Reach the Output?

Cells 4.2 to 4.4 load each model in turn, feed all three the same 8 WikiText-2 sequences of 512 tokens, and keep each model's predicted next-token distribution at every position. Cell 4.5 compares each 4-bit build with the QAT model on the same 4,096 positions:

- **KL divergence:** how far the 4-bit distribution is from the QAT one; zero means identical
- **same top token:** share of positions where both models rank the same token first
- **perplexity ratio:** the 4-bit model's perplexity on the text divided by the QAT model's

```text
build     mean KL  same top token  perplexity ratio   per-sequence mean KL, min to max
stock     0.07141          85.96%            1.0643   0.05807 to 0.08348
repack    0.00021          99.29%            0.9994   0.00015 to 0.00028

positions where the repack is closer to the QAT model than the stock build: 100.0% of 4,096
stock KL / repack KL: 342.6x
```

The repack is closer to the QAT model at every one of the 4,096 positions, and the gap holds on every sequence: the stock build's best sequence is still 200 times further off than the repack's worst. Google's export also predicts the text 6.4% worse than the model it was made from.

---

#### Step 5 — Speed, Memory and What Each One Writes

Cell 5.1 decodes a fixed prompt, "The history of the Roman Empire", for 128 greedy tokens, once to compile and three times timed:

```text
build    disk GB  weights GB  decode tok/s  tokens equal to QAT greedy
qat        10.21        9.26         135.0                  128 of 128
stock       8.32        6.56          93.7                    5 of 128
repack      7.51        6.56          93.3                  128 of 128

repack / stock decode speed: 0.996x
```

The two 4-bit builds tie on speed and put the same 6.56 GB on the chip, because the engine never loads the duplicate `lm_head`. The repack writes the QAT model's story token for token. Google's export opens with the same five words, " is a vast and complex", then writes "narrative" where the QAT model writes "tapestry", and the two stories diverge from there.

The bf16 QAT model decodes faster than either 4-bit build in this engine, because the engine unpacks every 4-bit weight at every step. The engine can unpack once at load instead (`dequant_at_load=True`), which trades the HBM saving for that speed.

---

#### 🔎 Tip: Compare Inside a Column Pair

Cell 6.1 prints the earlier vLLM measurements beside the ones from the notebook:

```text
                                           source (vLLM)               today (pure JAX)
                                            stock         repack          stock         repack
relative error vs QAT values        0.0665-0.0667            n/a         0.0667         0.0019
groups with step = max/7.5                100.00%            n/a        100.00%          0.00%
mean KL vs QAT model                          n/a            n/a        0.07141        0.00021
same top token as QAT model                   n/a            n/a         85.96%         99.29%
test suite, 3,880 records                   65.5%          67.8%        not run        not run
download, GB                                 8.32           7.51           8.32           7.51
decode tok/s, 1 request                     136.6          136.5           93.7           93.3
decode speed, repack / stock                              0.999x                        0.996x
```

The absolute speeds differ between the pairs because the engines differ: vLLM's int4 kernel against a reference unpack-then-multiply in pure JAX. The ratio inside each pair is what carries, and it is 0.999x under vLLM and 0.996x here. The same holds for quality: under vLLM on a 3,880-record test suite the repack scored 67.8% against 65.5%, 2.4 points higher (95% range +1.4 to +3.4) and within 0.6 points of the bf16 release.

---

#### Compare and Contrast

| | Google's export | The repack |
|---|---|---|
| Error against the trained weights | 🔴 6.7% | 🟢 0.19% |
| Mean KL against the QAT model | 🔴 0.07141 | 🟢 0.00021 |
| Same top token as the QAT model | 85.96% | 🟢 99.29% |
| Greedy story tokens equal to QAT | 🔴 5 of 128 | 🟢 128 of 128 |
| Download | 8.32 GB | 🟢 7.51 GB |
| Weights on the chip | 6.56 GB | 6.56 GB |
| Decode tok/s, one request | 93.7 | 93.3 |
| Format | compressed-tensors W4A16 | compressed-tensors W4A16 |

---

#### So, Which One?

The repack. It holds the weights Google trained, runs at the same speed in the same format, puts the same bytes on the chip and is 0.8 GB smaller to download. Any loader that reads Google's `-qat-w4a16-ct` reads it unchanged, so switching is a change of repo name.

---

#### Summary

The goal of this notebook was to measure, on a reader's own Colab TPU, whether Google's 4-bit Gemma 4 E2B export holds the weights the model was trained on, and what a repack that holds them exactly changes. The key to the solution was putting all three checkpoints through one pure-JAX loader, one forward pass, one text and one prompt, with the QAT checkpoint as the reference for both 4-bit builds. The results were:

- 🟢 The repack holds the trained grid: 0.19% error, 73.65% of values bit-identical, the largest weight on a trained level in 99.92% of groups
- 🟢 Its next-token predictions are 342.6 times closer to the QAT model's than the export's, at every one of 4,096 positions
- 🟢 It writes the QAT model's 128-token greedy story token for token
- 🟢 Same speed (0.996x) and same chip memory (6.56 GB), 0.8 GB smaller to download
- ⚠️ In this engine both 4-bit builds decode slower than bf16 (93 against 135 tokens per second), because the weights are unpacked at every step
- ❌ Google's export re-rounds every group with step = largest ÷ 7.5: 6.7% error, 85.96% top-token agreement and a 6.4% higher perplexity than the model it came from

Scope: one Colab v5e-1 runtime (one TPU v5 lite chip, 16.91 GB HBM), JAX 0.7.2, `tpu-jax` at commit `4b9f8e9`, run on 2026-10-09. The output comparison uses 8 WikiText-2 test sequences of 512 tokens, and speed is the median of three 128-token greedy runs of one prompt. The 3,880-record test suite and the vLLM speeds are earlier results on a v5e chip, cited as context. The repack is unofficial and derived from Google's release under Apache 2.0. Parts of the analysis and writing were done with AI assistance (Claude); every figure comes from the committed executed notebook.

The comparison of Google's 4-bit Gemma 4 E2B export against an exact repack was validated on a Colab TPU with an incremental step by step approach.

---

#### References

- The notebook in Colab: https://colab.research.google.com/github/xbill9/colab-jax/blob/main/notebooks/02_e2b_repack_vs_stock_on_jax.ipynb
- Source, build tools and the executed run: https://github.com/xbill9/colab-jax
- The executed notebook from this article: https://github.com/xbill9/colab-jax/blob/main/runs/2026-10-09/02_e2b_repack_vs_stock_on_jax_output.ipynb
- The pure-JAX engine: https://github.com/xbill9/tpu-jax
- The repack: https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct
- Google's 4-bit export: https://huggingface.co/google/gemma-4-E2B-it-qat-w4a16-ct
- Google's QAT checkpoint: https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-unquantized
- The source measurements: https://github.com/xbill9/gemma4-dev/blob/main/QUANTIZATION.md
