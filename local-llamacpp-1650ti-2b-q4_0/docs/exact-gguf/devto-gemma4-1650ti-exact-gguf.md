---
title: "Re-Packing Gemma 4's QAT GGUF: 12% Faster on a 4 GB Laptop GPU and 32x Closer to bf16"
published: false
description: "Gemma 4 E2B's quantization-aware-trained weights already sit on a 4-bit grid. Re-packing them into a GGUF on that exact grid made the file 22% smaller, decode 1.12x faster on a GTX 1650 Ti, and the output 32x closer to bf16."
tags: gemma, llamacpp, cuda, machinelearning
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/local-llamacpp-1650ti-2b-q4_0/docs/exact-gguf/devto-cover.8c299d36.jpg
---

This article provides a step by step guide to re-packing Google's quantization-aware-trained (QAT) Gemma 4 E2B GGUF so every weight lands on the grid it was trained onto, and deploying the result to a local, laptop hosted GPU enabled system — a Lenovo Yoga 9 with a 4 GB GTX 1650 Ti. A suite of Python MCP tools manages the llama.cpp hosted deployment.

https://github.com/xbill9/gemma4-dev/tree/main/local-llamacpp-1650ti-2b-q4_0

The re-packed file is on Hugging Face as `xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf`, with the script that builds it.

**Same model, same llama.cpp binary, same card, one different file: 22% smaller on disk, 134 MiB less GPU memory, decode 1.12x faster, and a mean KL divergence from bf16 of 0.0017 against 0.054 for Google's GGUF.**

---

#### What is this project trying to Do?

A previous article ran Google's `gemma-4-E2B-it-qat-q4_0-gguf` on this laptop. QAT is what made it fit: the model was trained knowing it would be stored at 4 bits, so the 4-bit copy keeps close to full quality.

This article asks whether the GGUF keeps what QAT trained. Google also publishes the QAT weights unpacked, as bf16 safetensors. Comparing the two shows that the GGUF moves a share of the trained values off their grid, and stores the two largest tensors in a different format altogether. Re-packing the same weights onto their own grid fixes both, and the 4 GB card is where the difference shows most.

---

#### At This Point You Should Have…

- llama.cpp built for `sm_75` with CUDA, as in the previous article (`make build` in the rig)
- Disk space for the unquantized source checkpoint (9.6G on disk) plus the 2.6 GB output
- `numpy`, installed into the system `python3`
- `google/gemma-4-E2B-it-qat-q4_0-unquantized` and `google/gemma-4-E2B-it-qat-q4_0-gguf` downloaded from Hugging Face

---

#### What Does QAT Store?

Inside the unquantized QAT checkpoint, every block of 32 values along a row holds at most 16 distinct values: an integer level from -8 to 7 times one step size per block. That is Q4_0's own layout — 4-bit levels plus one fp16 scale per 32 values — so a QAT weight matrix maps onto Q4_0 with nothing lost except rounding the step to fp16.

Both embedding tables carry the same grid. `embed_tokens` and `embed_tokens_per_layer` are QAT 4-bit data like the transformer layers, and so is the 27.5 MB `per_layer_model_projection`.

---

#### What Google's GGUF Does to It

Three things, measured against the unquantized QAT source:

- **The transformer layers are Q4_0 with llama.cpp's default step, `amax/8`.** In any block whose largest value sits below level 8, that step differs from the trained one, and values shift by up to a full step. In `blk.0.attn_q` only 51% of values equal the source.
- **Both embedding tables are Q6_K**, off the trained grid by up to 28% of a step. Together they are 2.257 GB of the 3.334 GB of tensors.
- **`per_layer_model_proj` is F16.**

```text
$ python3 inspect_gguf.py
gemma-4-E2B_q4_0-it.gguf
  by tensor type (the slot-5 token is q4_0; the file mostly is not):
    Q6_K      2.257 GB  (67.7%)
    Q4_0      1.048 GB  (31.4%)
    F16       0.028 GB  ( 0.8%)
    F32       0.001 GB  ( 0.0%)
```

A file named `q4_0` is 31.4% Q4_0.

---

#### Step 1 — Re-Pack on the Trained Step

`gguf_exact.py` is one numpy file. It copies Google's GGUF header, tokenizer, chat template and hyperparameters byte for byte, keeps every tensor it does not rebuild, and rewrites the 275 layer matrices, both embedding tables and `per_layer_model_proj` as Q4_0.

For each block of 32 it searches for the step the block was trained on: `amax/m` for the first `m` from 1 to 8 that puts every value on an integer level. It refines that step by least squares and rounds it to fp16. A block with no such step stops the build, so every block keeps its trained grid or the build fails.

```shell
python3 gguf_exact.py \
  ~/models/gemma-4-E2B-it-qat-q4_0-unquantized \
  ~/models/gemma-4-E2B-it-qat-q4_0-gguf/gemma-4-E2B_q4_0-it.gguf \
  gemma-4-E2B-it-q4_0-exact.gguf
```

```text
per_layer_model_proj.weight: 96.741151% exact
per_layer_token_embd.weight: 96.914884% exact
token_embd.weight: 96.760972% exact
...
blk.34.ffn_up.weight: 96.809647% exact
blk.34.inp_gate.weight: 96.996816% exact
blk.34.proj.weight: 96.883138% exact
```

It ran in 305 s on the laptop. No block was off the grid. Across 4.63 billion rebuilt values, 96.84% decode back to the exact source value after bf16 rounding; the rest are within a few percent of a step, from rounding the scale to fp16.

---

#### Step 2 — Check the Build Is Reproducible

```shell
sha256sum gemma-4-E2B-it-q4_0-exact.gguf
```

```text
419db9a6bf3bc15d770c85ccf9216827aa88c64fe8f3d72fbe5674f48efe2dc8  gemma-4-E2B-it-q4_0-exact.gguf
```

The same script on a second laptop, an i7-1360P, wrote the same hash. The build uses only numpy and the two downloads, so anyone can rebuild the published file and compare hashes.

---

#### What Is Left in the File

```text
$ python3 inspect_gguf.py .../gemma-4-E2B-it-q4_0-exact.gguf
gemma-4-E2B-it-q4_0-exact.gguf
  tensors:  541
  total:    2.605 GB

  largest tensors:
      1321.2 MB  per_layer_token_embd.weight  Q4_0    [8960, 262144]  <- LAZY, host-resident
       226.5 MB  token_embd.weight            Q4_0    [1536, 262144]

  lazy (never on GPU):        1.321 GB  (51% of file)
  must be resident:           1.283 GB = 1.20 GiB

  by tensor type (the slot-5 token is q4_0; the file mostly is not):
    Q4_0      2.603 GB  (100.0%)
    F32       0.001 GB  ( 0.0%)
```

Every weight matrix is Q4_0; the remaining 1.1 MB is F32 norms and scales. The file went from 3,349,516,256 bytes to 2,620,370,912, 22% smaller. Q4_0 is the format the model was trained onto, so there is no smaller llama.cpp type left to take without moving weights off the grid.

---

#### Step 3 — Measure Quality Against bf16

The reference is a bf16 GGUF written by the same script with `--bf16`: Google's metadata with the source bytes unchanged, so the files differ only in how the weights are stored. `llama-perplexity` records bf16's output distribution over wikitext-2 (16 chunks of 512 tokens) on the CPU, because bf16 does not fit in 4 GiB, then scores each quantized file on the card.

The two runs, in llama.cpp's standard KL-divergence form:

```shell
llama-perplexity -m ref-bf16.gguf -f wiki.test.raw -c 512 --chunks 16 \
  --kl-divergence-base base.kld -ngl 0
llama-perplexity -m gemma-4-E2B-it-q4_0-exact.gguf -c 512 --chunks 16 \
  --kl-divergence-base base.kld --kl-divergence -ngl 99
```

```text
Mean PPL(Q)/PPL(base)         :   1.021563 ±   0.005079
Mean    KLD:   0.001680 ±   0.000072
99.0%   KLD:   0.016059
Same top p: 98.260 ± 0.205 %
```

| on the GTX 1650 Ti | Google GGUF | Re-packed |
| :--- | ---: | ---: |
| Mean KL divergence from bf16 | 0.054264 | **0.001680** 🥇 |
| 99th percentile KL divergence | 0.406903 | **0.016059** 🥇 |
| Same top token as bf16 | 87.18% | **98.26%** 🥇 |
| Perplexity / bf16 perplexity | 1.092 | **1.022** 🥇 |

Google's file picks a different most-likely next token from bf16 about one time in eight. The re-packed file does so under one time in fifty.

---

#### Step 4 — Measure Speed

`llama-bench`, full offload, flash attention on, 512-token prefill and 128-token generation, five repeats per cell. Each pass runs all three files, in the order Google, v1, v2 or its reverse (ABBA across four passes). v1 is the same re-pack with `per_layer_model_proj` left at F16.

```shell
llama-bench -m google.gguf -m exact-v1.gguf -m exact-v2.gguf \
  -ngl 99 -fa 1 -t 6 -p 512 -n 128 -r 5
```

```text
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |   1 |           pp512 |        345.17 ± 0.66 |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | CUDA       |  99 |   1 |           tg128 |         81.82 ± 0.06 |
| gemma4 E2B Q4_0                |   2.44 GiB |     4.63 B | CUDA       |  99 |   1 |           pp512 |        338.73 ± 0.56 |
| gemma4 E2B Q4_0                |   2.44 GiB |     4.63 B | CUDA       |  99 |   1 |           tg128 |         80.26 ± 0.04 |
| gemma4 E2B Q4_0                |   3.10 GiB |     4.63 B | CUDA       |  99 |   1 |           pp512 |        337.62 ± 0.50 |
| gemma4 E2B Q4_0                |   3.10 GiB |     4.63 B | CUDA       |  99 |   1 |           tg128 |         73.10 ± 0.04 |
```

| tok/s, mean of 4 passes | Google | v1 | v2 |
| :--- | ---: | ---: | ---: |
| Generation (tg128) | 73.26 | 80.31 🥈 | **81.75** 🥇 |
| Prefill (pp512) | 338.26 | 339.06 🥈 | **344.64** 🥇 |
| v2 / Google, generation | | | **1.12x** |

The generation gain held at 1.111–1.121x in every pass and both orders. `token_embd` doubles as the output projection, so all of it is read for every generated token, and at Q4_0 it is 31% smaller than at Q6_K. Decode at one request on this card is limited by memory bandwidth, which makes fewer bytes read the likely cause; memory traffic was not measured. v2's extra 1.8% over v1 comes from one 27.5 MB tensor moving from F16, which this card multiplies without tensor cores, to Q4_0.

---

#### 🔎 Tip: Cool the Card Between Passes

The GTX 1650 Ti Max-Q is capped at 40 W and clocks down as it heats. The script behind these numbers waits until the GPU reads 50 °C or less before each pass, and runs the files in both orders. The four passes above agree to 1%, so the ordering and cooldown are why a 2% difference between v1 and v2 can be read at all.

---

#### Step 5 — Serve It

`tpu.env` is the rig's configuration, and it now names the re-packed file and pins its hash:

```shell
MODEL_NAME=xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf
MODEL_PATH=/home/xbill/models/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf
MODEL_SHA256=419db9a6bf3bc15d770c85ccf9216827aa88c64fe8f3d72fbe5674f48efe2dc8
N_GPU_LAYERS=99
FLASH_ATTENTION=1
REASONING=off
```

```shell
make serve
```

llama-server's allocation log, same flags, each file:

```text
# Google GGUF
load_tensors:   CPU_Mapped model buffer size =  2152.50 MiB
load_tensors:        CUDA0 model buffer size =  1341.78 MiB
# re-packed
load_tensors:   CPU_Mapped model buffer size =  1476.00 MiB
load_tensors:        CUDA0 model buffer size =  1223.91 MiB
llama_kv_cache:      CUDA0 KV buffer size =    48.00 MiB
llama_kv_cache:      CUDA0 KV buffer size =    12.00 MiB
sched_reserve:      CUDA0 compute buffer size =   122.52 MiB
```

`nvidia-smi` reports the server process at **1480 MiB** of 4096, against 1614 MiB with Google's file. The KV cache and compute buffers are the same size for both; the whole saving is in weights.

The MCP server's `model_server_status` reads the running process and confirms it is the GPU build serving before any query goes through.

---

#### Does the Whole Model Fit on the Card Now?

It does. llama.cpp normally leaves `per_layer_token_embd`, the largest tensor, in memory-mapped host RAM and reads one row per token. At Q6_K it was 1.93 GB and could never have fit; at Q4_0 it is 1.32 GB. Forcing it onto the card with `-ot 'per_layer_token_embd\.weight=CUDA0'` works:

| | table in host RAM (default) | table on the card |
| :--- | ---: | ---: |
| `nvidia-smi` process | **1480 MiB** 🥇 | 2732 MiB |
| Generation (tg128) | 82.17 / 81.96 | 82.16 / 82.05 |
| Prefill (pp512) | 347.96 / 345.00 | 345.68 / 345.84 |

Speed is unchanged within 0.7%, at a cost of 1.25 GiB of VRAM. Each token reads one 8960-value row, about 5 KB, while decode streams about 1.2 GB of weights per token, so where the table lives never mattered. Leave it in host RAM.

---

#### What It Did to the GPU vs CPU Comparison

This rig has a CPU-only twin that serves the same file with the same llama.cpp commit and flags, differing only in `-ngl`. Both moved to the re-packed file together, and the paired sweep ran again in CPU, GPU, GPU, CPU order.

| GPU / CPU, median over 8 cells | Google GGUF (2026-09-22) | Re-packed (2026-09-29) |
| :--- | ---: | ---: |
| Decode | 4.14x | **4.37x** |
| Prefill | 3.42x | **3.80x** |
| End to end | 3.62x | **3.99x** |

The GPU arm's decode rose 1.10x, as `llama-bench` predicted. A separate `llama-bench` run on the CPU build puts the file's effect there at 1.06x for decode and none for prefill. The rest of the wider lead is heat: both CPU passes ended at 90 °C with 50,918 and 59,190 throttle events. Quote these ratios with that beside them.

---

#### What Five Greedy Prompts Can Tell You

Five fixed prompts at temperature 0, thinking off, 200 tokens each, compared character by character with bf16:

- **On the CPU**, the v1 re-pack matched bf16 on all five. Google's file drifted on the longest answer, at character 192.
- **On the card**, v2 matched on four; the long French answer first differs at character 202, the same place as Google's file on the card.

The CUDA backend quantizes activations for its Q4_0 matrix multiplies, so some drift on the card comes from the kernels and belongs to any Q4_0 file. Five prompts are too few to rank files. KL divergence over 8,192 tokens is the measure to rank them by.

---

#### Compare and Contrast

| on the GTX 1650 Ti | Google GGUF | Re-packed v2 |
| :--- | ---: | ---: |
| File size | 3.12 GiB | **2.44 GiB** 🥇 |
| Share of weights stored as Q4_0 | 31.4% | **100%** 🥇 |
| Server VRAM | 1614 MiB | **1480 MiB** 🥇 |
| Generation | 73.26 tok/s | **81.75 tok/s** 🥇 |
| Mean KL divergence from bf16 | 0.054 | **0.0017** 🥇 |
| Same top token as bf16 | 87.2% | **98.3%** 🥇 |

---

#### So, Which One?

On this laptop, the re-packed file. It is smaller, faster, closer to bf16 and uses less of a card that has little to spare, and it loads in the same llama.cpp build with the same flags. Nothing about serving changes except the path in `tpu.env`.

Google's file remains the reference build, and its metadata is what the re-pack copies. The re-pack depends on Google's unquantized QAT checkpoint staying published, and it applies only where the source was trained onto a 4-bit grid: the same script refuses any tensor that has blocks off it.

---

#### Summary

The goal of this article was to find out whether Google's QAT GGUF of Gemma 4 E2B keeps the 4-bit values QAT trained, and what storing them exactly does on a 4 GB laptop GPU. The key to the solution was the unquantized QAT checkpoint: its weights already sit on Q4_0's grid, so they can be packed with the step each block was trained on. The measured results were:

- 🟢 **32x closer to bf16.** Mean KL divergence 0.0017 against 0.054 (0.054264 / 0.001680, arithmetic); same top token 98.3% of the time against 87.2%.
- 🟢 **1.12x faster generation**, 81.75 against 73.26 tok/s, holding in every pass and order; prefill 1.02x.
- 🟢 **22% smaller file and 134 MiB less VRAM**, with every weight matrix Q4_0.
- 🟢 **Bit-identical rebuilds** on two different machines, from a numpy-only script.
- ⚠️ **Moving the per-layer embedding table into VRAM fits and gains nothing.**
- ⚠️ **The GPU's lead over the CPU twin grew to 4.37x decode**, partly from the file and partly from a hot CPU session.
- ❌ **No smaller llama.cpp type is worth trying:** anything below Q4_0 moves weights off the grid the model was trained onto.

Scope: one Lenovo Yoga 9 15IMH5 (Core i7-10750H, GTX 1650 Ti Max-Q, 4096 MiB, 40 W cap), llama.cpp `f95b0d9` built for sm_75 with CUDA 13.4. Google's file is `google/gemma-4-E2B-it-qat-q4_0-gguf` at revision `675cff4`; the source is `google/gemma-4-E2B-it-qat-q4_0-unquantized` at `6befbac`. `llama-bench` figures are four ABBA passes of five repeats with a cooldown before each; KL divergence is 16 × 512 tokens of wikitext-2 against bf16 logits computed on the CPU. The GPU-vs-CPU sweep is single-stream and ran with thinking on. The reproducibility check used a second machine, an i7-1360P, on a different llama.cpp commit. Task accuracy beyond KL divergence and five greedy prompts was not measured.

The strategy for using MCP for a local GPU deployment of a re-packed QAT GGUF was validated with an incremental step by step approach.

---

#### References

* [local-llamacpp-1650ti-2b-q4_0 | GitHub](https://github.com/xbill9/gemma4-dev/tree/main/local-llamacpp-1650ti-2b-q4_0)
* [xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf | Hugging Face](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf)
* [google/gemma-4-E2B-it-qat-q4_0-gguf | Hugging Face](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf)
* [google/gemma-4-E2B-it-qat-q4_0-unquantized | Hugging Face](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-unquantized)
* [Gemma 4 on an Old 4 GB Laptop GPU: QAT Takes It From 9.5 GiB to 1.6 | dev.to](https://dev.to/gde/gemma-4-on-an-old-4-gb-laptop-gpu-qat-takes-it-from-95-gib-to-16-b5l)
* [Quantization-Aware Training for Gemma 4 | Google](https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/)
* [llama.cpp | GitHub](https://github.com/ggml-org/llama.cpp)
