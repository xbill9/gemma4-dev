---
title: "Gemma 4 QAT on One TPU v5e: What Runs and What Doesn't"
published: false
description: "Google's quantization-aware-trained Gemma 4 weights, repacked into int4 and int8 formats vLLM serves on TPU, on one v5e chip. What runs: every size from E2B to 26B, level with bf16 through 12B, up to 2.4 points above Google's own 4-bit exports at the same speed, and 12B at 675 output tokens per second. What doesn't: bf16 above E2B, 31B in any build, and 26B past a 2,176-token context."
tags: gemma, googlecloud, machinelearning, llm
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/jev-tpu-v5e1/devto-v5e1-qat-cover.955039d5.jpg
---

This article provides a step by step guide to repacking Google's quantization-aware-trained (QAT) Gemma 4 weights for vLLM and serving them on one Google Cloud TPU v5e chip, with every build scored for classification, math, tool calling, throughput and long prompts. Every per-record output, log and script is committed.

On one v5e chip the repacked QAT builds serve every Gemma 4 size from E2B to 26B. Through 12B they read level with bf16 on a 3,880-record classification suite, they score up to 2.4 points above Google's own 4-bit exports at the same speed, and they make 12B the largest model on the chip: 11.31 GiB of weights, 675 output tokens per second at 16 requests, 0.964 on GSM8K and 0.955 on BFCL tool calling.

https://github.com/xbill9/gemma4-dev/tree/main/jev-tpu-v5e1

---

#### Why Repack?

One TPU v5e chip (`v5litepod-1`) has 15.75 GiB of HBM. Gemma 4 E4B at bf16 is 14.9 GiB and 12B is 22.4 GiB, so everything above E2B needs 4-bit or 8-bit weights on this chip.

Google trained 4-bit versions of every Gemma 4 size and publishes the trained values as "unquantized" bf16 checkpoints (`-qat-q4_0-unquantized`). Every group of 32 weights in them already sits on a 16-level grid. A repack stores those values in a format vLLM serves, without re-rounding them:

| Build | What is stored |
|---|---|
| `q4w4a16` | int4 weights holding the QAT grid exactly, 16-bit activations |
| `q4w4a16emb4` | the same, with the vocabulary tables (`embed_tokens`, `lm_head`, per-layer embeddings) also int4 |
| `w8a8` | int8 weights per channel from the QAT values, int8 activations per token |
| `w8a8emb4` | the same, with int4 vocabulary tables |

v5e multiplies int8 by int8 natively, which makes the int8 builds the fast ones on this chip.

---

#### At This Point You Should Have…

- A Google Cloud project with TPU v5e flex-start quota in `us-west4-a`, and the `gcloud` CLI logged in
- A Cloud Storage bucket for checkpoints and results
- A Hugging Face token in Secret Manager as `hf-token`
- A clone: `git clone https://github.com/xbill9/gemma4-dev`

---

#### Step 1 — Repack the QAT Weights

The 4-bit repack recovers each group's trained step and stores the group as int4, so every value keeps its trained place on the grid:

```bash
huggingface-cli download google/gemma-4-12B-it-qat-q4_0-unquantized \
  --local-dir ~/models/gemma-4-12B-it-qat-q4_0-unquantized
python3 ../jev-tpu-31b/repack_q4_0.py repack ~/models/gemma-4-12B-it-qat-q4_0-unquantized \
  ~/models/gemma-4-12B-it-qat-q4_0-w4a16-ct
```

The int8 builds take the same QAT values to int8 per channel with `../jev-tpu-31b/w8a8_from_qat.py`. The published builds are on Hugging Face under `xbill9/gemma-4-*-it-qat-*`.

Serving them on the TPU backend uses three additions to vLLM's `tpu_inference`, in `jev-tpu-v5e1/patches/`: an int8 W8A8 method on the JAX path, int4 embedding tables that stay packed on the chip and unpack only the rows a step reads, and an int4 `lm_head`.

---

#### Step 2 — Serve on One v5e Chip

Each run is a flex-start queued resource that boots, applies the patches to the pinned vLLM image, serves a list of builds one after another and uploads every result:

```bash
gcloud alpha compute tpus queued-resources create jev-tpu-v5e1-$RUN --zone us-west4-a \
  --accelerator-type v5litepod-1 --runtime-version v2-alpha-tpuv5-lite \
  --node-id jev-tpu-v5e1-$RUN-node \
  --provisioning-model flex-start --max-run-duration 4h --valid-until-duration 2h \
  --metadata jev-code=<bundle>.tgz,jev-run=$RUN,jev-qr=jev-tpu-v5e1-$RUN \
  --metadata-from-file startup-script=tpu/startup_quant.sh,jev-arms=arms.txt,jev-patches=patches.txt
```

One line per build names the checkpoint, what to measure and its serving flags:

```text
/work/models/gemma-4-12B-it-qat-w8a8-int8-emb4=12b-w8a8-emb4=read+load+gen=tools,gmu=0.92
```

```text
READY /work/models/gemma-4-12B-it-qat-w8a8-int8-emb4 after 405s
```

---

#### What Fits on One v5e Chip?

The suite is 3,880 public records from Bespoke Labs' benchmark set, paired record for record against bf16 (points, 95% range). Throughput is output tokens per second at 1, 4 and 16 parallel requests:

| Build | Suite vs bf16 | Output tok/s |
|---|---|---:|
| E2B bf16 | 0.683 | 144 / 560 / 2,008 |
| E2B `w8a8` | 0.686 (+0.3, −0.7 to +1.3) | 220 / 841 / 2,872 |
| E2B `w8a8emb4` | 0.677 (−0.6, −1.5 to +0.4) | 243 / 923 / 3,086 |
| E4B `q4w4a16` | 0.729 (−0.2, −0.8 to +0.5) | 75 / 291 / 1,012 |
| E4B `w8a8emb4` | 0.728 (−0.3, −1.0 to +0.5) | 133 / 509 / 1,747 |
| 12B `q4w4a16emb4` | 0.762 (+0.2, −0.4 to +0.8) | 35 / 127 / 407 |
| 12B `w8a8emb4` | 0.761 (+0.1, −0.6 to +0.8) | 57 / 219 / 675 |
| 26B `q4w4a16` | 0.754 (−1.1, −1.8 to −0.3) | 31 / 102 / 201 |

Every build through 12B is level with bf16; the 26B repack reads 1.1 points below it. E4B, 12B and 26B at bf16 do not fit this chip, so their bf16 references ran on v6e.

12B `w8a8emb4` holds 11.31 GiB of weights and gives the KV cache the rest:

```text
TPU KV cache size: 9,728 tokens, Maximum concurrency for 4,096 tokens per request: 2.38x
```

The 26B mixture-of-experts repack serves at 13.58 GiB with a 2,176-token cache.

---

#### Repacks Against Google's 4-Bit Exports

Google's `-qat-w4a16-ct` exports re-round every group of the QAT weights. The repack keeps them, and reads the suite higher:

| Size | Google export | Repack | Difference |
|---|---:|---:|---|
| E2B | 0.655 | 0.678 | +2.4 (+1.4 to +3.4) |
| E4B | 0.716 | 0.729 | +1.3 (+0.7 to +2.0) |
| 12B | 0.752 | 0.758 | +0.6 (0.0 to +1.2) |

Served one after the other on the same VM, the two run at the same speed:

```text
e2b-google load at concurrency 16: 1910.8 tok/s, range 1884.7 to 1910.8
e2b-repack load at concurrency 16: 1910.2 tok/s, range 1878.3 to 1910.9
```

At 16 requests: E4B 1,020 against 1,009 and 12B 392 against 388.

---

#### Step 3 — Score Math and Tool Calls

`gen_eval.py` scores two tasks against the served model, in code:

```bash
JEV_GEN_MAX_TOKENS=2048 python3 gen_eval.py run http://localhost:8000 "$MODEL" gsm8k out/gsm8k.jsonl
python3 gen_eval.py run http://localhost:8000 "$MODEL" bfcl_simple out/bfcl_simple.jsonl
```

```text
12b-w8a8-emb4 gsm8k: {"task": "gsm8k", "max_tokens": 2048, "n": 1319, "right": 1272, "accuracy": 0.9644, "errors": 0, "truncated": 4, "wall_s": 584.8}
```

- **GSM8K**: all 1,319 test problems, zero-shot chain of thought, greedy, right when the final number matches.
- **BFCL v3 simple**: 400 records, one tool offered, served with `--tool-call-parser gemma4`, right when the call has the right name and accepted arguments.

---

#### How Do They Do on Math and Tool Calls?

| Build | GSM8K | BFCL |
|---|---:|---:|
| 12B `w8a8emb4` | 0.964 | 0.955 |
| 12B `q4w4a16emb4` | 0.958 | 0.948 |
| E4B `w8a8emb4` | 0.940 | 0.912 |
| E4B `q4w4a16` | 0.934 | 0.910 |
| E2B `q4w4a16` | 0.901 | 0.920 |
| E2B `w8a8emb4` | 0.895 | 0.920 |
| E2B `w8a8` | 0.889 | 0.915 |
| E2B bf16 | 0.910 | 0.928 |

Tool calling holds at every size: every E2B repack is within 1.3 points of bf16. On GSM8K the E2B repacks trail bf16 by 0.9 to 2.0 points, and the QAT weights themselves, stored at bf16, trail by 1.3 (−2.7 to 0.0), so the repack adds at most 0.8 to what QAT training left. At E2B the 4-bit repack holds math best. E4B and 12B have no bf16 reference for these tasks on this chip.

---

#### Int8 From the QAT Weights, or From bf16?

A third-party int8 build rounded directly from the bf16 release uses the same format and runs at the same speed. Built from the QAT weights instead, int8 reads the suite 1.5 points higher at E2B (+0.5 to +2.6). At 12B it scores 7.2 points higher on BFCL (0.955 against 0.882): the build rounded from bf16 writes string arguments wrapped in extra quote marks in 16 of the 400 records, and the QAT build writes none.

---

#### Step 4 — Long Prompts

`w4a16_client.py loadlong` streams requests with prompts of about 3,600 tokens and a unique prefix on each, so none is served from cache:

```text
12b-w8a8-emb4 long 3584 at concurrency 16: 3611 prompt tokens, 94.7 out tok/s, 1430.0 total tok/s, ttft 21.96 s
```

| Build | Output tok/s at 16 requests | Median first token |
|---|---:|---:|
| E2B bf16 | 906 | 1.10 s |
| E2B `w8a8emb4` | 1,249 | 0.82 s |
| E4B `w8a8emb4` | 649 | 1.43 s |
| 12B `w8a8emb4` | 95 | 22.0 s |

E2B `w8a8emb4` stays 1.38x bf16 with long prompts and answers sooner. At 12B the cache holds about two such requests at once.

---

#### 🔎 Tip: An fp8 KV Cache Doubles 12B's Room

`--kv-cache-dtype fp8` stores the cache at one byte per value:

```text
TPU KV cache size: 18,944 tokens, Maximum concurrency for 4,096 tokens per request: 4.62x
```

| 12B `w8a8emb4`, 16 requests | bf16 cache | fp8 cache |
|---|---:|---:|
| ~1,000-token prompts | 244 tok/s | 330 tok/s |
| ~3,600-token prompts | 95 tok/s | 157 tok/s |
| First token, ~3,600-token prompts | 22.0 s | 13.3 s |
| Suite / GSM8K / BFCL | 0.761 / 0.957 / 0.953 | 0.754 / 0.956 / 0.945 |

A single request runs at the same speed with either cache. The fp8 cache costs 0.7 suite points and holds GSM8K and BFCL.

---

#### What Doesn't Run

- **E4B and 12B at bf16.** At 14.9 and 22.4 GiB they leave no room on a 15.75 GiB chip; their 4-bit and 8-bit repacks run.
- **31B in any build.** Its 4-bit repack takes 19.04 GiB of HBM on its own.
- **26B past a 2,176-token context.** With int4 vocabulary tables it still takes 13.72 GiB of weights, and a 3,072-token cache failed to compile (`CompileTimeHbmOom`) and a 2,560-token cache failed to load.
- **fp8 weights, at speed.** 12B fp8 runs and scores level with int8, at 518 output tokens per second against 624 for int8 with the same bf16 tables: v5e has no fp8 compute and converts the weights before each multiply.

---

#### Compare and Contrast

| Size | Build | Suite | GSM8K | BFCL | tok/s at 16 |
|---|---|---:|---:|---:|---:|
| 12B | 🥇 `w8a8emb4` | 0.761 | 0.964 | 0.955 | 675 |
| 12B | `q4w4a16emb4` | 0.762 | 0.958 | 0.948 | 407 |
| E4B | 🥇 `w8a8emb4` | 0.728 | 0.940 | 0.912 | 1,747 |
| E4B | `q4w4a16` | 0.729 | 0.934 | 0.910 | 1,012 |
| E2B | 🥇 `w8a8` | 0.686 | 0.889 | 0.915 | 2,872 |
| E2B | `q4w4a16` | 0.678 | 0.901 | 0.920 | 1,906 |

---

#### So, Which One?

For the most capable model on one v5e chip, 12B `w8a8emb4`, with `--kv-cache-dtype fp8` when many long prompts arrive at once. It has its own serving directory, `tpu-vllm-v5e1-12b-w8a8emb4`, with an MCP server, and serves there at 57 / 216 / 713 output tokens per second. For E4B, `w8a8emb4`. For E2B, `w8a8` for speed, or the `q4w4a16` repack for math closest to bf16.

---

#### Teardown

Each run deletes its own queued resource when its last build finishes:

```bash
gcloud alpha compute tpus queued-resources list --zone us-west4-a
```

---

#### Summary

The goal of this article was to find what Google's QAT Gemma 4 weights can do on one TPU v5e chip. The key to the solution was repacking the QAT values into int4 and int8 formats without re-rounding them, and three additions to vLLM's TPU backend. The results were:

- 🟢 The repacks serve every Gemma 4 size from E2B to 26B on one v5e chip, level with bf16 on the suite through 12B
- 🟢 12B `w8a8emb4` fits at 11.31 GiB and serves 675 output tokens per second, with 0.964 on GSM8K and 0.955 on BFCL
- 🟢 The repacks read the suite up to 2.4 points above Google's 4-bit exports, at the same speed
- 🟢 int8 from the QAT weights beats int8 rounded from bf16: +1.5 suite points at E2B, +7.2 BFCL points at 12B
- 🟢 An fp8 KV cache doubles 12B's cache to 18,944 tokens and its long-prompt throughput by 1.65x
- ⚠️ On GSM8K the E2B builds trail bf16 by 0.9 to 2.0 points, most of it from QAT training itself
- ⚠️ 26B serves with a 2,176-token context and reads 1.1 points below bf16
- ❌ E4B and 12B at bf16 and 31B in any build do not fit one v5e chip

Scope: every build ran on one TPU v5e chip (`v5litepod-1`, flex-start) in us-west4-a, vLLM `0.29.1rc1.dev468+g0b7f11a1e` at `vllm/vllm-tpu@sha256:19a1a052…` with the three patches in `jev-tpu-v5e1/patches/`, one run per build. The E4B, 12B and 26B bf16 suite references ran on v6e, and E4B and 12B have no bf16 reference for GSM8K or BFCL. Throughput is the median of three passes of 256 output tokens. Pairings are record for record, and a difference counts only when its 95% range excludes zero. The repacked checkpoints are unofficial and derived from Google's release under Apache 2.0. Parts of the analysis and writing were done with AI assistance (Claude); every figure comes from the committed output files.

The strategy for serving Google's QAT Gemma 4 on one TPU v5e chip was validated with an incremental step by step approach.

---

#### References

- Code, patches, per-record results and logs: https://github.com/xbill9/gemma4-dev/tree/main/jev-tpu-v5e1
- The 12B serving directory: https://github.com/xbill9/gemma4-dev/tree/main/tpu-vllm-v5e1-12b-w8a8emb4
- 12B int8 with int4 tables: https://huggingface.co/xbill9/gemma-4-12B-it-qat-w8a8-int8-emb4
- 12B 4-bit repack: https://huggingface.co/xbill9/gemma-4-12B-it-qat-q4_0-w4a16-ct
- Google's QAT source checkpoints: https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized
- Google's QAT announcement: https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/
- GSM8K: https://github.com/openai/grade-school-math
- Berkeley Function Calling Leaderboard: https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard
- Bespoke Labs public suite: https://github.com/bespokelabsai/nimble/blob/0e67403/docs/PUBLIC_BENCHMARKS.md
- vLLM TPU documentation: https://docs.vllm.ai/projects/tpu/en/latest/
- Cloud TPU v5e: https://cloud.google.com/tpu/docs/v5e
