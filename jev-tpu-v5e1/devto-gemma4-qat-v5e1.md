---
title: "Every QAT Gemma 4 Build on One TPU v5e: 12B at 675 Tokens per Second, and a Math Loss the Suite Missed"
published: false
description: "Every build of Google's quantization-aware-trained Gemma 4, from E2B to 26B, served with vLLM on one TPU v5e chip and scored on a 3,880-record suite, GSM8K and BFCL tool calling. 12B int8 with int4 vocabulary tables fits at 11.31 GiB and serves 675 output tokens per second. E2B builds from the QAT weights read level with bf16 on the suite and lose 1 to 2 points on GSM8K."
tags: gemma, googlecloud, machinelearning, llm
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/jev-tpu-v5e1/devto-v5e1-qat-cover.955039d5.jpg
---

This article provides a step by step guide to serving every quantized build of Google's quantization-aware-trained (QAT) Gemma 4 on one Google Cloud TPU v5e chip with vLLM, and scoring each one for classification, math, tool calling, throughput and long prompts. Every per-record output, log and script is committed.

The biggest model that serves on one v5e chip is 12B with int8 weights and activations and int4 vocabulary tables: 11.31 GiB of weights, 675 output tokens per second at 16 requests, level with bf16 on a 3,880-record classification suite and 0.964 on GSM8K. An fp8 KV cache doubles its cache to 18,944 tokens for long prompts. On E2B, every build made from the QAT weights reads level with bf16 on the suite and trails it by 1 to 2 points on GSM8K. A 12B int8 build rounded directly from bf16 loses 7.2 points on BFCL tool calling to the one built from the QAT weights.

https://github.com/xbill9/gemma4-dev/tree/main/jev-tpu-v5e1

---

#### Why Measure This?

One TPU v5e chip (`v5litepod-1`) has 15.75 GiB of HBM, of which vLLM can use about 14.49. Gemma 4 E4B at bf16 is 14.9 GiB and 12B is 22.4 GiB, so on this chip everything above E2B has to be quantized, and the question is which quantization.

Google trained 4-bit versions of every Gemma 4 size and publishes them three ways: GGUF for llama.cpp, "unquantized" bf16 checkpoints holding the QAT values, and compressed-tensors W4A16 for vLLM. From the unquantized checkpoints the rest of the builds here follow: a W4A16 repack that keeps the QAT grid exactly, int8 W8A8 (v5e multiplies int8 natively), fp8, and variants that also store the vocabulary tables at int4.

A classification suite is quick to score and was the first measure. This article adds math and tool calling, because a build can hold one and lose the other.

---

#### At This Point You Should Have…

- A Google Cloud project with TPU v5e flex-start quota in `us-west4-a`, the only zone that accepted flex-start `v5litepod-1` here, and the `gcloud` CLI logged in
- A Cloud Storage bucket for checkpoints, code bundles and results
- A Hugging Face token in Secret Manager as `hf-token`
- A clone: `git clone https://github.com/xbill9/gemma4-dev`

---

#### Step 1 — The Builds

Each build lives in its own directory, named for its chip and its exact checkpoint. The last part of the name is the encoding:

| Encoding | What is stored |
|---|---|
| `w4a16` | Google's `-qat-w4a16-ct` export: int4 weights, re-rounded per group |
| `q4w4a16` | the same format holding the QAT grid exactly, repacked from `-qat-q4_0-unquantized` |
| `w8a8` | int8 weights per channel from the QAT weights, int8 activations per token |
| `w8a8rtn` | the same format rounded from the bf16 release (third-party `glenic/*-W8A8-INT8`) |
| `…emb4` | the vocabulary tables (`embed_tokens`, an untied `lm_head`, per-layer embeddings) also int4 |

Three patches to vLLM's TPU backend (`tpu_inference`) are needed on v5e: one adds the int8 W8A8 method on the JAX path, one keeps int4 tables packed on the chip and unpacks only the rows a step gathers, and one serves an int4 `lm_head`. They are in `jev-tpu-v5e1/patches/` and are applied to the pinned image at boot.

---

#### Step 2 — Launch One v5e Chip

Each run is a flex-start queued resource that boots, serves a list of builds one after another, uploads every result and deletes itself:

```bash
gcloud alpha compute tpus queued-resources create jev-tpu-v5e1-$RUN --zone us-west4-a \
  --accelerator-type v5litepod-1 --runtime-version v2-alpha-tpuv5-lite \
  --node-id jev-tpu-v5e1-$RUN-node \
  --provisioning-model flex-start --max-run-duration 4h --valid-until-duration 2h \
  --metadata jev-code=<bundle>.tgz,jev-run=$RUN,jev-qr=jev-tpu-v5e1-$RUN,jev-swap-gb=16 \
  --metadata-from-file startup-script=tpu/startup_quant.sh,jev-arms=arms.txt,jev-patches=patches.txt
```

An arm is `<model>=<tag>=<modes>=<flags>`. Modes are `read` (the suite), `load` (throughput), `long` (long prompts) and `gen` (GSM8K and BFCL), joined with `+`. Flags set serving options per arm:

```text
/work/models/gemma-4-12B-it-qat-w8a8-int8-emb4=12b-w8a8-emb4=long+gen=tools,mml=4096,gmu=0.92
```

The patch list has to be passed explicitly: without `jev-patches` the runner applies only its two default patches, and the int8 checkpoints fail to load.

---

#### Step 3 — Serve and Read

The suite is the 3,880 public records of Bespoke Labs' benchmark set, read by label probability and paired record for record against bf16. E2B's bf16 reference runs on the same v5e chip; E4B and 12B at bf16 do not fit, so their references come from one v6e chip.

```text
READY google/gemma-4-E2B-it after 241s
```

Throughput is 16 or fewer parallel requests of exactly 256 output tokens, three passes, median reported.

---

#### What Fits on One v5e Chip?

Suite score against bf16 (points, 95% range) and output tokens per second at 1, 4 and 16 requests:

| Build | Suite | Output tok/s |
|---|---|---:|
| E2B bf16 | 0.683 | 144 / 560 / 2,008 |
| E2B `w8a8` | 0.686 (+0.3, −0.7 to +1.3) | 220 / 841 / 2,872 |
| E2B `w8a8emb4` | 0.677 (−0.6, −1.5 to +0.4) | 243 / 923 / 3,086 |
| E4B `q4w4a16` | 0.729 (−0.2, −0.8 to +0.5) | 75 / 291 / 1,012 |
| E4B `w8a8emb4` | 0.728 (−0.3, −1.0 to +0.5) | 133 / 509 / 1,747 |
| 12B `q4w4a16emb4` | 0.762 (+0.2, −0.4 to +0.8) | 35 / 127 / 407 |
| 12B `fp8` | 0.757 (−0.3, −0.9 to +0.4) | 43 / 165 / 518 |
| 12B `w8a8emb4` | 0.761 (+0.1, −0.6 to +0.8) | 57 / 219 / 675 |

12B `w8a8emb4` holds 11.31 GiB of weights. With `--gpu-memory-utilization 0.92` vLLM gives the KV cache the rest:

```text
TPU KV cache size: 9,728 tokens, Maximum concurrency for 4,096 tokens per request: 2.38x
```

The 26B mixture-of-experts repack also serves, at 13.58 GiB of weights, with its cache capped at 17 blocks (2,176 tokens) so the compiled model still fits. It reads the suite at 0.754 and serves 31 / 102 / 201 output tokens per second. Storing its vocabulary tables at int4 scores the same (0.755) and runs 228 tokens per second at 16 requests, but takes 13.72 GiB, so its context stays at 2,176 tokens: at 3,072 tokens it failed to compile (`CompileTimeHbmOom`) and at 2,560 it failed to load (`RuntimeProgramAllocationFailure`).

fp8 costs 17% of the int8 throughput at 12B for no gain in accuracy, because v5e has no fp8 compute and converts the weights before each multiply.

---

#### Are Google's Exports Slower Than the Repacks?

The repack keeps the QAT grid and reads the suite 2.4 points higher than Google's `-w4a16-ct` export at E2B, 1.3 at E4B and 0.6 at 12B. Served one after the other on the same VM, the two run at the same speed:

```text
e2b-google load at concurrency 16: 1910.8 tok/s, range 1884.7 to 1910.8
e2b-repack load at concurrency 16: 1910.2 tok/s, range 1878.3 to 1910.9
```

At 16 requests: E4B 1,020 against 1,009, 12B 392 against 388.

---

#### Step 4 — Score Math and Tool Calls

`gen_eval.py` runs two tasks against the served model and scores them in code:

```bash
JEV_GEN_MAX_TOKENS=2048 python3 gen_eval.py run http://localhost:8000 "$MODEL" gsm8k out/gsm8k.jsonl
python3 gen_eval.py run http://localhost:8000 "$MODEL" bfcl_simple out/bfcl_simple.jsonl
```

```text
e2b-bf16 gsm8k: {"task": "gsm8k", "max_tokens": 2048, "n": 1319, "right": 1200, "accuracy": 0.9098, "errors": 0, "truncated": 3, "wall_s": 283.1}
```

- **GSM8K**: all 1,319 test problems, zero-shot chain of thought, greedy, right when the final number matches.
- **BFCL v3 simple**: 400 records, one tool offered through the OpenAI `tools` field, served with `--enable-auto-tool-choice --tool-call-parser gemma4`, right when exactly one call has the right name and accepted arguments. The checker follows BFCL's rules and passes all 400 reference answers in `tests/test_gen_eval.py`.

`gen_compare.py` pairs two outputs record by record, with a 95% range from 10,000 bootstrap resamples.

---

#### Does Math Hold?

GSM8K at a 2,048-token answer limit and BFCL, against E2B bf16 on the same chip:

| E2B build | GSM8K | vs bf16 (95% range) | BFCL |
|---|---:|---|---:|
| bf16 | 0.910 | — | 0.928 |
| QAT bf16 (`-qat-q4_0-unquantized`) | 0.897 | −1.3 (−2.7 to 0.0) | 0.922 |
| `q4w4a16` | 0.901 | −0.9 (−2.2 to +0.4) | 0.920 |
| `w8a8` | 0.889 | **−2.0 (−3.4 to −0.8)** | 0.915 |
| `w8a8emb4` | 0.895 | **−1.4 (−2.8 to −0.1)** | 0.920 |

The `w8a8` build read +0.3 against bf16 on the suite and reads −2.0 on GSM8K. Most of the gap is already in the QAT weights before any quantization (QAT bf16, −1.3); quantizing them costs at most 0.8 more, and every such step crosses zero on its own. BFCL is level for every E2B build.

At E4B, `w8a8emb4` and the `q4w4a16` repack are level: 0.940 against 0.934 on GSM8K, 0.912 against 0.910 on BFCL. At 12B, `w8a8emb4` reads 0.964 on GSM8K and 0.955 on BFCL; the int4 and fp8 builds are within a point of it. E4B and 12B have no bf16 reference for these tasks on this chip.

---

#### Does It Matter Where the Int8 Values Come From?

`w8a8` takes its int8 values from the QAT weights; `w8a8rtn` rounds the bf16 release. Same format, same speed:

| Size | Suite, QAT against rounded | Other tasks |
|---|---|---|
| E2B | **+1.5** (+0.5 to +2.6) | — |
| E4B | −0.3 (−1.1 to +0.4) | — |
| 12B | +0.5 (−0.2 to +1.3) | against `w8a8emb4`: BFCL **+7.2**, GSM8K +0.8 |

The rounded 12B build scores 0.882 on BFCL against 0.955. In 16 of the 400 records it wraps string arguments in literal quotes, `"activity_level": "\"lightly active\""`, which the parser passes through; the QAT build never does. With the quotes removed it still trails by 3 points.

---

#### 🔎 Tip: Strip a Tied `lm_head` Before Serving

The rounded builds store `lm_head.weight` beside a tied `embed_tokens`, and vLLM's JAX path loads both: 1.88 GiB extra at 12B, enough to not fit. `strip_tied_head.py` copies a checkpoint to Cloud Storage without the stored head, after checking that the config ties the embeddings and that the two tensors hash the same:

```bash
python3 strip_tied_head.py glenic/gemma-4-12B-it-W8A8-INT8 gs://<bucket>/models/gemma-4-12B-it-W8A8-INT8-glenic-notiedhead
```

```json
 "removed": ["lm_head.weight"],
 "reason": "tied; byte-identical to embed_tokens",
 "sha256": {
  "model.language_model.embed_tokens.weight": "9619fd670a4074367c72f2cf09d777970ad1eafa511bff605a82c8558dcc24d4",
  "lm_head.weight": "9619fd670a4074367c72f2cf09d777970ad1eafa511bff605a82c8558dcc24d4"
 },
```

Every served value stays the source's.

---

#### 🔎 Tip: Give GSM8K 2,048 Tokens

At a 768-token limit, the QAT-derived E2B builds had 74 to 86 answers cut off against bf16's 53, and every E2B gap came out wider: QAT bf16 read −1.9 there against −1.3 at 2,048. At 2,048 tokens no build had more than 11 answers cut off. A limit that cuts off one build more than another measures answer length.

---

#### Step 5 — Long Prompts

`w4a16_client.py loadlong` streams requests with prompts of about 1,000 and 3,600 tokens and a unique prefix on every request of every pass, so the prefix cache never serves one:

```text
12b-w8a8-emb4 long 3584 at concurrency 16: 3611 prompt tokens, 94.7 out tok/s, 1430.0 total tok/s, ttft 21.96 s
```

Output tokens per second at 16 requests, and median time to first token, with prompts of about 3,600 tokens:

| Build | Output tok/s | First token |
|---|---:|---:|
| E2B bf16 | 906 | 1.10 s |
| E2B `w8a8emb4` | 1,249 | 0.82 s |
| E4B `w8a8emb4` | 649 | 1.43 s |
| 12B `q4w4a16emb4` | 69 | 23.2 s |
| 12B `w8a8emb4` | 95 | 22.0 s |

At 12B the 9,728-token cache holds about two such requests at a time, and the rest wait.

---

#### Step 6 — An fp8 KV Cache

Adding `--kv-cache-dtype fp8` stores the cache at one byte per value:

```text
TPU KV cache size: 18,944 tokens, Maximum concurrency for 4,096 tokens per request: 4.62x
```

```text
12b-w8a8-emb4-kvfp8 long 3584 at concurrency 16: 3611 prompt tokens, 156.6 out tok/s, 2365.3 total tok/s, ttft 13.27 s
```

| 12B `w8a8emb4` | bf16 cache | fp8 cache |
|---|---:|---:|
| KV tokens | 9,728 | 18,944 |
| 16 requests, ~1,000-token prompts | 244 tok/s | 330 tok/s |
| 16 requests, ~3,600-token prompts | 95 tok/s | 157 tok/s |
| Suite | 0.761 | 0.754 (−0.7, −1.3 to −0.1) |
| GSM8K / BFCL (768-token limit) | 0.957 / 0.953 | 0.956 / 0.945 |

With one request the two run at the same speed. The fp8 cache costs 0.7 suite points and leaves math and tool calling level.

---

#### Step 7 — Serve the Pick From Its Own Directory

`tpu-vllm-v5e1-12b-w8a8emb4` is a full serving directory with an MCP server, and it provisions and serves the same checkpoint through its own startup script. Measured on the VM, it matched the runs above to within 1.3%:

| Requests | Own directory | Runs above |
|---:|---:|---:|
| 1 | 57.0 | 57.2 |
| 4 | 215.7 | 217.1 |
| 16 | 713.4 | 722.5 |

It was ready 17 minutes after the queued resource was created.

---

#### Compare and Contrast

| Size | Build | Suite | GSM8K | tok/s at 16 |
|---|---|---:|---:|---:|
| 12B | 🥇 `w8a8emb4` | 0.761 | 0.964 | 675 |
| 12B | `q4w4a16emb4` | 0.762 | 0.958 | 407 |
| E4B | 🥇 `w8a8emb4` | 0.728 | 0.940 | 1,747 |
| E4B | `q4w4a16` | 0.729 | 0.934 | 1,012 |
| E2B | 🥇 `w8a8` | 0.686 | 0.889 | 2,872 |
| E2B | `q4w4a16` | 0.678 | 0.901 | 1,906 |
| E2B | bf16 | 0.683 | 0.910 | 2,008 |

---

#### So, Which One?

For the most capable model on one v5e chip, 12B `w8a8emb4`, with `--kv-cache-dtype fp8` when many long prompts arrive at once. For E4B, `w8a8emb4`. For E2B, `w8a8` for speed at 1.4x to 1.5x bf16, or the `q4w4a16` repack for math closest to bf16. Take int8 builds from the QAT weights: the rounded ones lose tool-calling accuracy at 12B. Leave fp8 weights and the 26B for other chips.

---

#### Teardown

Each run deletes its own queued resource when its last build finishes. To check nothing is left:

```bash
gcloud alpha compute tpus queued-resources list --zone us-west4-a
```

---

#### Summary

The goal of this article was to find the best quantized Gemma 4 for one TPU v5e chip. The key to the solution was building int8 and int4 variants from Google's QAT weights, three vLLM TPU patches, and scoring every build on classification, math, tool calling, throughput and long prompts. The results were:

- 🟢 12B `w8a8emb4` serves on one v5e chip at 11.31 GiB, 675 output tokens per second at 16 requests, level with bf16 on the suite, 0.964 on GSM8K
- 🟢 An fp8 KV cache doubles its cache to 18,944 tokens: 157 against 95 tokens per second at 16 long prompts
- ⚠️ The fp8 cache costs 0.7 suite points; GSM8K and BFCL hold
- ⚠️ E2B builds from the QAT weights trail bf16 by 1 to 2 points on GSM8K while level on the suite; most of it is in the QAT weights themselves
- ❌ The 12B int8 build rounded from bf16 loses 7.2 points on BFCL, with quote-wrapped string arguments
- 🟢 Google's W4A16 exports and the QAT-grid repacks run at the same speed; the repacks read the suite 0.6 to 2.4 points higher
- ❌ fp8 weights run 17% slower than int8 at 12B for the same accuracy
- ⚠️ The 26B serves with a 2,176-token context, and int4 tables do not extend it

Scope: every build ran on one TPU v5e chip (`v5litepod-1`, flex-start) in us-west4-a, vLLM `0.29.1rc1.dev468+g0b7f11a1e` at `vllm/vllm-tpu@sha256:19a1a052…` with the three patches in `jev-tpu-v5e1/patches/`, one run per build, `--max-num-batched-tokens 512`; the E4B and 12B bf16 suite references ran on one v6e chip, and no bf16 E4B or 12B reference exists for GSM8K or BFCL. Throughput is the median of three passes with 256 output tokens and `ignore_eos`; long prompts used a unique prefix per request. Each build ran once; the pairings are record for record, and a difference counts only when its 95% range excludes zero. The rounded builds are third-party checkpoints served with a byte-identical duplicate head removed. Parts of the analysis and writing were done with AI assistance (Claude); every figure comes from the committed output files.

The strategy for choosing a quantized Gemma 4 for one TPU v5e chip was validated with an incremental step by step approach.

---

#### References

- Code, patches, per-record results and logs: https://github.com/xbill9/gemma4-dev/tree/main/jev-tpu-v5e1
- The 12B serving directory: https://github.com/xbill9/gemma4-dev/tree/main/tpu-vllm-v5e1-12b-w8a8emb4
- 12B int8 with int4 tables: https://huggingface.co/xbill9/gemma-4-12B-it-qat-w8a8-int8-emb4
- Google's QAT source checkpoints: https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized
- Google's QAT announcement: https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/
- GSM8K: https://github.com/openai/grade-school-math
- Berkeley Function Calling Leaderboard: https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard
- Bespoke Labs public suite: https://github.com/bespokelabsai/nimble/blob/0e67403/docs/PUBLIC_BENCHMARKS.md
- vLLM TPU documentation: https://docs.vllm.ai/projects/tpu/en/latest/
- Cloud TPU v5e: https://cloud.google.com/tpu/docs/v5e
