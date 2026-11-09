---
title: "Porting Gemma 4 E2B to SGLang-JAX: The Fastest Engine on One TPU v5e"
published: false
description: "SGLang-JAX serves the Gemma 4 31B and 26B A4B on TPU and fails to load E2B. A two-file patch adds per-layer embeddings, shared K/V and the double-wide MLP; on one flex-start v5e-1 it then serves 159.9 tokens per second for one request and 4,249.4 at 64, 2.52x vLLM, with GSM8K level with the bf16 JAX engine."
tags: googlecloud, tpu, jax, gemma
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/docs/sglangjax-e2b-v5e1/devto-cover.1cd3c453.jpg
---

This article provides a step by step guide to porting Gemma 4 E2B into SGLang-JAX and serving it on one Cloud TPU v5e chip, from the patch through an unattended benchmark run on a flex-start VM that deletes itself when it finishes.

https://github.com/xbill9/gemma4-dev

SGLang-JAX is the TPU build of SGLang: a Flax model library over Pallas attention kernels, with SGLang's scheduler and radix cache. It ships Gemma 4 model code for the 31B and the 26B A4B. The question is what it takes to serve the 2B-class E2B on it, and how the result stands against the other engines measured on the same chip type.

---

#### The Results in Brief

Unpatched, SGLang-JAX stops at model load on E2B. A patch to two files, 175 added lines and 14 removed, adds the four things E2B has that the larger Gemma 4 models lack: a null config field, per-layer embeddings, K/V shared from earlier layers, and a double-wide MLP on the sharing layers.

Patched, it serves `google/gemma-4-E2B-it` in bf16 on one `v5litepod-1` at 159.9 tokens per second for one request, the fastest of six engines measured on this chip. At 64 requests it reaches 4,249.4 tokens per second on 128 in / 512 out, 2.52x vLLM serving the int4 build.

GSM8K is 0.9143 on all 1,319 questions, level with the bf16 JAX engine the patch was ported from (+0.0015, 95% range −0.0091 to +0.0121).

---

#### Why a Patch?

Two unattended runs on the unpatched code, sglang-jax `b3e3e16b2`, failed before serving a request. The second reached the scheduler and stopped here:

```text
TypeError: '>=' not supported between instances of 'int' and 'NoneType'
```

E2B's `config.json` carries `"num_global_key_value_heads": null`, and SGLang-JAX reads it with `getattr(tc, "num_global_key_value_heads", <sliding count>)`, which returns the `None`. Past that line, its `models/gemma4.py` has no per-layer embedding and no K/V-sharing code, and E2B uses both.

---

#### At This Point You Should Have…

- A Google Cloud project with TPU v5e quota in `us-west4-a`
- A Cloud Storage bucket for results
- `gcloud` with the `alpha` component
- A clone: `git clone https://github.com/xbill9/gemma4-dev`
- The rig: `tpu-sglangjax-v5e1-2b/` (`tpu.env`, `patches/`, `flight/`)

---

#### Step 1 — What E2B Adds

From the published `config.json`:

| Field | E2B |
| :--- | :--- |
| `num_hidden_layers` | 35 |
| `num_kv_shared_layers` | 20 |
| `hidden_size_per_layer_input` | 256 |
| `num_key_value_heads` | 1 |
| `head_dim` / `global_head_dim` | 256 / 512 |
| `layer_types` | 4 sliding, 1 full, repeating |

- **Per-layer embeddings.** A second token table, 262,144 rows by 35 x 256, gives every layer its own 256-wide input, mixed into the stream after the MLP.
- **Shared K/V.** The last 20 layers compute no K or V; each reads the K/V of the last earlier layer of its own type.
- **A double-wide MLP** on those 20 layers: 12,288 against 6,144.
- **`num_global_key_value_heads: null`**, where the 31B config has a number.

The reference is this repo's pure-JAX E2B engine, which serves the same checkpoint and implements all four.

---

#### Step 2 — Fix the Config

One expression in `configs/gemma4.py`:

```python
tc.num_key_value_heads = (
    getattr(tc, "num_global_key_value_heads", None) or tc.swa_num_key_value_heads
)
```

---

#### Step 3 — Per-Layer Embeddings

The model builds a `[tokens, 35, 256]` tensor once per pass, from the per-layer token table and a projection of the main embedding, as Hugging Face's `Gemma4TextModel` does:

```python
tok = self.embed_tokens_per_layer(input_ids) * ple_dim**0.5
tok = tok.reshape(-1, num_layers, ple_dim)
proj, _ = self.per_layer_model_projection(inputs_embeds)
proj = self.per_layer_projection_norm((proj * hidden_size**-0.5).reshape(-1, num_layers, ple_dim))
per_layer_inputs = (proj + tok) * 2.0**-0.5
```

Each decoder layer takes its slice after the MLP residual and before its layer scale:

```python
gate, _ = self.per_layer_input_gate(outputs)
gate = jax.nn.gelu(gate, approximate=True) * per_layer_input
ple_out, _ = self.per_layer_projection(gate)
outputs = outputs + self.post_per_layer_input_norm(ple_out)
```

🔎 **Tip: SGLang-JAX's model is functional over its K/V pool.** Each layer returns its cache update, so a new input per layer is one more argument down the layer loop and no change to the scheduler.

---

#### Step 4 — Shared K/V, by Copy

The model loop keeps this pass's post-norm, post-RoPE K/V for each source layer, and hands it to the layers that share it:

```python
shared_kv = None
if layer_id >= self.first_kv_shared:
    shared_kv = pass_kv[self.kv_share_source[layer_id]]
hidden_states, kv_fused, _, _, layer_kv = layer(..., per_layer_input=..., shared_kv=shared_kv)
if layer_id < self.first_kv_shared:
    pass_kv[layer_id] = layer_kv
```

A sharing layer applies RoPE to its queries only and writes the source's K/V into its own cache slot. Every layer then holds its own history, so the memory pool and the attention kernel stay as they are. The cost is cache for 35 layers in place of 15:

```text
[2026-11-09 01:46:24] TPU Memory profiling: available_kv_cache=3.6GB, max_tokens=89345, cell_size=43008 bytes
[2026-11-09 01:46:24] ModelRunner final max_total_num_tokens: 103040
```

103,040 tokens of cache at 43,008 bytes a token is enough for 64 running requests at a 16,384-token context. vLLM on the int4 build holds 568,256 tokens on the same chip type.

---

#### Step 5 — Double-Wide MLP and Weight Names

The MLP width doubles from layer 15 on when the config sets `use_double_wide_mlp`, and the loader gains the per-layer names: `model.embed_tokens_per_layer`, `model.per_layer_model_projection`, `model.per_layer_projection_norm`, and per layer `per_layer_input_gate`, `per_layer_projection` and `post_per_layer_input_norm`. The server log on the TPU:

```text
[2026-11-09 01:46:22] Loaded 600 parameters in 6.27s
[2026-11-09 01:46:22] Gemma4 weights loaded successfully!
```

---

#### Step 6 — Install and Patch on the VM

`flight/setup-sgl.sh` installs Python 3.13, SGLang-JAX at the pinned commit with its own `jax[tpu]`, applies the patch to the installed package, and checks that it took:

```bash
python3.13 -m pip install -q --ignore-installed blinker \
  "sglang-jax[tpu] @ git+https://github.com/sgl-project/sglang-jax@${SGL_COMMIT}#subdirectory=python"
SP=$(python3.13 -c "import sgl_jax, os; print(os.path.dirname(os.path.dirname(sgl_jax.__file__)))")
patch -d "$SP" -p2 --forward < $L/patches/gemma4-e2b.diff
python3.13 -c "import sgl_jax.srt.models.gemma4 as m; assert hasattr(m, '_kv_share_source'); print('patch ok')"
```

It records the stack:

```text
sglang-jax b3e3e16b2d3f0281a9e23f547cd4382f4a166eb6 + patches/gemma4-e2b.diff sha256 d6ac1d7a2f8f6e55
Version: 0.1.0rc1.dev113+gb3e3e16b2
jax 0.11.1 jaxlib 0.11.1
Name: libtpu Version: 0.0.46.3 
transformers 5.12.1 torch 2.15.0+cpu
```

🔎 **Tip: install CPU torch and torchvision.** Transformers builds `Gemma4Processor` only when both are present, and the server stops at its tokenizer without it. The TPU does all the model work.

---

#### Step 7 — Three Serving Flags

```text
python3.13 -m sgl_jax.launch_server --model-path google/gemma-4-E2B-it --device tpu --tp-size 1 \
  --dtype bfloat16 --context-length 16384 --max-running-requests 64 --mem-fraction-static 0.8 \
  --reasoning-parser gemma4 --page-size 128 --watchdog-timeout 7200
```

- **`--page-size 128`.** At the default page size of 1, the FlashAttention backend caps the server at 8 running requests at a 16,384-token context. The server logs, at page size 1 and then at 128:

```text
[2026-11-09 01:32:06]   → Final max_running_requests: 8
[2026-11-09 01:41:37]   → Final max_running_requests: 64
```

- **`--watchdog-timeout 7200`.** E2B's full-attention layers are 512 wide, and their Pallas kernel compiles for about five minutes per prefill shape on a fresh v5e:

```text
[EXTEND] PRECOMPILE:  14%|█▍| 1/7 [05:06<30:36, 306.02s/it, bs=8, tokens=64]
[EXTEND] PRECOMPILE:  29%|██▊| 2/7 [10:08<25:19, 303.90s/it, bs=8, tokens=128]
```

The default 300 s watchdog stops the scheduler during a compile that long:

```text
[2026-11-09 01:39:36] Watchdog timeout (watchdog_timeout=300)
```

- **`JAX_COMPILATION_CACHE_DIR`.** With the cache on disk, the next start compiles the same shapes from it: 94 s for seven prefill shapes and 88 s for seven decode shapes.

---

#### Step 8 — Run Unattended on Flex-Start

Create the queued resource:

```text
gcloud alpha compute tpus queued-resources create tpu-sglangjax-v5e1-2b \
  --node-id tpu-sglangjax-v5e1-2b-node --zone us-west4-a --project aisprint-491218 \
  --accelerator-type v5litepod-1 --runtime-version v2-alpha-tpuv5-lite \
  --provisioning-model=flex-start --max-run-duration=24h --valid-until-duration=2h --async
```

Copy `flight/` and `patches/` to `/opt/ladder` on the VM and start one systemd unit:

```text
sudo systemd-run --unit=ladder-sgl bash /opt/ladder/flight/run-sgl.sh tpu-sglangjax-v5e1-2b
```

It runs setup, the six-prompt timing, a 50-question GSM8K check, the full GSM8K and the load sweep, then uploads the results and deletes the queued resource whatever the outcome:

```text
2026-11-09 01:50:23 START one:sglang ready_s=270.2 server_log=/opt/ladder-out/2026-11-08-first-light-v5e1/one-sglang/server.log
2026-11-09 01:53:27   acc:sglang gsm8k-gate rc=0 {"task": "gsm8k", "max_tokens": 2048, "n": 50, "right": 45, "accuracy": 0.9, "errors": 0, "truncated": 0, "wall_s": 125.4}
2026-11-09 02:55:56   acc:sglang gsm8k rc=0 {"task": "gsm8k", "max_tokens": 2048, "n": 1319, "right": 1206, "accuracy": 0.9143, "errors": 0, "truncated": 3, "wall_s": 3748.3}
2026-11-09 02:59:53 END sweep:sglang ok=True
2026-11-09 02:59:58 SUITE-DONE
```

---

#### Step 9 — One Request at a Time

Six chat prompts, 256 tokens each, greedy, thinking off, three repeats; the geometric mean of each prompt's median. The other engines are from the v5e-1 serving ladder on the same chip type:

| Engine | Weights | 1 request | 4 at once | x vLLM |
| :--- | :--- | :--- | :--- | :--- |
| SGLang-JAX, patched | bf16 | 🥇 159.9 | 🥇 631.5 | 1.111x |
| Rust model over PJRT | bf16 | 145.2 | 145.4 | 1.009x |
| vLLM | int4 | 143.9 | 477.2 | 1.000x |
| JAX engine | bf16 | 134.3 | 150.1 | 0.933x |

SGLang-JAX runs at 1.19x the JAX engine whose E2B code the patch follows. Its time to first token is 20.2 ms, and it repeats its own greedy text on all six prompts.

Its greedy text matches none of the other engines word for word on these prompts; the bf16 JAX engine and the Rust model match each other on 3 of 6. GSM8K below is the check on its answers.

---

#### Step 10 — 1 to 64 Requests

Two shapes, five repeats, a unique prompt per request, aggregate output tokens per second:

| Requests | SGLang-JAX, 128/512 | vLLM, 128/512 | SGLang-JAX, 512/128 | vLLM, 512/128 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 🥇 160.3 | 144.5 | 🥇 154.2 | 140.6 |
| 8 | 🥇 943.1 | 778.4 | 🥇 915.6 | 682.2 |
| 16 | 🥇 1,671.9 | 1,112.8 | 🥇 1,663.0 | 937.9 |
| 32 | 🥇 2,550.2 | 1,462.5 | 🥇 2,358.4 | 1,156.8 |
| 64 | 🥇 4,249.4 | 1,689.1 | 🥇 2,758.9 | 1,288.4 |

From 8 requests up SGLang-JAX leads on both shapes, 2.52x and 2.14x vLLM at 64. At 2 requests of 512 in / 128 out it trails (117.4 against 257.4).

🔎 **Tip: read its medians, and expect spread.** SGLang-JAX's fastest repeat was 49% to 205% above its slowest at a level; vLLM's spread was at most 5.0%.

---

#### Step 11 — Answers

All 1,319 GSM8K test questions, 2,048-token limit, one request at a time, paired question by question with a 95% range from 2,000 resamples:

| Against | Their GSM8K | SGLang-JAX minus theirs [95%] |
| :--- | :--- | :--- |
| JAX engine, bf16 | 0.9128 | +0.0015 [−0.0091, +0.0121] |
| Rust model over PJRT, bf16 | 0.9113 | +0.0030 [−0.0076, +0.0144] |
| vLLM, int4 | 0.9007 | +0.0136 [+0.0023, +0.0258] |

SGLang-JAX scores 0.9143 (1,206 right, 3 replies at the token limit) in 3,748 s. It is level with both bf16 engines, so the patched model answers as the reference does. Its lead over the int4 vLLM build has a range above zero.

SGLang-JAX has no Gemma 4 tool-call parser, so BFCL was not run.

---

#### Step 12 — Tear Down

The run deletes its own queued resource after the upload checks out:

```text
2026-11-09 03:00:08 uploaded gs://aisprint-491218-ladder-v5e1/2026-11-08-first-light-v5e1/tpu-sglangjax-v5e1-2b/ladder-run.tgz sha256 2c35d88e65a7b8190f90ba1cc6b44638c013d047d1376ef94d054cd159a6a57c (29 entries)
2026-11-09 03:00:08 deleting queued resource tpu-sglangjax-v5e1-2b
2026-11-09 03:00:10 delete requested (rc=0)
```

---

#### Compare and Contrast

| | SGLang-JAX, patched | vLLM | JAX engine |
| :--- | :--- | :--- | :--- |
| 1 request | 🥇 159.9 | 143.9 | 134.3 |
| 64 requests, 128/512 | 🥇 4,249 | 1,689 | one at a time |
| GSM8K | 🟢 0.9143 | 🟢 0.9007 | 🟢 0.9128 |
| Tool calls | ❌ no parser | 🟢 BFCL 0.920 | ❌ none |
| Cache on one chip | ⚠️ 103,040 tokens | 🟢 568,256 tokens | — |
| Fresh start | ⚠️ ~5 min per prefill shape | ⚠️ 34 min | 🟢 42 s |
| As installed | ❌ two-file patch | 🟢 pinned image | 🟢 yes |

---

#### So, Which One?

For the most throughput from one v5e chip on Gemma 4 E2B, run SGLang-JAX with this patch: it is the fastest engine measured for one request and at every load from 8 requests up, and its answers match the bf16 reference. Give it `--page-size 128`, a long watchdog and a compile cache. When the workload needs tool calls, a deep cache or an unpatched install, vLLM remains the engine to serve.

---

#### Summary

The goal of this article was to serve Gemma 4 E2B on SGLang-JAX on one Cloud TPU v5e chip. The key was porting E2B's four differences from an engine that already serves it, and storing shared K/V again in each sharing layer so the cache and kernels needed no change. The results were:

- 🟢 A two-file patch, 175 lines added and 14 removed, makes SGLang-JAX load and serve E2B.
- 🟢 159.9 tokens per second for one request, the fastest of six engines measured on this chip, and 1.19x the JAX engine it was ported from.
- 🟢 4,249.4 tokens per second at 64 requests of 128 in / 512 out, 2.52x vLLM; 2,759 on 512 / 128, 2.14x.
- 🟢 GSM8K 0.9143, level with the bf16 JAX engine and the Rust model.
- ⚠️ Shared K/V stored per layer costs 43,008 bytes a token, 103,040 tokens on the chip.
- ⚠️ Without `--page-size 128` the server runs 8 requests at once; without a long watchdog a fresh start dies compiling.
- ⚠️ Repeat-to-repeat spread of up to 205% at a load level.
- ❌ No Gemma 4 tool-call parser, so no BFCL.

Scope: one flex-start `v5litepod-1` in `us-west4-a`, sglang-jax `b3e3e16b2` with the patch (sha256 `d6ac1d7a2f8f6e55…`), jax 0.11.1, `google/gemma-4-E2B-it` bf16 at a 16,384-token context, greedy decoding, thinking off, on 2026-11-09. The other engines ran on 2026-11-06 and 07 on other v5e-1 VMs on jax 0.12.0.1, vLLM on the int4 build, so the throughput comparison is engine and weights together. One-request speed is six prompts of 256 tokens, three repeats; load is two synthetic shapes at 1 to 64 requests, five repeats each; accuracy is one pass of GSM8K. The start-up time in the run came with a compile cache already on the VM.

The strategy for porting Gemma 4 E2B to SGLang-JAX on Cloud TPU was validated with an incremental step by step approach.

---

#### References

- [gemma4-dev on GitHub](https://github.com/xbill9/gemma4-dev)
- [The rig: tpu-sglangjax-v5e1-2b](https://github.com/xbill9/gemma4-dev/tree/main/tpu-sglangjax-v5e1-2b)
- [The patch: gemma4-e2b.diff](https://github.com/xbill9/gemma4-dev/blob/main/tpu-sglangjax-v5e1-2b/patches/gemma4-e2b.diff)
- [First-light run: 2026-11-08-first-light-v5e1](https://github.com/xbill9/gemma4-dev/blob/main/tpu-sglangjax-v5e1-2b/benchmarks/runs/2026-11-08-first-light-v5e1/REPORT.md)
- [Six Serving Engines on One Cloud TPU v5e Chip](https://dev.to/gde/six-serving-engines-on-one-cloud-tpu-v5e-chip-gemma-4-e2b-22hn)
- [SGLang-JAX](https://github.com/sgl-project/sglang-jax)
- [Cloud TPU v5e](https://docs.cloud.google.com/tpu/docs/v5e)
- [JAX](https://github.com/jax-ml/jax)
- [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)
- [GSM8K](https://github.com/openai/grade-school-math)
