# Pure-JAX Gemma 4 E2B on TPU v6e-1 — cache-correct decode measurements

**Run:** 2026-07-28 · `jax-bench-v6e1` (ct6e-standard-1t, flex-start, europe-west4-a), deleted after collection · ~29 min ≈ $0.65
**Stack:** Python 3.13.14 (system, no venv) · JAX/jaxlib 0.11.0 · libtpu 0.0.44.1 · numpy 2.5.1 · scipy 1.18.0 · ml_dtypes 0.5.4
**Harness:** `ports/gemma4/jax_e_benchmark_sweep_v2.py` — jitted prefill and steady-state cached decode timed separately, 2 warmup iterations discarded, median of 5–7 repeats.
**Weights:** synthetic, architecture-shaped (throughput is value-independent under static shapes). Note the config's KV head count does not match the shipped checkpoint — see the caveat in the harness — so memory/OOM behaviour here is pessimistic.

> **⚠ These decode numbers predate a working model.** They were measured on
> 2026-07-28 against synthetic weights and an architecture that was wrong in five
> ways — see "Five bugs between loading and generating" below. The first correct
> generation from the real checkpoint happened *after* these runs. The fixes are
> close to FLOP-neutral (norm reordering, one multiply per layer, a RoPE frequency
> layout change; removing the attention softcap is a small win), so the figures are
> expected to hold roughly — but "expected to hold" is the reasoning that produced
> the 2.79x retracted twice on this page. **Re-run before citing.**

> **⚠ SUPERSEDED: this page's central conclusion is wrong.** Section 2c concludes
> "the binding constraint is **memory, not latency**." Both halves have since been
> retracted by `benchmarks/runs/2026-07-29-kv-quant-v6e1/REPORT.md` and
> `benchmarks/runs/2026-07-29-real-http-v6e1/REPORT.md`:
>
> * **Every capacity ceiling here is ~2x low.** These runs used no buffer
>   donation, so `dynamic_update_slice` kept two full KV caches live at once
>   (addendum 4). Re-bisected with donation, ctx 8192 goes 712,704 -> **1,425,408**
>   resident KV tokens at bf16 and **2,818,048** at int8 — exactly 2.000x, i.e.
>   B=87 -> B=344 concurrent 8K sequences (addendum 5).
> * **The OOM was not the KV cache.** Section 2c already identifies the real cause
>   — `eager_attention_jax` materializing a fp32 `[B, heads, S_q, S_kv]` score
>   matrix — but still labels the constraint "memory". It is a kernel defect.
> * **The measured constraint is request scheduling, not memory.** On the real
>   checkpoint over HTTP, one stream decodes at 140.5 tok/s and eight concurrent
>   streams aggregate to 143.3 tok/s, with no OOM and no failed request. The
>   device serializes B=1 executions; nothing runs out of HBM.
>
> The goodput table in 2c is also measured against a batch ceiling that no longer
> exists ("64+ HBM OOM"), and the "B=32 / 5,743 tok/s" figure was separately
> retracted as an artifact of prefilling every sequence simultaneously.

## 1. Baseline: what the engine actually does

W4A16 reference path (dequantize → BF16 → matmul), bf16 KV cache.

| users (B) | ctx | prefill (TTFT) | decode step | aggregate | per-user |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 512 | 13.8 ms | 20.57 ms | 48.6 tok/s | 48.6 tok/s |
| 1 | 2048 | 43.3 ms | 20.65 ms | 48.4 tok/s | 48.4 tok/s |
| 2 | 512 | 21.9 ms | 7.37 ms | 271.2 tok/s | 135.6 tok/s |
| 2 | 2048 | 98.9 ms | 7.88 ms | 253.7 tok/s | 126.9 tok/s |
| 4 | 512 | 34.0 ms | 7.55 ms | 529.5 tok/s | 132.4 tok/s |
| 4 | 2048 | 200.2 ms | 9.43 ms | 424.3 tok/s | 106.1 tok/s |
| 8 | 512 | 67.7 ms | 7.90 ms | 1,012.4 tok/s | 126.5 tok/s |
| 8 | 2048 | 492.7 ms | 11.10 ms | 720.8 tok/s | 90.1 tok/s |

Raw: [`reference.json`](reference.json)

## 2. The B=1 penalty: three answers, and why the third is right

A previous sweep reported a "~2.7x per-user speedup" going from one stream to two. That
claim was withdrawn on the argument that MXU underutilization should make B=1 and B=2 take
roughly the *same* wall time, so a 2.8x penalty for the smaller batch looked like a
first-cell measurement artifact.

**That reasoning was wrong.** Re-running the sweep with the batch order reversed — so B=1 is
measured *last*, after the cache and compilation are thoroughly warm — reproduces it:

| B | forward order (B=1 first) | reversed order (B=1 last) | delta |
| ---: | ---: | ---: | ---: |
| 1 | 20.57 ms | 20.52 ms | −0.05 |
| 2 | 7.37 ms | 7.30 ms | −0.07 |
| 4 | 7.55 ms | 7.43 ms | −0.12 |
| 8 | 7.90 ms | 7.74 ms | −0.16 |

Every cell agrees within 1%. Per-user throughput 48.7 → 137.0 tok/s = **2.81x**, against the
withdrawn figure of 2.80x. The effect is real and reproducible; a single stream costs ~2.8x
per token what each of two concurrent streams costs.

**Cause not established.** Decode step cost is flat across B=2/4/8 (7.3–7.9 ms), consistent
with a per-step cost that barely depends on batch — yet B=1 sits at 20.5 ms rather than the
~7.4 ms that flatness would predict.

Raw: [`reversed-batch-order.json`](reversed-batch-order.json)

### 2b. Corrected: it was mostly the wrong model config

Everything above ran against `E2B_CONFIG` values that did not match the shipped checkpoint.
Reading `config.json` off the real model showed `hidden_size` 2048 → **1536**,
`num_key_value_heads` 4 → **1**, `num_global_key_value_heads` 4 → **1**. Re-running the sweep
with the corrected dimensions:

| B | step (wrong cfg) | step (real cfg) | speedup | per-user (real) | aggregate (real) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 20.57 ms | **6.80 ms** | 3.03x | 147.1 tok/s | 147.1 tok/s |
| 2 | 7.37 ms | **4.86 ms** | 1.52x | 206.0 tok/s | 411.9 tok/s |
| 4 | 7.55 ms | **4.89 ms** | 1.55x | 204.6 tok/s | 818.5 tok/s |
| 8 | 7.90 ms | **4.88 ms** | 1.62x | 204.8 tok/s | 1,638.6 tok/s |
| 16 | — | 5.08 ms | — | 196.9 tok/s | 3,151.0 tok/s |
| 32 | — | 5.57 ms | — | 179.5 tok/s | **5,743.0 tok/s** |

**The single-stream penalty is 1.40x, not 2.8x.** Sequence of claims about this one number:

| measurement | B=1 → B=2 per-user | verdict |
| --- | ---: | --- |
| original sweep (no KV cache, un-jitted prefill) | 2.80x | withdrawn as an artifact |
| rebuilt harness, **wrong** model config | 2.79x | "retraction reversed" — also wrong |
| rebuilt harness, **real** model config | **1.40x** | current best measurement |

The middle row reproduced the original figure closely enough to look like confirmation, which
is exactly why it was convincing. Both runs shared the same wrong architecture, so they agreed
with each other rather than with the model. Reproducibility is not validity when the two runs
share an assumption.

A 1.40x single-stream penalty is still real and still unexplained by arithmetic intensity
(which predicts B=1 ≈ B=2), but it is a modest implementation effect rather than the headline
phenomenon it was written up as.

Raw: [`batch-scaling-realconfig.json`](batch-scaling-realconfig.json)

### 2c. Goodput: we are memory-limited, not latency-limited

> **SUPERSEDED — see the retraction at the top of this page.** The ceilings below
> are ~2x low (no buffer donation), the OOM is a kernel defect rather than a
> capacity limit, and the constraint measured on the real checkpoint is request
> scheduling. Retained as written for provenance.

Applying the service-objective framing of DOI 10.5281/zenodo.21462837 (see
`docs/references/tpu-inference-measurement-series.md`), at a 50 ms/token objective:

| B | ms/token/user | meets 50 ms? | goodput |
| ---: | ---: | :---: | ---: |
| 1 | 6.80 | yes | 147 tok/s |
| 8 | 4.88 | yes | 1,639 tok/s |
| 16 | 5.08 | yes | 3,151 tok/s |
| 32 | 5.57 | yes | **5,743 tok/s** |
| 64+ | — | — | HBM OOM |

**Every measured point clears the objective with ~9x headroom**, and throughput was still
climbing when the sweep hit HBM. That is the opposite of the referenced 31B result, where
goodput collapses at the knee while throughput plateaus: for a 2B model on one chip the
binding constraint is **memory, not latency**.

The OOM is not the KV cache (18 KiB/token here). It is `eager_attention_jax` materializing the
full `[B, heads, S_q, S_kv]` score matrix in fp32 — at B=64, ctx 512 that is hundreds of MB per
layer. So the unlock for higher concurrency is a flash/splash attention kernel that never
materializes the score matrix, not weight-traffic reduction.

**What the earlier critique got right:** the old harness timed prefill on an *un-jitted*
call. It reported ~544 ms at B=1; the correct measurement is **13.8 ms** at ctx 512 — the old
number was inflated ~40x and measured op dispatch, not prefill. That part stands. Its
decode-step numbers happened to land close to correct because at short context attention is
a small fraction of per-step work, so the missing KV cache cost little.

## 3. Optimizations that did not pay off

### Fused W4A16 Pallas kernel — a regression

| B | ctx | reference step | fused step | ratio |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 512 | 20.57 ms | 34.94 ms | **0.59x** |
| 1 | 2048 | 20.65 ms | 35.00 ms | **0.59x** |
| 2 | 512 | 7.37 ms | 34.87 ms | **0.21x** |
| 2 | 2048 | 7.88 ms | `CompileTimeScopedVmemOom` | — |
| 4, 8 | any | — | `CompileTimeScopedVmemOom` | — |

Two separate problems:

1. **It doesn't tile the sequence axis.** The kernel's `BlockSpec` loads all of `x` as one
   VMEM block, so required VMEM grows with `B x bucket_S` and blows the 32 MB scoped limit
   from B=2/ctx=2048 upward. That is a kernel design fix, not a tuning knob.
2. **It's slower where it fits.** The int4 unpack (8 shifts/masks/subtracts plus scale
   expansion per chunk) runs on the VPU on the critical path of what is, at decode, a
   matrix-*vector* product. There is no sequence length to amortize it over.

The prediction behind this work — 4-bit weights cut decode traffic ~4x, so ~3.4x on the
memory floor — did not survive contact with the chip. The floor we compared against
(~3.6 ms/token at B=1, versus a measured 20.6 ms) counted **weights and the LM head only**.
It omitted KV-cache traffic and the logits convert/copy, so "decode is not bandwidth-bound"
overstates what the gap proves — see `docs/references/tpu-inference-measurement-series.md`,
where a published series measures decode attention at an arithmetic intensity of ~8 FLOPs/byte
that never amortizes across batch, and the logits pipeline at 18% of the step. What the
measurement does establish is narrower and still decisive: **this kernel, as written, is slower
than the reference path**, so weight-traffic reduction is not where the win is until the step is
properly attributed by a profile.

Raw: [`fused-pallas.json`](fused-pallas.json)

### int8 LM head — within noise

| B | bf16 LM head | int8 LM head | speedup |
| ---: | ---: | ---: | ---: |
| 1 | 20.57 ms | 20.55 ms | 1.00x |
| 2 | 7.37 ms | 7.10 ms | 1.04x |
| 8 | 7.90 ms | 7.52 ms | 1.05x |

4–5% at batch ≥ 2, nothing at B=1 — for ~0.8% logit error. Same root cause: the LM head is
the largest single *read*, but reads are not the constraint. Not worth enabling by default.

Raw: [`int8-lm-head.json`](int8-lm-head.json)

### Pallas layout: `interleaved` is TPU-infeasible

The fused kernel originally unpacked nibbles plane-major (column `i*(K/8)+j`) while
contracting activations in natural order — numerically wrong, and invisible because a bare
`except Exception` fell back to the reference on any host without a TPU. Both corrected
layouts verify exact under Pallas interpret mode, but on real hardware:

- `interleaved` (unpack to checkpoint order in-tile) — **OOMs**: the in-tile `stack+reshape`
  is padded by Mosaic to 35.43 MB against a 32 MB scoped-VMEM limit, for a single 1024x2048
  projection. This vindicates the original plane-major choice; the layout was not arbitrary.
- `plane` (plane-major weights + per-chunk activation permutation) — exact on TPU:
  max |diff| 0.000000 at seq 1 and 8, 0.0039 at seq 64 (bf16 accumulation rounding).

`plane` is therefore the default layout; `interleaved` is retained for reference/testing only.

## 4. Standing conclusions

- **Per-user throughput is best at 2–8 concurrent streams** (~127–136 tok/s each at ctx 512),
  not at 1. A single stream pays a ~2.8x per-token penalty. For an agent fleet this is good
  news, and it is the opposite of the usual "batching costs latency" intuition.
- **Prefill scales as expected** — 13.8 → 43.3 ms as ctx goes 512 → 2048 at B=1, and
  67.7 → 492.7 ms at B=8 — which is the sanity check the old un-jitted harness failed.
- **Weight-traffic optimizations (fused int4, int8 LM head) do not pay here.** The ~6x gap
  between our weights-only bandwidth floor and the measured step is unattributed, not proof
  that reads are irrelevant — the floor omitted KV traffic and the logits pipeline. Profile
  before optimizing further.
- **The B=1 penalty contradicts published measurements** of this class of workload, where
  weight-read cost is flat across an 8x batch increase (as our own B=2/4/8 are). That points at
  an implementation artifact in the batch-1 path rather than a hardware property. See
  `docs/references/tpu-inference-measurement-series.md`.
- **We pay bandwidth for masked padding**: decode attends over the whole preallocated KV buffer
  every step. That is the documented failure mode of an oversized KV block size, worth up to
  +68.7% elsewhere when fixed.

## 5. Reproduce

```bash
python3 ports/gemma4/jax_e_benchmark_sweep_v2.py \
  --batch-sizes 1,2,4,8 --contexts 512,2048 --repeats 5 --json-out out.json
```

Environment as above. Provision it with the maintained MCP path:

```
create_tpu_vm_instance(accelerator="v6e-1", workload="jax")
wait_for_jax_ready()
verify_jax_tpu()
```

`startup-script.as-run.txt` is the script that actually ran here, kept verbatim for
traceability. It is broken — no `set -e`, and a pip too old for
`--break-system-packages` — so it reported ready with nothing installed and the stack
was fixed by hand. It is stored as `.txt` without a shebang so it cannot be reused;
`startup_script_jax_template.sh` is the corrected version.


## 6. Five bugs between loading and generating

Found on 2026-07-28 by diffing against `transformers.models.gemma4.modeling_gemma4`
(shipped in transformers 5.14.1 — the reference was readable locally all along, no
torch and no TPU required). Each only became visible after the previous was fixed:

1. **Loader read nothing.** The checkpoint is multimodal (751 audio + 659 vision
   tensors); the text decoder lives under `model.language_model.`, not `model.`.
   Every lookup returned `None`, and "load" succeeded in 6.4 s with **0.00 GB** on
   device. Now auto-detects the prefix and raises listing what is missing.
2. **Norm placement.** `post_attention_layernorm` was applied as a *pre*-norm for
   the MLP. Gemma norms the attention *output* before the residual add, then
   pre/post-norms the feed-forward block.
3. **Attention softcap.** Scores were softcapped at 30.0. The config declares
   `final_logit_softcapping` only — no `attn_logit_softcapping` — and capping the
   scores saturates tanh and destroys the attention distribution.
4. **`layer_scalar` never applied.** The reference ends each decoder layer with
   `hidden_states *= layer_scalar` — the whole residual stream, after every residual
   add. It is the counterweight to this checkpoint's large RMSNorm weights
   (`final_norm` mean ~14, max 118). Without it the stream grows layer over layer and
   the logits pin against the +/-30 softcap. Fixing this moved output from random
   multilingual tokens to structured chat markers.
5. **RoPE frequency layout.** `rotate_half` pairs channel *i* with *i + d/2*, which
   requires `cat(freqs, freqs)`; we used `repeat(2)` (interleaved), so every channel
   rotated against the wrong frequency. Partial rotary is also done the reference way
   now — mask the frequencies (cos->1, sin->0 past the factor) rather than slice
   channels. This moved output from garbage to a verbatim echo of the prompt.

Plus one self-inflicted regression: a `1/sqrt(head_dim)` query scaling was added on
the theory that scores were 16x too large. The reference sets `scaling = 1.0` —
`q_norm`/`k_norm` already normalize before the dot product. Reverted.

The last piece was not architectural: this tokenizer does **not** add `<bos>`
(`tok("hello")` -> `[23391]`), and without it Gemma echoes the prompt. `JaxGemmaEngine`
now prepends it.

**First correct output:**

```
Q: What is 2+2?              A: '4'
Q: The capital of France is  A: 'Paris'
Q: What is 2+2?   (plain)    A: '2+2 equals 4.'
```
