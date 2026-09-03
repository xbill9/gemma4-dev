# Quantization on vLLM + tpu_inference

What the **serving stack** supports for Gemma 4 — which routes are reachable, which are dead, and how to
enable the live ones. These are properties of vLLM and tpu_inference, not of any one chip, so they apply
equally to every `*-vllm-*` rig regardless of the hardware slot.

The companions: `HARDWARE.md` for which formats the silicon can actually *compute* in (that part **does**
vary by generation and changes the value of everything below), `MODELS.md` for checkpoint properties, and
each rig's `benchmarks/runs/` for what was measured where.

Verified against vLLM `0.26.1rc1.dev125+ga7a204cc6` / `vllm/vllm-tpu:nightly`
(`sha256:2a4a1f82…`) on 2026-08-07. The stack moves; re-check before trusting a negative.

## What the hardware allows before the software matters

From `HARDWARE.md`: **v5e and v6e have bf16 and int8 in the MXU and no fp8; v7/Ironwood is the first with
native fp8.** Consequences that shape every choice below, on v5e/v6e:

- **int8 is the only low-precision format with a compute win** (2x bf16).
- **fp8 and 4-bit are storage/bandwidth-only** — values widen back to bf16 before the matmul.
- A quantization that reduces *footprint* is still worth having when the model does not otherwise fit.
  That is the entire case for 4-bit on 12B, and it does not require any MXU support.

On v7 this inverts for fp8. **Do not carry a conclusion here across generations without rechecking it.**

## Choosing a format and a size: what quantization actually buys

**Added 2026-08-28.** Three axes decide a rig's configuration, and **only one of them is a
quantization decision.** They are routinely conflated, and the conflation is expensive.

### 1. Compute dtype must match the chip — this dominates, and is not a quant choice

Get it wrong and nothing else you do matters. MEASURED on a T4G
(`gpu-jax-g5g-2b/benchmarks/runs/2026-08-27-baseline-xprof-g5g/`), 86.8% of decode goes to dtype work:

```
dtype conversion  39.60 ms/step  54.0%      <- computes nothing
fp32 GEMV         24.08 ms/step  32.8%      <- what the converts leave behind
                                 -------
                                  86.8% of decode
```

That is **larger than any quantization decision in this document would buy**.

**But be careful attributing it — on that rig it is NOT a storage-dtype mismatch.** The obvious reading
is "weights are bf16, the device computes float16, so XLA converts". That was tested directly on
2026-08-28 by converting the whole tree to float16 and re-profiling: the convert and GEMV kernels came
back **identical to the microsecond**, for **+0.0% throughput**. At `B=1` decode is a matrix-*vector*
product, cuBLAS dispatches `gemvx::kernel<..., float, float, float, ...>` reporting
`is_op_tensor_core_eligible = False`, and **a GEMV has no half-precision path** — so the weights are
promoted to fp32 whatever they are stored as. The converts were promotions *to* fp32.

**The driver is still real and still first.** bf16 on a pre-Ampere GPU does not *fail*, it emulates
through fp32, so the numbers come out right and every matmul quietly pays. What the T4G measurement
adds is that **a large dtype cost can have a cause other than the storage dtype** — read the kernel
signature before choosing a remedy, or you will spend three attempts on the wrong one.

| Target | Compute dtype | Trap |
| :--- | :--- | :--- |
| v5e / v6e / v7 | `bfloat16` | — |
| L4 (SM 8.9) | `bfloat16` | — |
| **T4G (SM 7.5)** | **`float16`** | bf16 emulates silently; fp8 absent |
| inf2 | see `HARDWARE.md` | — |

**Check this before evaluating any quantization scheme**, and check it from the device rather than from
config: `ports/gemma4/jax_e_model.py` reads the live compute capability and picks, which is the pattern
to copy.

### 2. Quantization buys residency and bandwidth. It essentially never buys FLOPS

Across every target in this monorepo the only genuine compute wins are **int8 on v5e/v6e (2x bf16)** and
**fp8 on v7**. Everything else — 4-bit anywhere, fp8 on v5e/v6e, and every scheme on Turing — is
**dequantize-then-matmul**: the weights are unpacked to the compute dtype and the same matmul runs.

Measured twice on the same rig, and the pattern held both times — **the memory claim lands to the byte
and the speed claim does not land at all**:

| Lever | Memory | Throughput |
| :--- | :--- | :--- |
| `ple_bits=4` (E2B) | −3.505 GB against −3.51 predicted (**0.1% error**) | **0.0%** — decode identical to `ple0` |
| `int8_lm_head` | +0.403 GB, exact to the byte | **+2.3%** |

`int8_lm_head` is the instructive one: it looks like an int8 matmul and is not. The table is
**dequantized to fp16 in full — 0.75 GiB — on every decode step**, and Turing's int8 tensor cores
(~130 TOPS) are never touched. It halves the bytes *read* and pays a full-table convert regardless.

**Corollary for sizing a new rig: pick the COARSEST quantization that fits.** Finer quantization is not
faster; it is more unpack work in the hot path for the same matmul. Quantize to hit a residency target,
then stop.

**A fused kernel is the exception, and it does not travel.** The W4A16 Pallas kernel *is* fused on TPU
(16 MB VMEM per core) and is **refused at startup on Turing** — it needs 550 KiB–1.1 MiB of shared memory
per block against a 64 KiB ceiling. Same checkpoint, same code, completely different economics. Never
assume a quantized path that pays on TPU will pay on a GPU rig.

### 3. What actually binds as models grow is TRANSIENTS, not resident weights

This is the axis most likely to be missed when planning a larger sibling, because the weight table in
`MODELS.md` invites you to plan against residency alone. MEASURED on a T4G with a 14.07 GB budget:

| Model | Weights | Fits? | What actually failed |
| :--- | ---: | :--- | :--- |
| E2B `ple4` | 3.05 GB | serves | — |
| **E4B** | fits | **no** | OOM **5.25 GiB during load** |
| **12B** | **8.15 GB — fits easily** | **no** | OOM **12.61 GiB per request** |

Both failures are transients, on models whose weights were never the problem. Three further properties
of the same class:

- **Fragmentation, not free bytes.** Allocator fragmentation measured **0.661** at peak with 2.9 GiB
  free — two of three quantization bugs on that rig failed with GBs nominally free. **Quote the largest
  contiguous block in any capacity claim.**
- **Prefill temporaries have a flat term AND a linear one.** Flat below ~4K, then ~**0.9 MiB/token**.
  A context limit derived from KV arithmetic alone will be wrong: that rig advertised
  `MAX_MODEL_LEN=8192` while 5,120 tokens OOMed, and was lowered to 4096.
- **Quantizing costs memory while it runs.** `quantize_ple_table` upcasts to float32 and needs >15 GiB
  of host RSS on E2B; the destination is allocated before the source is freed unless explicitly
  released. **The load-time peak, not the steady state, sets the floor.**

**So the order of operations for a new rig is:** match the compute dtype to the chip → size the model
against transients rather than weights → choose the coarsest quantization that reaches that residency
target → and only then look for a compute win, which exists on v5e/v6e int8 and v7 fp8 and essentially
nowhere else.

## Gemma 4 is JAX-path only, and that decides everything

Gemma 4 exists solely as a JAX implementation — `models/jax/gemma4.py`, `gemma4_mm.py`, `gemma4_mtp.py`,
with nothing under `models/vllm/`. Quant methods resolve through `layers/jax/quantization/`, so anything
in the torch path is unreachable **no matter what `tpu_platform.supported_quantization` advertises**.
That list is actively misleading read on its own.

| Route | Implemented in | Reachable for Gemma 4? |
| :--- | :--- | :--- |
| **qwix PTQ** — int8/int4/fp8, weight-only or W8A8 | `models/jax/utils/qwix/` | reachable, but **does not boot** — see below |
| compressed-tensors fp8 w8a8 | `layers/jax/quantization/compressed_tensors.py` | yes (needs a pre-quantized ckpt) |
| compressed-tensors **w4a16 / wNa16** | nowhere on the JAX path | **no — `NotImplementedError`** |
| **mxfp4** (4-bit) | `layers/jax/quantization/mxfp4.py` | **no — MoE-only**, see below |
| compressed-tensors int8 w8a8, w4a8 fp8, w4a4 nvfp4 | `layers/vllm/.../schemes/` | no — torch path only |
| AWQ | `layers/vllm/quantization/awq.py` | no — torch path only |
| GGUF / q4_0 | absent from `QUANTIZATION_METHODS` — **and not a TPU-only gap**, see below | no |

The JAX compressed-tensors dispatcher handles `_is_fp8_w8a8`, then falls off the end:

```python
# TODO: w4a8 / wNa16 schemes need their own JAX methods (not yet ported).
raise NotImplementedError(...)
```

> **The GGUF row is a property of vLLM itself, not of the TPU platform. Verified 2026-09-02** against a
> stock **vLLM 0.26.0 CUDA** install: `grep -ril gguf` over the entire installed package returns **two**
> files, both incidental (`lora/layers/utils.py`, `models/qwen2_moe.py`); there is no `gguf.py` under
> `model_executor/layers/quantization/`; and `QUANTIZATION_METHODS` lists 31 entries with no `gguf` among
> them. So the original finding — recorded against `tpu_platform.py`'s `supported_quantization` — understated
> its own scope. **No vLLM rig in this monorepo can load a GGUF, on any platform slot.** Do not "check the
> CUDA build" expecting a different answer; it has been checked.

`wNa16` is w4a16 — exactly the format of Google's QAT releases
(`google/gemma-4-{E2B,12B}-it-qat-w4a16-ct`). A **second, independent** failure hits the QAT exports:
`k_norm.weight` "missing" for layers 15-34. Upstream:
[tpu-inference #3225](https://github.com/vllm-project/tpu-inference/issues/3225).
`tpu-pytorch-v5e1-12b` is currently pinned at `gemma-4-12B-it-qat-w4a16-ct` and does not load.

> **Which export shows which failure is not cleanly recorded.** The devto forensics tabulate them
> against **E2B**, and there they are *different* checkpoints: `-qat-w4a16-ct` dies on the int4
> compressed-tensors scheme being unimplemented for `per_layer_model_projection`, while
> `-qat-q4_0-unquantized` is the one that dies on `k_norm`. Whether the 12B `-qat-w4a16-ct` reaches
> the `k_norm` error or hits the `wNa16` `NotImplementedError` first has not been separately
> established. Don't assume both failures apply to both exports.

> **The `k_norm` failure is a loader bug. Re-diagnosed 2026-08-07 — this supersedes both the original
> architectural explanation and its retraction.**
>
> The sequence: this file first said layers 15-34 "legitimately have no K projection", citing KV
> sharing. `MODELS.md` then read the safetensors headers, found **all 35 layers carry `k_proj` and
> `k_norm`** (`layers missing k_norm: []`), and the claim was retracted as unexplained.
>
> **That retraction over-corrected, because `MODELS.md` read the *base* export.** The forensics in
> `tpu-jax-v5e1-2b/devto-jax-gemma4-e2b.md` and `tpu-pytorch-v5e1-12b/devto-post.md` read *both*
> repos: the plain export ships `self_attn.k_norm` for all 35 layers, and **the QAT export ships it
> only for the 15 non-KV-shared layers** — both configs declaring `num_kv_shared_layers: 20`. Those
> two readings are compatible: the base carries tensors layers 15-34 never use, and the QAT export
> drops them. So the QAT export is the architecturally honest one and the loader is wrong to demand
> the tensors. #3225 proposes skipping K/V-side parameters for KV-shared layers.
>
> **What survives from the retraction:** KV sharing is a runtime property, so the base checkpoint's
> header count is not evidence about the QAT export *in either direction*. Never cite it as proof
> about the QAT export — that is the move that produced both the original error and the overshoot.
>
> **Independent confirmation that `-qat-w4a16-ct` is loadable** — measured 2026-07-31 on the pure-JAX
> engine in `~/tpu-jax-*` (a **different stack**; see the boundary note in the next section), filed
> in the artifact rigs: `gemma-4-12B-it-qat-w4a16-ct` loads at **100% exact token parity** with the
> HF PyTorch reference (`tpu-jax-v6e1-12b-w4a16`), and `gemma-4-31B-it-qat-w4a16-ct` loads, fits one
> v6e chip and answers correctly **with zero engine changes** (`tpu-jax-v6e1-31b-w4a16`). Both are
> different sizes and layer geometries from the E2B export the `k_norm` complaint is documented
> against, so this is corroboration rather than a same-checkpoint reproduction — but the format is
> demonstrably readable, which leaves the loader as the variable.

## The QAT checkpoint formats themselves

Properties of **Google's QAT artifacts**, independent of which engine reads them — established while
porting the 12B/26B/31B to the hand-rolled JAX engine (`tpu-jax-v6e1-31b-w4a16`,
`tpu-jax-v6e1-26b-q4_0`). They apply to any decoder, including the vLLM path above if `wNa16` is ever
implemented there.

**Note the stack boundary.** Everything above this section is vLLM + tpu_inference. Everything in
this section was measured on a *different* stack — the pure-JAX engine in `~/tpu-jax-*`. What carries
across is the on-disk format; nothing about performance does.

### Decoding w4a16: two traps that both yield negative SNR

1. **The int4 nibbles are BIASED, not two's complement.** Stored value is `value + 8`: 0 → −8, 8 → 0,
   15 → +7. Sign-extending as two's complement scrambles the weights.
2. **Packed words are `I32` and must never pass through float32.** A word packing eight nibbles
   routinely exceeds 2²⁴, which float32's mantissa cannot hold — the bit pattern is *silently rounded*
   and the nibbles are destroyed. Any reader that normalizes tensors to float32 needs a raw path.

Both were caught the same way: **SNR ≈ −8 dB**. A quantizer cannot produce error larger than signal,
so any "quantization error" above 100% is a decoder bug, never a model property. Keep that tripwire.

Aside: `safetensors` cannot decode bf16 on a bare JAX VM — `framework="np"` and `framework="flax"`
both raise `data type 'bfloat16' not understood`, and torch is usually not installed. The container is
trivial (8-byte header length, JSON header, raw buffer) and bf16 → f32 is a 16-bit left shift.

### Measured w4a16 error: 6.67%, and flat everywhere

Measured on the 31B by dequantizing `-qat-w4a16-ct` against `-qat-q4_0-unquantized`, which ships the
same QAT weights in half precision. Control: every tensor neither variant quantizes — all four
RMSNorms, `layer_scalar`, `final_norm`, all 1.4B parameters of `embed_tokens` — is **bit-identical**,
so the two are the same base model.

| | relative Frobenius error | SNR |
| :--- | ---: | ---: |
| every projection, layers 0 / 30 / 59 | **0.0667** | **23.52 dB** |

Across 20 (layer, projection) pairs: min 0.06663, max 0.06671 — a spread of **0.12%**.

> **Scale dynamic range does NOT predict quantization damage.** A proxy metric (p99.9/median of
> |scale|) ranked `v_proj` hardest at 4.22 and `up_proj` easiest at 2.41, and concluded late-layer
> attention was where extra bits would pay. Measured error is flat to 0.1% and `v_proj` is among the
> *lowest*. The proxy measured the spread of the **scales**, and the scales exist precisely to absorb
> that spread — with one bf16 scale per 32 input columns, a wide scale distribution means the
> mechanism is working, not struggling.

**Consequence: there is no cheap mixed-precision win on Gemma 4 weights.** Nothing is
disproportionately damaged by W4A16, so spending extra bits on a subset of projections buys
proportionally little.

### `-q4_0-unquantized` is QAT data in an unquantized container

The 26B A4B is **the only size with no `-w4a16-ct` release** — enumerated from the Hub 2026-07-31, so
do not assume the suffix set is uniform across sizes:

| size | `-w4a16-ct` | `-q4_0-unquantized` | `-q4_0-gguf` | mobile |
| :--- | :---: | :---: | :---: | :---: |
| E2B, E4B | ✅ | ✅ | ✅ | ✅ |
| 12B, 31B | ✅ | ✅ | ✅ | — |
| **26B A4B** | **❌** | ✅ | ✅ | — |

> **Two migrated reports name `gemma-4-26B-A4B-it-qat-w4a16-ct` anyway, and that is not a
> counter-example.** `gpu-vllm-l4-26b-w4a16` carries L4 grids from 2026-06-10 and 2026-07-12 whose
> `Model:` line is exactly that string — but as a **local mount path** (`/mnt/models/...`), not a Hub
> id, and a local directory is named whatever the operator called it. The Hub enumeration above stands.
> What the runs *do* establish, by physical bound, is that the weights were 4-bit: 51.61 GB of bf16
> cannot fit a 24 GB L4 and ~15.27 GB can. The likely history is a local repack by the route described
> immediately below, stored under an aspirational name.

**"Unquantized" describes the container, not the values.** Those weights already sit on a Q4_0 grid —
verified by range-reading the shards: all 256 sampled groups of 32 land exactly on a 4-bit grid, for
expert, attention, MLP, router and embedding tensors alike. **Group size 32 is measured, not assumed**
— group size 64 fails the same test. So 51.61 GB of BF16 repacks to 15.27 GB losslessly enough to fit
a 33.55 GB chip.

### `-q4_0-gguf` is the same QAT data, actually packed — and it is the only 4-bit artifact that loads anywhere

**Measured 2026-09-02** by range-reading the file off the Hub; nothing was downloaded whole.
`google/gemma-4-E2B-it-qat-q4_0-gguf` is ungated and ships two files:

| File | Bytes | sha256 |
| :--- | ---: | :--- |
| `gemma-4-E2B_q4_0-it.gguf` | 3,349,516,256 | `fa401b55…` |
| `gemma-4-E2B-it-mmproj.gguf` | 986,833,664 | `021059cc…` |

GGUF v3, `general.architecture = gemma4`, **541 tensors**, 49 KV pairs, and the full E2B shape is intact —
`attention.shared_kv_layers=20`, `sliding_window=512`, mixed `key_length=512` / `key_length_swa=256`,
`embedding_length_per_layer_input=256`. Dtype histogram: **Q4_0 ×275, F32 ×263, Q6_K ×2, F16 ×1**.

**It is the same QAT weights as `-q4_0-unquantized`, proven rather than assumed.** Four F32 norm tensors
read out of the GGUF are bit-identical to the bf16 tensors in the `-unquantized` repo:

| GGUF | safetensors | first values | mean |
| :--- | :--- | :--- | ---: |
| `blk.0.attn_norm.weight` | `layers.0.input_layernorm.weight` | 9.375, 7.9375, 10.6875 | +10.67993 |
| `output_norm.weight` | `model.norm.weight` | 13.4375, 8.75, 14.375 | +14.20042 |
| `blk.0.ffn_norm.weight` | `layers.0.pre_feedforward_layernorm.weight` | 21.75, 4.71875, 23.0 | +19.20642 |
| `blk.0.layer_output_scale.weight` | `layers.0.layer_scalar` | 0.02087402 | +0.02087 |

So the section above — QAT values already on a Q4_0 grid at group size 32, shipped in a bf16 container —
describes **this file's contents in their native packing**. 51.61 GB → 15.27 GB on the 26B is the same
relationship, done for you.

**Where the bytes go, and it is the PLE table again.** Summing the tensor table by role:

| Component | fp16 bytes | GGUF bytes | Streamed per decode step? |
| :--- | ---: | ---: | :--- |
| Transformer matmuls, 35 layers | 3.709 GB | **1.049 GB** (Q4_0) | yes |
| LM head / `token_embd` (tied) | 0.805 GB | **0.330 GB** (Q6_K) | yes |
| `per_layer_model_proj` | — | 0.028 GB (F16) | yes |
| **PLE table** `per_layer_token_embd` | 4.698 GB | **1.927 GB** (Q6_K) | **no — indexed lookup** |
| | | **3.334 GB total** (file 3.3495) | |

**Streamed drops 4.514 GB → 1.407 GB, a 3.2x cut.** Do not convert that into a throughput prediction: on
every GPU rig here decode at `B=1` is launch-bound, not bandwidth-bound, and the two measurements at the top
of this file (`ple_bits=4` → **0.0%**, `int8_lm_head` → **+2.3%**) are what a bandwidth cut actually buys.
**The win is residency**, which is what pays for batching.

### Which runtimes can load a GGUF at all

Verified 2026-09-02 against the installed versions named.

| Stack | Loads Google's Gemma 4 GGUF? |
| :--- | :--- |
| **vLLM 0.26.0** (CUDA or TPU) | **No.** No `gguf` module at all — see the correction in the route table above |
| **JAX** | **No.** No GGUF reader exists in the JAX ecosystem |
| **transformers 5.12.1** | Yes, `from_pretrained(gguf_file=…)` — but see the two defects below |
| **llama.cpp / Ollama** | Yes, natively — upstream `src/models/gemma4.cpp`, plus the four `mtmd` multimodal variants |

**`ggmlc` does not help.** It compiles PyTorch/JAX graphs *to* GGUF — an exporter, the opposite direction —
its text coverage stops at Gemma 3, and it writes Q4_0/Q8_0 rather than reading them.

#### transformers loads it, but dequantizes to fp32 and silently drops 35 tensors

Two independent defects, both verified offline:

1. **No memory or bandwidth win.** `modeling_gguf_pytorch_utils.py:791` calls `gguf.dequantize(...)`, which
   returns **float32**, and only then casts to `torch_dtype` per tensor. The transient is the problem:
   `per_layer_token_embd` is 8960 × 262144 = 2.349 B params, so that one dequantize allocates **9.395 GB of
   host fp32**. A 16 GiB box cannot survive it. You arrive at fp16 weights you could have loaded from
   safetensors.
2. **35 tensors are dropped without an error.** `model.layers.N.layer_scalar` is a bare `nn.Parameter`, so
   the generated map key is `blk.N.layer_output_scale` while the file names it
   `blk.N.layer_output_scale.weight`. Line 806 is `if name not in tensor_key_mapping: continue`. Building the
   real Gemma4 text model on `meta` and running the real `get_gguf_hf_weights_map` against the file's actual
   541 names gives **505 mapped, 36 unmapped: all 35 `layer_output_scale` plus `rope_freqs`**. `rope_freqs`
   is a harmless llama.cpp artifact; the 35 are not — layer 0's true value is 0.02087402, and the model would
   run on whatever `from_config` initialized.

**So transformers is a converter, not a serving path, for this file.**

### Two ways to destroy those weights while "just repacking" them

1. **`d = amax / 8` is the wrong step.** The textbook Q4_0 rule assumes each block's largest magnitude
   sits at level ±8; plenty of blocks peak lower. When they do, the derived step is a fraction of the
   true one, `round(x/d)` lands between grid points, and the block is requantized onto a grid that does
   not contain its own values — **4.9e-2 median error, and nothing raises.** The model loads and
   generates fluent text while being 5% wrong in every expert weight. Search for the level the peak
   actually occupies (m over 1..8) and refine by least squares: 93.1% of values then reconstruct
   exactly. Return a count of unplaceable groups and *raise* on any nonzero count rather than logging.
2. **Packing after the transpose.** W4A16 packs nibbles along the **last** axis, and the Q4_0 grid runs
   along `in`. A loader that transposes `[out, in] → [in, out]` before packing groups across `out`,
   where no grid exists — a real requantization dressed up as a repack. Pack in the loader, before the
   transpose.

Done correctly: 89–93% of values bit-identical, worst case ~1.6 BF16 ULP. That residue is **scale
precision, not level assignment** — Q4_0 carries an fp16 block scale and this format stores BF16,
three mantissa bits shorter. Refining the step moves zero levels.

### mxfp4 is MoE-only

`layers/jax/quantization/mxfp4.py` is `Mxfp4FusedMoEMethod` — it attaches to `JaxRoutedExperts` and works
on `w13_blocks` / `w2_blocks` / `gate_up_proj_scales`, i.e. gpt-oss-style **MoE expert weights**. Gemma 4
E2B is dense (`enable_moe_block=False`, `num_experts=None`), so there are no `JaxRoutedExperts` layers for
it to attach to. It also calls `dequantize_tensor_from_mxfp4_packed` in `process_weights_after_loading`,
so even where it does apply it unpacks to bf16 — consistent with there being no fp4 MXU anywhere yet.

## qwix is the only way in, and as of 2026-08-07 it does not get there

Everything below about how qwix is invoked is confirmed correct — the config routes end to end, the
rule parses exactly as written, and both code paths run. **Neither reaches an allocation.** Measured
on E2B / v5e-1, full write-up in
`tpu-vllm-v5e1-2b/benchmarks/runs/2026-08-07-qwix-int8-v5e1/REPORT.md`:

| path | how far it gets | failure |
| :--- | :--- | :--- |
| concrete (default) | loads bf16 weights, `hbm=[(8.97, 15.75)]Gb`, starts quantizing | `RESOURCE_EXHAUSTED: HLO temporaries (16.23G) exceeds available HBM (15.75G)` |
| `use_abstract_model: true` | `hbm=[(0.0, 15.75)]Gb` — no bf16 copy, as designed | `ValueError: no module or parameter named '…layers.0.mlp.down_proj.weight'`; the quantized `JaxLinear` exposes `set()` |

**The two failures are independent** — fixing either leaves the other. And the OOM lands on **E2B**,
the one size whose bf16 weights fit a v5e-1 with 5.5 GiB spare; the concrete path needs the bf16 model
and the quantization temporaries resident at once, so "it fits at bf16" does not imply "it can be
quantized in place". Both fail in 2.5–4 min, well before the ~738 s compile, so probing further
configurations is cheap.

**No weight-quantization footprint number exists for Gemma 4 on this stack.** Anything downstream —
int8 throughput, the int4-for-12B plan, E4B's ~131K KV tokens at int8 — is contingent on one of these
two being fixed. The `HARDWARE.md` case for int8 (2x bf16 in the MXU, the only compute win on
v5e/v6e) is untouched by this and is still the reason to want it.

### How it is invoked

`tpu_inference/models/jax/utils/qwix/` ships Google's QWIX library wired into the JAX path, with four
configs under `configs/`, all using `module_path: '.*'`:

| Config | Scheme |
| :--- | :--- |
| `int8_all_modules_w_only.yaml` | W8A16 — `weight_qtype: int8` |
| `int8_default.yaml` | W8A8 — adds `act_qtype: int8` |
| `fp8_all_modules_w_only.yaml` | `weight_qtype: float8_e4m3fn` |
| `fp8_default.yaml` | adds `act_qtype: float8_e4m3fn` |

**qwix applies PTQ in memory to the bf16 checkpoint** — no pre-quantized artifact required, which is what
lets it sidestep both blockers above.

Invocation (the YAMLs are just a serialization of `additional_config["quantization"]`):

```
--additional-config '{"quantization":{"qwix":{"rules":[{"module_path":".*","weight_qtype":"int8"}]}}}'
```

**Rig startup-script templates run through `str.format()`** — every literal `{`/`}` in that JSON breaks
the deploy. Escape as `{{`/`}}`, or add a `{qwix_config}` placeholder the way `{limit_mm_per_prompt}`
already works.

`qwix.QuantizationRule` exposes more than the shipped configs use:

| Field | Default | Targets |
| :--- | :--- | :--- |
| `weight_qtype` | None | weights — `int8`, **`int4`**, `float8_e4m3fn` |
| `act_qtype` | None | activations |
| `tile_size` | None | group-wise scales instead of per-tensor |
| `weight_calibration_method` | `absmax` | scale derivation |
| `act_calibration_method`, `act_static_scale`, `act_batch_axes` | | activation scaling |
| `module_path`, `op_names` | `.*` | regex per-module targeting |

### int4 is constructible and is the route for 12B

Verified in-container:

```python
QuantizationRule(module_path=".*", weight_qtype="int4", tile_size=128)
# -> rule ok: int4 tile 128        jnp.int4 exists
```

**`tile_size` is mandatory at 4 bits, not optional.** Per-tensor `absmax` gives 16 levels across an entire
weight tensor. This is the same failure mode as a fixed per-tensor scale on fp8 KV (below), one bit-width
worse. The four shipped configs are per-tensor over `.*` and are therefore **not** a starting point for
4-bit: you want group scales plus an `lm_head` / embedding exclusion via `module_path` (Gemma's vocab is
262,144, so the head is both large and quality-sensitive).

Constructing a rule is not evidence it works end-to-end — and int8, the simplest rule of all, was
constructed successfully and still did not boot (see the top of this section). Nothing about int4 has
been attempted on device. Verify per the rule at the bottom of this file.

### `use_abstract_model` is the critical path for anything that doesn't fit at bf16

`apply_qwix_quantization` quantizes *"the concrete model, which already has the weights loaded in"* —
impossible when bf16 weights exceed HBM. The alternative is gated by `apply_qwix_on_abstract_model`
(`qwix_utils.py:433`), reading `additional_config["quantization"]["qwix"]["use_abstract_model"]`
(default `False`), which quantizes the shape-only model so weights load straight into QArrays.

Its docstring marks that path **(Deprecated)**. On a 16 GB chip both E4B (14.9 GiB) and 12B (22.4 GiB)
depend on it.

**Verified 2026-08-07 on E2B / v5e-1: it does not function.** The memory behaviour is right — it
reports `hbm=[(0.0, 15.75)]Gb` before quantizing, against 8.97 GiB on the concrete path, so no bf16
copy is materialized. Weight loading then raises `ValueError: There is no module or parameter named
'model.language_model.layers.0.mlp.down_proj.weight'`, listing the available parameters of that
`JaxLinear` as `set()` — the quantized abstract module carries no parameter structure for the loader
to bind to. The design is sound and the implementation is broken downstream of it.

Testing on E2B was deliberate: bf16 fits there, so the failure isolates to the code path rather than
to memory pressure. It did — this is a `ValueError`, not an OOM.

## KV cache allocation: sliding windows are switched off for every Gemma 4 size

Not a quantization route, but it sits in the same budget and is the larger number, so it belongs
beside the section below. **tpu_inference allocates full-length KV for sliding-attention layers**,
even though those layers are masked to `sliding_window` tokens and can never read past it.

The trigger is a model having more than one head dim, in
`tpu_inference/runner/kv_cache_manager.py`'s `get_kv_cache_spec` (read from `main`, 2026-08-09):

```python
head_size_set = {common_utils.get_padded_head_dim(getattr(attn_module, "head_size", 0))
                 for attn_module in layers.values() if not isinstance(attn_module, MambaBase)}
disable_sliding_window = len(head_size_set) > 1
# TODO(yuyanpeng): enable sliding windows once mixed dims support
#   Currently, with sliding windows, there is shared_kv_cache_layers among each group.
```

`SlidingWindowSpec` is constructed and then discarded: `if disable_sliding_window:
attn_module.sliding_window = None`, collapsing everything to one full-attention group.

**Every Gemma 4 size trips this**, because `head_dim` 256 / `global_head_dim` 512 is a family-wide
split (`@MODELS.md` family table) — the same two-geometry fact that caused the 15-vs-18 KiB/token
error. Confirmed against a boot log: E2B on v5e-1 reports `Hybrid KV cache layout:
num_kv_cache_groups=1, num_kv_cache_tensors=15` with identical block counts on all 15 tensors.

**What it costs**, on E2B (12 sliding layers at 1,024 B/token windowed to 512, 3 full at 2,048 B/token):

| context | allocated | if windowed | forgone |
| ---: | ---: | ---: | ---: |
| 8,192 | 144.0 MiB/seq | 54.0 MiB/seq | 2.67x |
| 16,384 | 288.0 MiB/seq | 102.0 MiB/seq | 2.82x |
| 32,768 | 576.0 MiB/seq | 198.0 MiB/seq | 2.91x |

So **18 KiB/token is the operative figure and no flag reduces it** — `--disable-sliding-window` goes
the wrong way, and there is no switch for the reverse. This is worth more than every KV dtype below
combined; re-check it on each image bump, since it is gated on a named upstream TODO.

**Two things this does not mean.** Attention masking is unaffected — `attn_module` here comes from
vLLM's `static_forward_context` and is spec bookkeeping, not the JAX `Gemma4Attention`, which takes
`attention_chunk_size` from its own config; sliding layers still attend over exactly their window.
And the waste is *allocation*, not bandwidth: the ragged-paged attention kernel still reads only what
the mask admits.

## KV cache quantization

Only 5 of vLLM's 15 `--kv-cache-dtype` values work on this path. `_DTYPE_STR_ALIAS_TO_JAX_DTYPE`
(`tpu_inference/utils.py:36`) maps exactly four — `fp8`, `fp8_e4m3`, `fp8_e5m2`, `fp4` — and
`to_jax_dtype` sends **everything else** to `jnp.dtype(...)`:

```python
if isinstance(dtype, str) and (dict_dtype := _DTYPE_STR_ALIAS_TO_JAX_DTYPE.get(dtype, None)):
    return dict_dtype
return jnp.dtype(dtype)          # <- anything not in the four-entry table lands here
```

So the survivors are the four aliases plus whatever `jnp.dtype` accepts (`int8`, `bfloat16`,
`float16` — note it is **`jnp.dtype`, not numpy**, which is why `bfloat16` resolves at all).
`int8_per_token_head`, `turboquant_*`, `nvfp4`, `fp8_inc`, `fp8_ds_mla` raise
`TypeError: data type not understood` and **kill the server at boot** — all of them are in vLLM's
`CacheDType` literal, so passing CLI validation proves nothing here.

> **`auto` never reaches that function**, since `jnp.dtype("auto")` would raise and the server boots
> fine on `auto`. An explicit dtype therefore takes a **different code path** from `auto` — including
> for `bfloat16`, which resolves to the same dtype the model already uses. Prefer `--kv-cache-dtype
> auto` with `--dtype` pinned: identical bytes and block shape, without entering the branch where
> `gemma4.py`'s write-side KV quantization and its hardcoded `_k_scale`/`_v_scale = 1.0` live.

The enum itself moves: current `main` lists 17 values, against 15 on the pinned build. Re-count before
quoting a fraction.

- **Retracted 2026-08-09: plain `int8` is NOT passable, so it cannot silently corrupt.** This file
  previously called it "the most dangerous value in the list", reasoning that `to_jax_dtype` falls
  through to `jnp.dtype("int8")` while `gemma4.py:406-408` hardcodes `_q_scale`/`_k_scale`/`_v_scale`
  to `1.0`. That reasoning skipped a step: **vLLM's CLI enum is checked first, and does not contain
  `int8`** — the same failure mode already documented for `fp4` one bullet down. Measured on the
  running image: `vllm serve: error: argument --kv-cache-dtype: invalid choice: 'int8'`. The full
  accepted set on this build is **16 values**: `auto, bfloat16, float16, fp8, fp8_ds_mla, fp8_e4m3,
  fp8_e5m2, fp8_inc, fp8_per_token_head, int4_per_token_head, int8_per_token_head, nvfp4,
  turboquant_3bit_nc, turboquant_4bit_nc, turboquant_k3v4_nc, turboquant_k8v4`. Note `int8_per_token_head`
  *is* accepted and dies loudly at boot; `nvfp4_4over6` is in the source `Literal` but not the CLI enum.
  The scale-hardcoding in `gemma4.py` is real and would matter if a narrow integer dtype ever became
  passable — it is not currently reachable.
- **`fp4` is mapped but not passable** — the source comments `# NOTE: vLLM doesn't have this str dtype
  yet`, and it isn't in vLLM's CLI enum, so it fails validation before reaching tpu_inference.
- **`--calculate-kv-scales` is a no-op** for Gemma 4 — honored only in
  `layers/vllm/custom_ops/mla_attention.py`, the DeepSeek MLA path.

### Measured: fp8 KV changes nothing, and 4-bit KV probably won't either

Measured on v5e-1, full write-up in
`tpu-vllm-v5e1-2b/benchmarks/runs/2026-08-07-kv-quant-v5e1/REPORT.md`.

`--kv-cache-dtype fp8_e4m3` gives a **1.000x** capacity ratio — 321,376 tokens and 10,043 blocks in both
arms — because the KV layout is **word-aligned**: the block shape goes `(32,1,2,256)` -> `(32,1,4,256)` as
the element width halves, so the byte count never changes. All 8 throughput cells lost 1.8-5.6%; quality
was unchanged (8/9 byte-identical, 0/3 needles lost). **Do not set this flag.**

The mechanism is a tpu_inference layout property rather than a v5e one, so the same result should be
expected on other TPU generations — but that has not been measured, and 4 fp8 values filling a 32-bit word
is a size coincidence worth re-checking on a chip with different vector widths.

Extrapolating the pattern (2 bytes -> dim 2, 1 byte -> dim 4, both 32,768 B/block/layer), **4-bit KV
should land at dim 8 and the same byte count — a third 1.000x.** That is a prediction, not a measurement,
but after fp8 it is the default expectation.

## The verification rule

The fp8 KV flag was accepted at the CLI, echoed in `non-default args`, praised in an engine log line,
reported in `/metrics` as `cache_dtype="fp8_e4m3"`, and allocated a genuinely `float8_e4m3fn` tensor —
**five independent signals it had worked** — while delivering nothing.

**Verify quantization from the boot allocation log, never from the flag being accepted.**

- **KV:** compare `kv_cache_size_tokens` and `num_gpu_blocks` across arms against the ratio the element
  width predicts. Read the `Init kv-cache` line for the actual block shape and dtype.
- **Weights:** check whether `Memory statistics | total_hbm_used_gb` drops. E2B's bf16 figure is
  **8.97 GiB**; if the number does not move, nothing downstream matters and there is no point
  benchmarking. **If the line never prints, that is the answer too** — on both qwix arms the engine
  died before `tpu_worker.py:557`, and an absent allocation log is a cleaner negative than a
  suspicious number.
