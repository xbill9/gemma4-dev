# Quantization on vLLM + tpu_inference

What the **serving stack** supports for Gemma 4 — which routes are reachable, which are dead, and how to
enable the live ones. These are properties of vLLM and tpu_inference, not of any one chip, so they apply
equally to every `*-vllm-*` rig regardless of the hardware slot.

The companions: `HARDWARE.md` for which formats the silicon can actually *compute* in (that part **does**
vary by generation and changes the value of everything below), `MODELS.md` for checkpoint properties, and
each rig's `benchmarks/runs/` for what was measured where.

Verified against vLLM `0.26.1rc1.dev125+ga7a204cc6` / `vllm/vllm-tpu:nightly`
(`sha256:2a4a1f82…`) on 2026-08-07. The W4A16, 12B and KV-sharing findings were re-measured on
2026-09-25 against `sha256:19a1a052…` (vLLM `0.29.1rc1.dev468+g0b7f11a1e`), with and without three
unmerged upstream patches; the section "W4A16 on the JAX path" says which result needs which. The
stack moves; re-check before trusting a negative.

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
| **1650 Ti (SM 7.5, TU117)** | **`float32` or `bfloat16` — NOT fp16** | see below |
| inf2 | see `HARDWARE.md` | — |

> **THE TWO SM 7.5 ROWS DISAGREE, AND THAT IS THE POINT.** Compute capability 7.5 spans TU104
> (T4/T4G, **has** tensor cores) and TU116/TU117 (GTX 16-series, **has none** — Nvidia cut them from
> that die). MEASURED 2026-09-04 on a GTX 1650 Ti under torch 2.15.0.dev: fp16 matmul is pinned at
> **0.41 TFLOP/s against fp32's 1.68**, flat across N=1024/2048/4096 — a **4x penalty** on the very
> dtype this table recommends for "Turing". bf16 runs at parity with fp32 there. Full table and the
> caveats in `@HARDWARE.md`, "TU117 is Turing without tensor cores".
>
> **So do not resolve this table by compute capability alone.** Read the die, or read the device from
> the live driver as `ports/gemma4/jax_e_model.py` does — and note that reading `get_device_capability`
> is exactly what would get this wrong, because both parts answer `(7, 5)`.

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

On TPU the fused path pays at decode. MEASURED 2026-09-25 on v6e: `gmm_v2` W4A16 serves 1.14–1.41x the
output tokens/s of bf16 (E2B → 12B), where the unfused XLA path managed roughly 0.3–0.6x. The compute
dtype is the same bf16 either way; the gain is the bytes read per decode step. Details and the kernel's
limits in "W4A16 on the JAX path" below.

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

> **Except 12B, which on the stock image never reaches the JAX path at all. MEASURED 2026-09-25.**
> Gemma 4 12B ships as a different architecture: `google/gemma-4-12B-it` and its QAT builds declare
> `Gemma4UnifiedForConditionalGeneration` (`model_type: gemma4_unified`), and tpu_inference registers
> only `Gemma4ForConditionalGeneration` / `Gemma4ForCausalLM`. So 12B falls back to the vLLM PyTorch
> path even at bf16 — the boot log says `Falling back to vLLM-native Pytorch definition` — and every
> route in the table below is irrelevant to it. On that path its `-qat-w4a16-ct` build crashes at load
> with `AttributeError: 'NoneType' object has no attribute 'num_bits'` (weight-only compressed-tensors
> with `input_activations: null`,
> [tpu-inference #3539](https://github.com/vllm-project/tpu-inference/issues/3539)). Its decoder is the
> same as 31B's — identical `model.language_model.*` tensor names and layer structure; only the
> multimodal front end differs (`model.vision_embedder.*` and `model.embed_audio.*` instead of a
> `vision_tower`) — which is what lets
> [#3654](https://github.com/vllm-project/tpu-inference/pull/3654) serve it text-only on the JAX path.

| Route | Implemented in | Reachable for Gemma 4? |
| :--- | :--- | :--- |
| **qwix PTQ** — int8/int4/fp8, weight-only or W8A8 | `models/jax/utils/qwix/` | reachable, but **does not boot** — see below |
| compressed-tensors fp8 w8a8 | `layers/jax/quantization/compressed_tensors.py` | yes (needs a pre-quantized ckpt) |
| compressed-tensors **w4a16 / wNa16** | nowhere on the JAX path in the stock image | **no — `NotImplementedError`**; yes with [#3653](https://github.com/vllm-project/tpu-inference/pull/3653) (unmerged), see "W4A16 on the JAX path" |
| **mxfp4** (4-bit) | `layers/jax/quantization/mxfp4.py` | **no — MoE-only**, see below |
| compressed-tensors int8 w8a8, w4a8 fp8, w4a4 nvfp4 | `layers/vllm/.../schemes/` | no — torch path only |
| AWQ | `layers/vllm/quantization/awq.py` | no — torch path only |
| GGUF / q4_0 | absent from `QUANTIZATION_METHODS` — **and not a TPU-only gap**, see below | no |

**Measured 2026-09-24 on one v6e chip** (`vllm/vllm-tpu@sha256:19a1a052…`, vLLM `0.29.1rc1.dev468+g0b7f11a1e`, `jev-tpu/RESULTS.md`):

- **The fp8 w8a8 row serves a 26B-A4B on one chip.** `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic` (26.67 GiB on disk) booted in 616 s at `--max-model-len 2048 --max-num-seqs 16` and read the 3,880-record public suite at 76.0%, level with the 26B AWQ 4-bit build on an L4 (75.3%). It is the first 26B served by vLLM on one v6e chip in this monorepo. The 31B fp8 builds (30.98 GiB) exceed the ~28.7 GiB cap and were not tried.
- **The w4a16 row is still dead on this build** (the stock image; with three unmerged patches every size serves, see "W4A16 on the JAX path" below). `google/gemma-4-31B-it-qat-w4a16-ct` and `cyankiwi/gemma-4-31B-it-AWQ-4bit` — the cyankiwi "AWQ" exports are stored as compressed-tensors w4a16 — both fail in 120 s with `NotImplementedError: compressed-tensors scheme for layer 'model.language_model.layers.0.self_attn.q_proj' is not yet supported in the JAX path`.
- **Per-request `logprob_token_ids` is not implemented.** `tpu_inference` gathers only the top-k log-probabilities (`compute_and_gather_logprobs(..., max_logprobs)`); a completions request carrying `logprob_token_ids` returns HTTP 500 `list index out of range`, while token-id prompts and `logprobs` up to `--max-logprobs` work. Read label probabilities out of the top k.

The JAX compressed-tensors dispatcher handles `_is_fp8_w8a8`, then falls off the end:

```python
# TODO: w4a8 / wNa16 schemes need their own JAX methods (not yet ported).
raise NotImplementedError(...)
```

### W4A16 on the JAX path: three patches make every size serve on one v6e chip

**MEASURED 2026-09-25** on one v6e chip at `vllm/vllm-tpu@sha256:19a1a052…`, with three patches applied to
the image's `tpu_inference`. All three are upstream pull requests and **none is merged**, so every
result in this section needs them; the stock image still fails as the 2026-09-24 bullets above say.
Measurements, logs and the read's pre-registration are in `jev-tpu-31b/` (`results/2026-09-25-w4a16-*`,
`results/2026-09-25-followup-*`, `PREREGISTRATION.md`).

| Patch | Pull request | Needed by |
| :--- | :--- | :--- |
| KV-shared layers own no K/V parameters | [#3299](https://github.com/vllm-project/tpu-inference/pull/3299) (fixes #3225) | E2B, E4B QAT exports |
| JAX compressed-tensors W4A16 linear method on `gmm_v2` | [#3653](https://github.com/vllm-project/tpu-inference/pull/3653) | every `-qat-w4a16-ct` export |
| ~~`Gemma4UnifiedForConditionalGeneration` text-only on JAX~~ | ~~[#3654](https://github.com/vllm-project/tpu-inference/pull/3654)~~ closed 2026-09-26 | 12B serves with a flag instead (below) |
| JAX compressed-tensors W4A16 fused-MoE method | branch [`gemma4-w4a16-moe`](https://github.com/xbill9/tpu-inference/tree/gemma4-w4a16-moe), no PR yet | 26B (below) |

Scope of #3653: symmetric int4, `group` or `channel` strategy, `pack-quantized`, no activation
quantization — the format of every Google QAT release. Asymmetric, 8-bit, `actorder=group` and other
formats still raise. Weights are unpacked on the host, so HBM holds half a byte per weight.

**The kernel path is the whole speed story.** `sharded_quantized_matmul` picks its implementation from
the *rank of the scale*: a 3D `[in // group, 1, out]` scale routes to the `gmm_v2` Pallas kernel, which
dequantizes each tile in VMEM; a 2D scale routes to `xla_quantized_matmul`, which rebuilds the
full-precision weight in HBM on every call. Same weights, same numbers (the two agree to 2.4e-3), very
different cost:

| Per layer, 16 tokens, device time | bf16 | 2D scale (XLA) | 3D scale (`gmm_v2`) |
| :--- | ---: | ---: | ---: |
| 31B gate_up, 5376→43008 | 418 µs | 1654 µs | 168 µs |
| 12B down, 15360→3840 | 82 µs | 422 µs | 59 µs |
| E4B down, 10240→2560 | 46 µs | 192 µs | 48 µs |
| 31B o_proj, 8192→5376 | 30 µs | 298 µs | 43 µs |

Whole-model, the XLA path served at roughly 0.3–0.6x bf16 (single, EOS-terminated passes); the kernel
path, 16 concurrent requests of exactly 256 tokens, median of 3 passes:

| Model | HBM used | KV cache | Output tok/s | vs bf16 |
| :--- | ---: | ---: | ---: | ---: |
| E2B `-qat-w4a16-ct` | 7.17 GiB | 1,256,448 tokens | 4060 | 1.14x |
| E4B `-qat-w4a16-ct` | 10.15 GiB | 348,160 tokens | 2157 | 1.21x |
| 12B `-qat-w4a16-ct` | 9.46 GiB | 60,160 tokens | 984 | 1.41x |
| 31B `-qat-w4a16-ct` | 21.67 GiB | 8,320 tokens | 500 | bf16 does not fit |

E2B saves little HBM because the QAT export leaves its PLE table at bf16 (see the PLE section below).

- **The kernel has a fixed cost of about 40 µs per call**, so the smallest projections run faster in
  bf16 (the o_proj row). The gain grows with matrix size.
- **The kernel branch depends on the chip.** `gmm_v2` dequantizes in VMEM only when the quant group is
  narrower than the MXU (128 columns before v6e, 256 on v6e); a wider group takes its
  dequantize-after-matmul branch instead. #3653 applies that same test on the attached chip, so a
  group-128 checkpoint uses the kernel on v6e and the XLA path on v5e. Google's QAT exports use group 32.
- **Startup compile is about 2.7x the XLA path's** (warm-up pass 181 s → 472 s on E4B, 207 s → 570 s on
  12B), and **31B takes about 27 minutes to boot cold**. A persistent compile cache
  (`VLLM_XLA_CACHE_PATH` on a host volume) brings the same boot to 9 minutes (541 s), about half of it weight loading (279 s).
- **Untested:** tensor parallelism above 1, and every chip but v6e.

**The accuracy cost is 1–3 points and shrinks with size.** A label read on the 3,880-record public suite,
each W4A16 export paired per record with its bf16 checkpoint on the same patched image:

| Model | bf16 | W4A16 | Difference (95% bootstrap range) |
| :--- | ---: | ---: | :--- |
| E2B | 68.5% | 65.7% | −2.8 pts (−4.0 to −1.6) |
| E4B | 73.1% | 71.4% | −1.6 pts (−2.5 to −0.9) |
| 12B | 76.0% | 75.0% | −1.0 pts (−1.7 to −0.3) |
| 31B | — | 77.6% | bf16 does not fit one chip |

That read takes label probabilities from the top 32 log-probabilities (per-request
`logprob_token_ids` is unimplemented, above), and arms differ in how often every label comes back;
restricting to records where both arms returned every label moves the task-level differences by at most
1.5 points (E2B, AG News). The patches themselves leave bf16 unchanged: on the same suite, E2B and E4B on the patched
image differ from the stock image by 0.0 and +0.3 points, and 12B on the JAX path from the PyTorch path by
−0.3 (95% range −0.6 to +0.0).

**A quantized MoE on the stock JAX path loads as dense, by code reading.** For a MoE layer the stock
compressed-tensors dispatcher returns `Fp8FusedMoEMethod` for fp8 and otherwise falls back to
`UnquantizedFusedMoEMethod` — including for int4 experts
([tpu-inference #3542](https://github.com/vllm-project/tpu-inference/issues/3542)).
`cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` packs its experts that way (`experts.N.*.weight_packed`, I32),
so on the stock image it would read packed words as dense weights. It has not been served here. #3653
makes that case raise at load; fp8 MoE is unaffected (`RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic` still
serves, 669 output tok/s on the same benchmark).

### 12B needs no patch: `--hf_overrides` to `Gemma4ForCausalLM`

**MEASURED 2026-09-26**, one v6e chip, same image, #3299 and #3653 applied and #3654 **not** applied
(`jev-tpu-31b/results/2026-09-26-override-*`, `2026-09-26-q4_0-*`). 12B ships as
`Gemma4UnifiedForConditionalGeneration`, which nothing on the JAX path registers, so it falls back to the
PyTorch path. The JAX `Gemma4ForCausalLM` already reads `hf_config.text_config` and skips every tensor
whose name contains `vision` or `audio`, so the flag alone puts 12B on the JAX path:

```
--hf_overrides '{"architectures": ["Gemma4ForCausalLM"]}'
```

| 12B checkpoint | HBM used | KV blocks | Output tok/s | Suite |
| :--- | ---: | ---: | ---: | ---: |
| `-it` (bf16) | 22.18 GiB | 160 | 682 | 76.0% |
| `-it-qat-q4_0-unquantized` | 22.18 GiB | 160 | 682 | 75.7% (−0.2 vs bf16, −0.9 to +0.4) |
| `-it-qat-w4a16-ct` (needs #3653) | 9.46 GiB | 470 | 993 | 75.1% (−0.9 vs bf16, −1.6 to −0.2) |

- Memory, KV blocks and suite accuracy are identical to the #3654 build (bf16: 13 records flip each way),
  and −0.3 points against the PyTorch path, the figure #3654 reported. #3654 was closed in its favour.
- **`--limit-mm-per-prompt` is not needed** with the override: vLLM then treats the model as text-only.
- The boot log's tell for the path taken: the PyTorch fallback prints `Falling back to vLLM-native
  Pytorch definition` and uses 24.56 GiB for bf16 12B; the JAX path prints neither.
- A second W4A16 run on a second VM repeated the four tasks record for record and the suite within
  −0.1 points, so a read here carries across VMs; throughput moved about 2% between VMs.

### W4A16 experts: what the MoE path does to them

26B is the one size with no `-qat-w4a16-ct`, and `-q4_0-unquantized` does not fit one chip (48.07 GiB).
Serving its QAT weights at TP=1 takes a repack (`jev-tpu-31b/repack_q4_0.py`, see `MODELS.md` §26B)
and a MoE method that #3653 does not have. Read from `tpu_inference` at #3653's base (2026-09-26); each
of these fails quietly or changes the numbers:

- **`process_quantized_moe_weights` requantizes by default.** It dequantizes and requantizes into the
  source dtype with `requant_block_size=None`, which is per-channel. For group-32 int4 experts that
  throws away the QAT grid. A W4A16 method must call `process_moe_weights` and `shard_moe_weights`
  itself, as the mxfp4 method does.
- **`quantize_tensor` scales a group by `abs_max / 7`.** Q4_0 grids run −8…+7 with the peak usually at
  −8, so requantizing Q4_0 data through it moves nearly every group off its grid.
- **The fused MoE kernel wants blocks that are multiples of 256** (`fused_moe/v1/kernel.py`,
  `subc_quant_w1_sz % 256`). Group 32 must use the GMM backends (`GMM_TP`/`GMM_EP`), whose `gmm_v2`
  takes `[E, in // 32, 1, out]` scales.
- **`gmm_v2` quantizes the activation only on its dequantize-after-matmul branch**, taken when the group
  is at least the MXU width. Group 32 dequantizes before the matmul and keeps activations bf16; a
  channelwise or wide-group MoE would silently run as W4A8.
- **The intermediate padding is harmless.** GMM_TP pads `w13`'s intermediate 704 → 768; `moe_gmm_local`
  trims the first matmul's output back to `w2`'s own 704 before the second, so `w2` and its 22 scale
  groups keep the checkpoint shape.
- **`shard_moe_weights` cannot take host arrays under the TPU mesh.** It puts with a layout, which under
  the TPU context mesh becomes a jitted reshard: `ValueError: Received incompatible devices for jitted
  computation ... on platform CPU and jit's context mesh ... on platform TPU`. Copy with
  `shard_moe_weights_to_tpu(..., source_mesh=cpu_mesh())` first, as the fp8 path does. Construct and
  test the layer under `jax.set_mesh(mesh)`: an old-style `with Mesh(...)` neither builds a real
  `JaxRoutedExperts` nor reproduces this failure.
- The MoE path widens scales to float32 on the chip, so expert scales stored at better than bf16
  precision would survive to the kernel. #3653's dense path loads `weight_scale` as bf16 regardless.

**MEASURED 2026-09-26: Google's QAT 26B serves on one v6e chip.** The `gemma4-w4a16-moe` branch (on
#3653) passes 36 of 36 unit tests on the chip, including a forward test through a real
`JaxRoutedExperts` against a hand-written reference at the 26B's expert shape (2816 / 704), and serves
the repacked checkpoint (`jev-tpu-31b/results/2026-09-26-moe2-*`; FP8 from `2026-09-26-moe` and
`../jev-tpu/results/2026-09-24-v6e1-26b-fp8`):

| 26B-A4B on one v6e chip | HBM used | KV cache | Output tok/s | Suite |
| :--- | ---: | ---: | ---: | ---: |
| `RedHatAI/...-FP8-dynamic` | 27.99 GiB | 27 blocks, 3,456 tokens | 668 | 76.0% |
| QAT W4A16 repack (this route) | 17.43 GiB | 421 blocks, 53,888 tokens | 1,283 | 75.3% |

- **15.6x the KV cache and 1.92x the throughput** (16 concurrent requests of exactly 256 tokens,
  median of 3 passes; FP8 measured the same day on a second VM, and throughput moved ~2% between VMs).
- **Accuracy, paired per record against FP8:** suite −0.7 points (95% range −1.5 to +0.1; 133 records
  right→wrong, 106 wrong→right); the four tasks within ±1.7 points, every range spanning zero. FP8 is
  itself a quantization of bf16, and no bf16 26B fits one chip, so there is no bf16 reference here.
- Boot 796 s cold (334 s of it compilation); the saved compile cache is
  `xla-cache/vllm-tpu-19a1a052-patched-moe`.
- The KV cost is **120 KiB/token** at fp8 (8 heads × K,V × 256 × 30 layers, windows off), from the
  allocation shape `(num_blocks, (128, 8, 2, 256))` over 30 layers.
- vLLM gave KV 6.17 GiB of the 11.32 GiB left after weights; the rest is its activation reserve.

**The same checkpoint on stock vLLM CUDA, unpatched** (MEASURED 2026-09-26, one NVIDIA L4 on
`g2-standard-8`, `vllm/vllm-openai@sha256:8a69ffad…`, vLLM 0.30.0; `jev-tpu-31b/results/2026-09-26-gpu-l4-*`,
runner `jev-tpu-31b/gpu/run_gpu.sh`, same serve flags). It loads as it stands: vLLM picks
`CompressedTensorsWNA16MoEMethod` with `MarlinExperts` for the experts and `MarlinLinearKernel` for the
dense layers. Model loading takes 14.8 GiB; the KV cache holds 17,990 tokens; boot 255 s; 394 output
tok/s on the same 16 × 256 load. Paired per record with the TPU W4A16 run, the four tasks agree within
one example each; the suite reads **+0.7 points on GPU** (76.0% against 75.3%, 95% range +0.3 to +1.1;
16 records right→wrong, 42 wrong→right). Both runs set `kv_cache_dtype=auto`, which is **fp8_e5m2 on
v6e and bf16 on CUDA**, so the KV cache is one difference between them and the int4 kernels
(`gmm_v2` against Marlin) the other.

> **The GGUF row is a property of vLLM itself, not of the TPU platform. Verified 2026-09-02** against a
> stock **vLLM 0.26.0 CUDA** install: `grep -ril gguf` over the entire installed package returns **two**
> files, both incidental (`lora/layers/utils.py`, `models/qwen2_moe.py`); there is no `gguf.py` under
> `model_executor/layers/quantization/`; and `QUANTIZATION_METHODS` lists 31 entries with no `gguf` among
> them. So the original finding — recorded against `tpu_platform.py`'s `supported_quantization` — understated
> its own scope. **No vLLM rig in this monorepo can load a GGUF, on any platform slot.** Do not "check the
> CUDA build" expecting a different answer; it has been checked.
>
> **RE-VERIFIED 2026-09-04 on vLLM 0.28.0**, a live install rather than a survey — two releases later
> and the answer is unchanged: no `gguf.py` under `layers/quantization/`, `QUANTIZATION_METHODS` has
> **30** entries and none is `gguf`, and `LoadFormats` offers no gguf option. Three incidental
> mentions remain in the package, and **one of them is the strongest evidence yet**:
>
> ```
> models/exaone_moe.py:225   if quant_config is not None and quant_config.get_name() == "gguf":
> models/qwen2_moe.py:497    # GGUF: make sure that shared_expert_gate is a 2D tensor.
> lora/layers/utils.py:69    # MoE GPTQ/AWQ/GGUF
> ```
>
> That first line is a live branch guarded on a name **no quant config in this build can return**.
> This is not a feature that was never added — it is one that was removed, leaving a dangling check.
> So the right expectation is that it stays gone, not that a later release restores it.
>
> **And GGUF would not have rescued the 4 GiB card anyway, though it was the best remaining shot.**
> The `VocabParallelEmbedding` that OOMs at construction takes a `quant_config`, so a GGUF config
> that quantized embeddings would have allocated the PLE at Q6_K — 1.93 GB in llama.cpp's file
> against the 4.38 GiB fp16 the w4a16 path asked for. That is the one mechanism that could have
> shrunk the tensor *at construction*, which is where the failure is. It does not exist here.

### vLLM CANNOT offload Gemma 4's PLE, and that closes the 4 GiB GPU route

**MEASURED 2026-09-04**, vLLM 0.28.0 + torch 2.13.0+cu130 on a GTX 1650 Ti (4096 MiB), serving
`google/gemma-4-E2B-it-qat-w4a16-ct` with the Turing Triton clamp applied:

```
--cpu-offload-gb 6 --cpu-offload-params embed_tokens_per_layer vision_tower audio_tower
-> torch.OutOfMemoryError: Tried to allocate 4.38 GiB.
   GPU 0 has a total capacity of 3.64 GiB of which 1.87 GiB is free.
```

**4.38 GiB is the PLE exactly**: 262,144 vocab x (256 x 35 layers = 8,960) x 2 B = 4,697,620,480 B.
The flag was accepted and did nothing, and the traceback says why — the allocation happens at
**construction**, not at weight loading:

```
gemma4.py:979                   self.embed_tokens_per_layer = VocabParallelEmbedding(...)
vocab_parallel_embedding.py:325 __init__
vocab_parallel_embedding.py:50  create_weights
```

`cpu_offload_gb` / `cpu_offload_params` act during weight *loading*. `Gemma4Model.__init__` has
already put the tensor on the device by then, so **no offload flag can reach it.** This is a
construction-order property of the implementation, not a budget more offload would fix.

**Consequence: vLLM cannot serve E2B on a 4 GiB card at all, and quantization does not change it**,
because the QAT w4a16 export leaves the PLE at bf16 — one tensor, 56.5% of that checkpoint (see
`@MODELS.md`). The contrast is the whole point: llama.cpp creates the same tensor with
`TENSOR_READ_LAZY` and gathers rows out of the mmap so it never reaches VRAM, and
`local-llamacpp-1650ti-2b-q4_0` serves the same weights on the same card in **1612 MiB**.

**Two things this run settled in vLLM's favour**, both against what this repo assumed:

- **bf16 on SM 7.5 is a warning and an automatic cast, not a hard failure.** Logged verbatim: "Your
  device ... doesn't support torch.bfloat16. Falling back to torch.float16 for compatibility."
  `@HARDWARE.md`'s T4G section calls `--dtype bfloat16` "a hard failure, not a slow path"; on the
  CUDA path it downgrades itself instead.
- **The Turing shared-memory clamp works against a stock PyPI wheel.**
  `gpu-vllm-g4dn-2b/patch_triton_turing.py` applied cleanly to vLLM 0.28.0's site-packages copy
  (inserting after the last tile assignment, before all three reads), `TRITON_ATTN` was forced, and
  the run reached model construction without an `OutOfResources`. No from-source build on x86_64.

**Cost of learning this: an 8.32 GB download and a torch downgrade** — vLLM 0.28.0 pins
`torch==2.13.0` exactly, so it and a torch nightly are mutually exclusive.

`wNa16` is w4a16 — exactly the format of Google's QAT releases
(`google/gemma-4-{E2B,12B}-it-qat-w4a16-ct`). A **second, independent** failure hits the QAT exports:
`k_norm.weight` "missing" for layers 15-34. Upstream:
[tpu-inference #3225](https://github.com/vllm-project/tpu-inference/issues/3225).
`tpu-pytorch-v5e1-12b` is currently pinned at `gemma-4-12B-it-qat-w4a16-ct` and does not load.

> **Which export shows which failure — settled 2026-09-25.** The devto forensics tabulate them
> against **E2B**, and there they are *different* checkpoints: `-qat-w4a16-ct` dies on the int4
> compressed-tensors scheme being unimplemented for `per_layer_model_projection`, while
> `-qat-q4_0-unquantized` is the one that dies on `k_norm`. Read from the safetensors headers, the
> E2B `-qat-w4a16-ct` export ships `k_proj`, `v_proj` and `k_norm` for layers 0–14 only (`q_proj` for
> all 35), so it carries the `k_norm` gap too and needs both #3653 and #3299 to load; with both it
> serves. **12B and 31B never hit `k_norm`**: both declare `num_kv_shared_layers: 0`. 12B's failure on
> the stock image is the PyTorch-path fallback above; 31B's is the `wNa16` `NotImplementedError`.

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
> the tensors. #3225 proposes skipping K/V-side parameters for KV-shared layers;
> [#3299](https://github.com/vllm-project/tpu-inference/pull/3299) implements it (approved
> 2026-07-30, rebased onto `main` 2026-09-25, **unmerged**: it has not yet received the `ready` label
> this repo requires before CI runs). With it, bf16 E2B/E4B read the same on the 3,880-record suite
> (0.0 / +0.3 points against the stock image) and the E2B/E4B QAT exports load.
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
`tpu-jax-v6e1-26b-q4_0`). They apply to any decoder, including the vLLM path above — whose `wNa16`
method ([#3653](https://github.com/vllm-project/tpu-inference/pull/3653), unmerged) avoids both decoding
traps below: it unpacks with `u32_unpack_i4` (biased nibbles) and refuses packed words that arrive as a
float dtype.

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

**A Q4_0 GGUF converts to compressed-tensors W4A16 directly** (analysis, 2026-09-26; not built). A Q4_0
block is 32 weights, a 4-bit `q` each and one fp16 `d`, weight `d × (q − 8)`; compressed-tensors
symmetric int4 group 32 is `scale × level`, level −8…7. So `scale = d`, `level = q − 8`, with no step
to recover. What it would buy over repacking `-q4_0-unquantized` is only the exact fp16 step, and only
where the chip keeps it: the MoE path widens expert scales to f32, #3653's dense path rounds every scale
to bf16. The levels are the same data either way. The costs are a GGUF reader, llama.cpp's tensor
names, and the tensors that are not Q4_0 (E2B's GGUF has `Q6_K` ×2, `F16` ×1 and `F32` ×263 beside
275 `Q4_0`; the 26B's mix is unread). Storing fp16 scales in the `-unquantized` repack gets most of the
way (97% of values bit-identical against 89–93% at bf16, per `~/tpu-jax-26b/ports/gemma4/jax_q4_0.py`).

### Which runtimes can load a GGUF at all

Verified 2026-09-02 against the installed versions named.

| Stack | Loads Google's Gemma 4 GGUF? |
| :--- | :--- |
| **vLLM 0.26.0** (CUDA or TPU) | **No.** No `gguf` module at all — see the correction in the route table above |
| **vLLM 0.29.1rc1** (ROCm, gfx942) | **No, and it got further away.** Still no `gguf` module, `'gguf'` is absent from the global `QUANTIZATION_METHODS` registry, and **no ggml/gguf symbols are compiled into `vllm._custom_ops`**. Verified 2026-09-16 on MI300X |
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

**Confirmed against silicon 2026-09-16, and the "anywhere yet" now has a boundary.** On MI300X
(gfx942, CDNA 3) torch refuses outright — `Block-wise scaling for Float8_e8m0fnu is only supported
on gfx950,gfx1250` — and `Float4_e2m1fn_x2` cannot even be cast to. vLLM gates the same way:
`supports_mx()` is `any(gfx in _GCN_ARCH for gfx in ["gfx95", "gfx1250"])`, False on gfx942, so it
falls back to `EmulationMxfp4LinearKernel`, whose `apply_weights` dequantizes the weights to bf16
**every forward pass** and runs `F.linear` in high precision. Same unpack-to-bf16 shape as the JAX
path above, reached by a different route. **fp4 as arithmetic arrives with CDNA 4 (gfx950) /
Blackwell**; on everything this monorepo currently touches it is storage or emulation. See the
MI300X section of `HARDWARE.md`.

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
