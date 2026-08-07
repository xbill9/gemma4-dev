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

## Gemma 4 is JAX-path only, and that decides everything

Gemma 4 exists solely as a JAX implementation — `models/jax/gemma4.py`, `gemma4_mm.py`, `gemma4_mtp.py`,
with nothing under `models/vllm/`. Quant methods resolve through `layers/jax/quantization/`, so anything
in the torch path is unreachable **no matter what `tpu_platform.supported_quantization` advertises**.
That list is actively misleading read on its own.

| Route | Implemented in | Reachable for Gemma 4? |
| :--- | :--- | :--- |
| **qwix PTQ** — int8/int4/fp8, weight-only or W8A8 | `models/jax/utils/qwix/` | **yes — the live route** |
| compressed-tensors fp8 w8a8 | `layers/jax/quantization/compressed_tensors.py` | yes (needs a pre-quantized ckpt) |
| compressed-tensors **w4a16 / wNa16** | nowhere on the JAX path | **no — `NotImplementedError`** |
| **mxfp4** (4-bit) | `layers/jax/quantization/mxfp4.py` | **no — MoE-only**, see below |
| compressed-tensors int8 w8a8, w4a8 fp8, w4a4 nvfp4 | `layers/vllm/.../schemes/` | no — torch path only |
| AWQ | `layers/vllm/quantization/awq.py` | no — torch path only |
| GGUF / q4_0 | absent from `QUANTIZATION_METHODS` in this build | no |

The JAX compressed-tensors dispatcher handles `_is_fp8_w8a8`, then falls off the end:

```python
# TODO: w4a8 / wNa16 schemes need their own JAX methods (not yet ported).
raise NotImplementedError(...)
```

`wNa16` is w4a16 — exactly the format of Google's QAT releases
(`google/gemma-4-{E2B,12B}-it-qat-w4a16-ct`). Those checkpoints fail a **second, independent** way too:
`k_norm.weight` "missing" for layers 15-34, which are precisely the KV-shared layers that legitimately
have no K projection (see `MODELS.md`). Upstream:
[tpu-inference #3225](https://github.com/vllm-project/tpu-inference/issues/3225).
`tpu-pytorch-v5e1-12b` is currently pinned at one of these checkpoints and therefore does not load.

### mxfp4 is MoE-only

`layers/jax/quantization/mxfp4.py` is `Mxfp4FusedMoEMethod` — it attaches to `JaxRoutedExperts` and works
on `w13_blocks` / `w2_blocks` / `gate_up_proj_scales`, i.e. gpt-oss-style **MoE expert weights**. Gemma 4
E2B is dense (`enable_moe_block=False`, `num_experts=None`), so there are no `JaxRoutedExperts` layers for
it to attach to. It also calls `dequantize_tensor_from_mxfp4_packed` in `process_weights_after_loading`,
so even where it does apply it unpacks to bf16 — consistent with there being no fp4 MXU anywhere yet.

## qwix is the way in

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

Constructing a rule is not evidence it works end-to-end. Verify per the rule at the bottom of this file.

### `use_abstract_model` is the critical path for anything that doesn't fit at bf16

`apply_qwix_quantization` quantizes *"the concrete model, which already has the weights loaded in"* —
impossible when bf16 weights exceed HBM. The alternative is gated by `apply_qwix_on_abstract_model`
(`qwix_utils.py:433`), reading `additional_config["quantization"]["qwix"]["use_abstract_model"]`
(default `False`), which quantizes the shape-only model so weights load straight into QArrays.

Its docstring marks that path **(Deprecated)**. On a 16 GB chip both E4B (14.9 GiB) and 12B (22.4 GiB)
depend on it. **Verify it still functions before planning around it** — and test on E2B, where bf16 does
fit, so a failure isolates to that code path rather than to memory pressure.

## KV cache quantization

Only 5 of vLLM's 15 `--kv-cache-dtype` values work on this path. `_DTYPE_STR_ALIAS_TO_JAX_DTYPE`
(`tpu_inference/utils.py:36`) maps `fp8`, `fp8_e4m3`, `fp8_e5m2`, `fp4`, plus numpy-parseable names like
`int8`. `int8_per_token_head`, `turboquant_*`, `nvfp4`, `fp8_inc`, `fp8_ds_mla` raise
`TypeError: data type not understood` and **kill the server at boot**.

- **`int8` resolves and is the most dangerous value in the list.** `gemma4.py:406-408` hardcodes
  `_q_scale`/`_k_scale`/`_v_scale` to `1.0`, so int8 rounds K/V to whole integers — and it boots cleanly
  while doing so.
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
  benchmarking.
