# Quantization on vLLM + TPU: what works, what doesn't, and what was measured

Runtime-specific notes for **this** rig — vLLM `0.26.1rc1.dev125+ga7a204cc6` on `vllm/vllm-tpu:nightly`
(image `sha256:2a4a1f82…`), tpu_inference JAX path, TPU v5e-1. Gathered 2026-08-07 alongside
`benchmarks/runs/2026-08-07-kv-quant-v5e1/`.

**Model characteristics live at the monorepo root in `@MODELS.md`** — layer structure, head shape, KV
cost per token, weight footprints. Those are properties of the checkpoint and identical whatever serves
it. This file covers only what depends on the engine build and the chip.

## Hardware constrains the choice before software does

v5e publishes peaks for **bf16 (197 TFLOPs) and Int8 (393 TOPs) only — no fp8**. Ironwood/v7 is the
first TPU generation with fp8 in the MXU. So on v5e:

- **int8 is the only low-precision format with a compute win** (2x bf16).
- **Every fp8 route is storage/bandwidth-only** — values widen back to bf16 before the matmul.
- 4-bit has no MXU path either; int4 weights buy footprint and bandwidth, then unpack for compute.

Which produces an awkward pairing worth remembering: the format the JAX path supports best (fp8) is the
one v5e cannot accelerate, and the one v5e accelerates (int8) is unreachable through compressed-tensors.

## Gemma 4 is JAX-path only, and that decides everything

Gemma 4 exists solely as a JAX implementation — `models/jax/gemma4.py`, `gemma4_mm.py`, `gemma4_mtp.py`,
with nothing under `models/vllm/`. Quant methods resolve through `layers/jax/quantization/`, so anything
in the torch path is unreachable **no matter what `tpu_platform.supported_quantization` advertises**.
That list is misleading read on its own.

| Route | Implemented in | Reachable for Gemma 4? |
| :--- | :--- | :--- |
| qwix PTQ (int8 / fp8, weight-only or W8A8) | `models/jax/utils/qwix/` | **yes** |
| compressed-tensors fp8 w8a8 | `layers/jax/quantization/compressed_tensors.py` | yes (needs a pre-quantized ckpt) |
| compressed-tensors **w4a16 / wNa16** | nowhere on the JAX path | **no — `NotImplementedError`** |
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
have no K projection (see `MODELS.md`). Upstream: [tpu-inference #3225](https://github.com/vllm-project/tpu-inference/issues/3225).
`tpu-pytorch-v5e1-12b` is currently pinned at one of these checkpoints and therefore does not load.

## qwix is the way around it

`tpu_inference/models/jax/utils/qwix/` ships Google's QWIX library wired into the JAX path, with four
configs under `configs/`, all using `module_path: '.*'`:

| Config | Scheme |
| :--- | :--- |
| `int8_all_modules_w_only.yaml` | W8A16 — `weight_qtype: int8` |
| `int8_default.yaml` | W8A8 — adds `act_qtype: int8` |
| `fp8_all_modules_w_only.yaml` | `weight_qtype: float8_e4m3fn` |
| `fp8_default.yaml` | adds `act_qtype: float8_e4m3fn` |

**qwix applies PTQ in memory to the bf16 checkpoint** — no pre-quantized artifact required, which is
what lets it sidestep both blockers above.

Invocation (the YAMLs are just a serialization of `additional_config["quantization"]`):

```
--additional-config '{"quantization":{"qwix":{"rules":[{"module_path":".*","weight_qtype":"int8"}]}}}'
```

**`startup_script_template.sh` runs through `str.format()`** — every literal `{`/`}` in that JSON breaks
the deploy. Escape as `{{`/`}}`, or add a `{qwix_config}` placeholder the way `{limit_mm_per_prompt}`
already works.

`qwix.QuantizationRule` exposes more than the shipped configs use:

| Field | Default | Targets |
| :--- | :--- | :--- |
| `weight_qtype` | None | weights — `int8`, `int4`, `float8_e4m3fn` |
| `act_qtype` | None | activations |
| `tile_size` | None | group-wise scales instead of per-tensor |
| `weight_calibration_method` | `absmax` | scale derivation |
| `act_calibration_method`, `act_static_scale`, `act_batch_axes` | | activation scaling |
| `module_path`, `op_names` | `.*` | regex per-module targeting |

Two levers the shipped configs leave on the table: **`tile_size`** (group scales — what makes int4 usable
at all; per-tensor `absmax` at a fixed scale is exactly what made fp8 KV lossy) and **`module_path`**
(all four quantize `.*`, including the 262k-vocab embedding and `lm_head` — excluding the head is the
standard first refinement).

### `use_abstract_model` is the critical path for E4B and 12B

`apply_qwix_quantization` quantizes *"the concrete model, which already has the weights loaded in"* —
impossible when bf16 weights exceed HBM, which is the case for both E4B (14.9 GiB) and 12B (22.4 GiB)
against 14.49 GiB usable. The alternative is gated by `apply_qwix_on_abstract_model`
(`qwix_utils.py:433`), reading `additional_config["quantization"]["qwix"]["use_abstract_model"]`
(default `False`), which quantizes the shape-only model so weights load straight into QArrays.

Its docstring marks that path **(Deprecated)**. Both larger models depend on it. **Verify it still
functions before planning around it** — and test on E2B, where bf16 does fit, so a failure isolates to
that code path rather than to memory pressure.

## Measured: fp8 KV cache does nothing on this build

Full write-up in `benchmarks/runs/2026-08-07-kv-quant-v5e1/REPORT.md`.

`--kv-cache-dtype fp8_e4m3` gives a **1.000x** capacity ratio — 321,376 tokens and 10,043 blocks in both
arms — because the KV layout is word-aligned: the block shape goes `(32,1,2,256)` -> `(32,1,4,256)` as
the element width halves, so the byte count never changes. All 8 throughput cells lost 1.8-5.6%; quality
was unchanged (8/9 byte-identical, 0/3 needles lost). **Do not set this flag on this build.**

Only 5 of vLLM's 15 `--kv-cache-dtype` values even work here — `_DTYPE_STR_ALIAS_TO_JAX_DTYPE`
(`tpu_inference/utils.py:36`) maps `fp8`, `fp8_e4m3`, `fp8_e5m2`, `fp4`, and numpy-parseable names like
`int8`. `int8_per_token_head`, `turboquant_*`, `nvfp4`, `fp8_inc`, `fp8_ds_mla` raise
`TypeError: data type not understood` and kill the server at boot. **`int8` resolves and is the most
dangerous value in the list** — at the hardcoded scale of 1.0 it rounds K/V to whole integers, and it
boots cleanly while doing so.

`--calculate-kv-scales` is a **no-op** for Gemma 4 — honored only in
`layers/vllm/custom_ops/mla_attention.py` (the DeepSeek MLA path). `gemma4.py:406-408` hardcodes
`_q_scale`/`_k_scale`/`_v_scale` to `1.0`.

## The verification rule this produced

The fp8 KV flag was accepted at the CLI, echoed in `non-default args`, praised in an engine log line,
reported in `/metrics` as `cache_dtype="fp8_e4m3"`, and allocated a genuinely `float8_e4m3fn` tensor —
**five independent signals it had worked** — while delivering nothing.

**Verify quantization from the boot allocation log, never from the flag being accepted.** For KV, compare
`kv_cache_size_tokens` and `num_gpu_blocks` across arms against the ratio the element width predicts. For
weights, check whether `Memory statistics | total_hbm_used_gb` drops from **8.97 GiB**. If the number
does not move, nothing downstream matters and there is no point benchmarking.
