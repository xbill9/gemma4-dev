# qwix int8 weight quantization on E2B, v5e-1 — does not boot

**Result: no HBM measurement exists, because the engine never reached allocation.** Both qwix code
paths fail during model load, for two unrelated reasons. This is a negative result about the
*stack*, not about int8 as a format.

- Date: 2026-08-07
- Host: `tpu-2B-v5e1-devops-agent`, `v5litepod-1`, us-west4-a, `aisprint-491218`
- Image, pinned by ID: `sha256:2a4a1f82793f748e02af54d77a62e590d34d2c9c68e833a8bb00d26a878a684c`
  (`vllm/vllm-tpu:nightly`), vLLM `0.26.1rc1.dev125+ga7a204cc6`
- Model: `google/gemma-4-E2B-it`, serving flags identical to the live baseline
  (`--max-model-len 16384`, `--max_num_batched_tokens 4096`, TP=1)
- Script: `swap_qwix.sh` (this directory) — same stop-and-rename discipline as the KV-quant run

## Baseline being compared against

| | value |
| :--- | ---: |
| `total_hbm_used_gb` | **8.97 GiB** |
| `total_hbm_limit_cap_gb` | 14.49 GiB |
| `total_hbm_avail_gb` | 5.52 GiB |
| `GPU KV cache size` | 321,376 tokens |

## Arm 1 — concrete model (the default): HBM OOM

```
--additional-config '{"quantization":{"qwix":{"rules":[{"module_path":".*","weight_qtype":"int8"}]}}}'
```

The flag is accepted and reaches tpu_inference intact — it is echoed in `non-default args`, and
`qwix_utils.py:498` logs `Overwriting default Qwix quantization config with user provided
quantization config`. The rule is parsed exactly as constructed:

```
Qwix rules: [QuantizationRule(module_path='.*', op_names=(), weight_qtype='int8', act_qtype=None,
             tile_size=None, act_static_scale=None, weight_calibration_method='absmax', ...)]
Memory usage before applying quantization of params: hbm=[(8.97, 15.75)]Gb
Applying Qwix quantization on concrete model
```

Then, ~9 s later:

```
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Ran out of memory on HBM,
the total memory required for HLO temporaries (16.23G) exceeds available HBM (15.75G).
```

Raised from `model_loader.py:305 _get_nnx_model` → `create_jit_model` → flax nnx jit.

**The 8.97 GiB in that log line is the point.** The concrete path quantizes weights that are
already resident, so it needs the bf16 model *and* the quantization temporaries at once. Per
`HARDWARE.md`, XLA compares temporaries alone against the whole chip without subtracting resident
weights, so the true requirement is worse than 16.23 G vs 15.75 G suggests.

**This happens on E2B — the one Gemma 4 size whose bf16 weights fit a v5e-1 with 5.5 GiB to
spare.** `QUANTIZATION.md` predicted the concrete path would fail for E4B and 12B, which cannot
load at bf16 at all. It also fails for the model that can.

## Arm 2 — `use_abstract_model: true`: weight loading raises

```
--additional-config '{"quantization":{"qwix":{"rules":[...],"use_abstract_model":true}}}'
```

The flag routes correctly and the mechanism does what it claims — note the resident figure:

```
Applying Qwix quantization on abstract model
Memory usage before applying quantization of params: hbm=[(0.0, 15.75)]Gb
```

**0.0 GB, against 8.97 GB on the concrete path.** No bf16 copy is materialized, which is exactly
the property that would make this the route for anything that does not fit. Then load fails:

```
ValueError: There is no module or parameter named
'model.language_model.layers.0.mlp.down_proj.weight' in Gemma4ForConditionalGeneration.
The available parameters belonging to model.language_model.layers.0.mlp.down_proj (JaxLinear)
are: set()
```

The quantized abstract `JaxLinear` exposes **no parameters at all** (`set()`), while the weight
loader still addresses `.weight`. Not an OOM and not a rule problem — the abstract module is not
carrying the parameter structure the loader expects.

`QUANTIZATION.md` flagged this path as marked **(Deprecated)** in its own docstring and said to
verify it before planning around it. Verified: on this build it does not work.

## What this does and does not establish

**Does:**

- int8 weight quantization is **not reachable** for Gemma 4 on this vLLM/tpu_inference build,
  by either qwix path, on a v5e-1.
- The failures are independent. Fixing the OOM would still leave arm 2's loader mismatch, and
  vice versa — they are not two symptoms of one cause.
- The `use_abstract_model` design is sound (0.0 GB resident before quantization); its
  implementation is broken downstream of that.

**Does not:**

- Say anything about int8's *value*. No weights were quantized, so no footprint, throughput or
  quality number exists. The `HARDWARE.md` case for int8 — the only low-precision format with an
  MXU compute win on v5e/v6e, at 2x bf16 — is untouched by this and still the reason to want it.
- Rule out other qwix configurations. Only `module_path='.*'`, `weight_qtype='int8'`,
  no `tile_size` was tried, on E2B, on one image.
- Establish that a lower `--gpu-memory-utilization` cannot rescue arm 1. It probably cannot — the
  OOM is on jit temporaries measured against the whole chip, not against the KV pool — but that
  was not tested.

## Verification discipline

Per the rule at the bottom of `QUANTIZATION.md`, the intended check was whether
`total_hbm_used_gb` drops from 8.97 GiB. It never printed: the engine died before
`tpu_worker.py:557`. **A missing allocation line is itself the answer** — there is no partial
success to interpret here, and no benchmark worth running.

Both failures were fast (~2.5 min and ~4 min), well before the ~738 s compile, so retrying other
qwix configurations is cheap.

## Logs

- `logs/arm2-abstract-int8.log` — the full container log for arm 2.
- `logs/arm1-concrete-int8-excerpt.log` — **an excerpt only.** Arm 1's container was
  `docker rm -f`'d when arm 2 started under the same name, so its full log is gone. Reproduce with
  `./swap_qwix.sh forward int8` (~2.5 min to the OOM) if it is needed in full; a future arm should
  take a distinct container name per configuration.

## Rig state

Restored to the bf16 baseline. The original container was never deleted — stopped, renamed
`vllm-gemma4-bf16`, and started back as `vllm-gemma4`, so its JAX compile cache made the revert
warm. The failed arm remains as the exited `vllm-gemma4-qwix` for log inspection.
