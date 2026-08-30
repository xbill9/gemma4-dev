# What came from the fork, and what did not

Retargeted from **`gpu-pytorch-g5g-2b`** on 2026-08-29 — same runtime, different chip and
host architecture. Not forked from `gpu-jax-g6-2b`, whose directory this briefly occupied as
a byte-identical copy.

**This rig now has one measurement of its own**
(`benchmarks/runs/2026-08-29-first-serve-g6/`). Everything else below is labelled.

## Inherited and still true — properties of the model, checkpoint or host

- **Warm up at the shape you measure.** Nothing is compiled here, but cuBLAS autotune and
  allocator growth are per-shape. `sweep.py` warms every cell.
- **The kernel refuses to mmap the 10.2 GB checkpoint** without swap on a small host. That is
  a property of the checkpoint and the kernel, not the GPU, so it carries — but the *size
  names* do not (below).
- **`/v1/completions` is unreliable on `-it` models**, and a non-empty response is not a
  health check. `verify_model_health` reads the degenerate counter instead.
- **The two-stage bootstrap.** User data is capped at 16 KB; the payload ships over SSM.
- **The interpreter must be probed, not named.** Torch lives in the DLAMI's venv. CONFIRMED
  again here, and more sharply: Ubuntu 26.04's system Python is **3.14** while the DLAMI's
  torch venv is **3.13**.

## Inherited and now FALSE — Turing facts that do not survive the chip change

- **"float16 is the only real 16-bit path."** Ada has a native bfloat16 datapath.
  `resolve_compute_dtype` resolves `bfloat16` here; `is_bf16_supported()` is True. The guard
  stays because the failure is silent in the *other* direction.
- **"87% of decode is dtype tax."** That is the T4G. Conversion cost is not the constraint
  here; this rig runs at 71% of its bandwidth roofline, and the missing 29% is eager-mode
  kernel-launch overhead, a different problem.
- **"Upstream wheels omit sm_75, so torch must come from the DLAMI."** Upstream wheels carry
  `sm_89`. The DLAMI is now an install-time preference, not a requirement. **Do not
  re-derive the Turing rationale.**
- **"fp8 is refused."** Ada has an fp8 datapath. It is available and probably still not
  worth reaching for — KV is not the binding constraint.
- **Every host-RAM verdict by size name.** G6 has twice the RAM of G5g at each suffix.
  `g6.xlarge` is 16 GiB, not 8; `g6.2xlarge` is 32 GiB, not 16.
- **The instance topology table.** `g6.16xlarge` is single-GPU; G5g's 16xlarge was not.
  There is no `g6.metal`. Multi-GPU sizes are 12/24/48xlarge.
- **The `arm64` AMI requirement**, and with it the DLAMI name filter — the x86_64 images do
  not carry an arch in their names at all.

## Deliberately NOT carried across

- **`benchmarks/runs/` and `reports/`** — the G5g runs stayed in the G5g rig, so
  `benchmarks/rollup.py` cannot count another rig's results against this one. The
  `2026-08-28-first-serve-g6` JAX run was likewise left in `gpu-jax-g6-2b`.
- **The XLA compilation-cache S3 feature** (`JAX_CACHE_S3_URI`, the restore, the upload
  timer, `CompilationCacheTests`). **Removed, not disabled.** torch compiles nothing on this
  path, so it would have synced an empty prefix and uploaded one, reporting success forever.
- **`XLA_PYTHON_CLIENT_MEM_FRACTION` and `JAX_COMPILATION_CACHE_DIR`** from the rendered
  systemd `EnvironmentFile`. Inert under torch, and they read as configuring something.
- **The JAX `tests/test_engine.py`.** It imported `ports.gemma4.jax_e_model`, caught its own
  ImportError, and skipped every test — silently, forever. Replaced with a real test of
  `resolve_compute_dtype` that stubs the compute capability.

## Inherited debris found in the G5g PyTorch rig itself

Worth recording because these were *not* fixed there and the same fork will reproduce them:

- **`.codex/config.toml` was still the JAX rig's** — wrong server name, a `skills/` path that
  does not exist, and approval gates naming `*_g5g_*` tools that rig does not have. **A gate
  on a nonexistent tool name fails open silently.**
- **`server.py`'s module docstring** described "EC2 G6 (Graviton2 + NVIDIA T4G) … JAX path"
  and `get_help` rendered a full pure-JAX configuration table.
- **`make_report.py` hardcoded `NVIDIA T4G` / `AWS Graviton2` / 15.36 GB** into every report's
  hardware block — the exact misattribution the monorepo `CLAUDE.md` warns about.
- **`SKILL.md`** described "pure JAX" and a `ports/gemma4/` payload.
- **`sweep.py`'s prompt filler** described a Graviton2 host and a Turing GPU — and that text
  is the actual prompt content.

## Still inert, still present

`tpu.env` carries `QUANT_MODE`, `PLE_BITS`, `INT8_LM_HEAD` and `PREFILL_CHUNK_SIZE` because
it was forked wholesale. They name knobs of this repo's **JAX** port and there is nothing
here to forward them to. `_serve_argv` emits only `--model --host --port --seq`, and its
docstring says why at the call site. They surface as `quant_mode="fp16"` / `ple_bits="0"`
labels on `/metrics`: passengers, not descriptions.
