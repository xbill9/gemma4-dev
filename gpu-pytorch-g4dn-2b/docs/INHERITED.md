# What came from where, and what did not

This rig was forked **twice over** and the two forks changed different axes, so provenance
here has two directions rather than one. **No measurement in this rig is its own.**

```
gpu-jax-g5g-2b  --(runtime: JAX -> PyTorch, 2026-08-28)-->  gpu-pytorch-g5g-2b
                                                                    |
                                              (host: aarch64 -> x86_64, 2026-08-29)
                                                                    v
                                                            gpu-pytorch-g4dn-2b
```

So an inherited claim has to be checked against **which axis it belongs to**:

- A claim about the **GPU** carries down both arrows. T4 and T4G are both Turing SM 7.5.
- A claim about the **host** dies at the second arrow. Graviton2, aarch64, ARM64 DLAMIs,
  aarch64 wheel availability — none of it is about this rig.
- A claim about the **runtime** died at the first arrow. Pallas, XLA, `ports/gemma4/`, the
  compilation cache, the KV ring.

The rest of this file is that split, written out. It exists because the recurring failure in
this monorepo is not a wrong number, it is a **right number attributed to the wrong rig**.

## Inherited and still true — the chip

- **Turing has no bf16 and no fp8.** float16 is the compute dtype, and asking for bf16
  *emulates* rather than failing. Ported as `resolve_compute_dtype()`, and asserted in
  `tests/test_engine.py` at the SM 8.0 boundary rather than by chip name.
- **The 64 KiB per-block shared-memory ceiling.** It is what forces the vLLM sibling onto
  `TRITON_ATTN` for Gemma 4's 512-wide global-attention head, whose tile wants ~96 KiB. It
  does not bite here — see below — but it is a property of the chip and it is real.
- **Warm up at the shape you measure.** Cold 18.77 s against warm 4.35 s on the JAX rig.
  torch has its own first-call costs (autotune, allocator growth) and the rule holds.
- **15360 MiB of device memory, 14.07 GiB usable** — and this one is a *correction* to an
  inherited assumption rather than a carry. `describe_instance_types` reports 16384 MiB for the
  T4, and `gpu-jax-g4dn-2b`'s README predicted a real 1 GiB advantage over the T4G on that
  basis. **Measured 2026-08-29 on this exact instance type, nvidia-smi reports 15360 MiB —
  identical to the T4G.** Size from 14.07 GiB. The device is what allocates.

## Inherited and still true — the host, only where it is generic

- **A swapfile at or below 16 GiB of host RAM.** The threshold carries; **neither of its two
  documented causes does.** The G5g rig's 8 GiB size could not mmap the checkpoint at all —
  this family has no 8 GiB size, g4dn starts at 16. Its 16 GiB size was OOM-killed inside the
  JAX loader's PLE-table quantiser — this rig has no such code. What remains is generic
  pressure: 16 GiB staging a 10.2 GB checkpoint is thin, and the failure mode is a kernel
  kill under `Restart=on-failure`, which reads as a crash-loop. Cheap insurance, unmeasured.
- **`mkswap -q` is a busybox flag** and util-linux rejects it, killing cloud-init under
  `set -e` before `install.sh` is even written. On the G5g rig this stayed latent behind a
  size nobody launched. **Here the swap block renders for the DEFAULT size**, so the same bug
  would break the first launch.
- **The apt hazards**: unattended-upgrades holding the dpkg lock, and a regional EC2 mirror
  returning 503 over IPv4 while resolving AAAA-only. Both were measured on Graviton hosts but
  neither is architecture-specific.

## NOT inherited — the host

**Everything about aarch64 is gone, and one inherited argument is actively misleading here.**

- **The AMI is x86_64**, and the SSM parameter, the architecture filter and the
  `describe-images` name pattern all changed. AWS names the two architectures' images in
  **different word order** — `Deep Learning ARM64 AMI OSS Nvidia Driver GPU PyTorch …` against
  `Deep Learning OSS Nvidia Driver AMI GPU PyTorch …` — so the sibling's pattern matches
  **zero** x86_64 images (verified 2026-08-29). Carried over unchanged it would have failed
  only when SSM was also unavailable, which is exactly when a fallback matters.
- **`docs/turing-aarch64-gap.md` was not carried.** Half of it is about the wrong host. Its
  Turing half is summarised above; its aarch64 half does not apply.
- **⚠️ "Upstream PyPI wheels omit sm_75" DOES NOT CARRY.** The G5g rig states this flatly and
  is entitled to: it was measured for **aarch64** wheels. Upstream x86_64 CUDA wheels have
  carried Turing for years. Taking torch from the AMI is still right here — vendor build,
  vendor driver, no surprises — but **that is not the reason**, and repeating it would turn an
  unverified claim into a cited one. `verify_gpu_arch` prints `torch.cuda.get_arch_list()` and
  runs a real fp16 matmul; it settles the question on any given image in one call.
- **⚠️ The `torch_xla` argument is half dead.** `docs/INHERITED.md` in the G5g rig rules out
  `torch_xla` on two grounds, and only one survives the move. **"It cannot" is now false**:
  `torch_xla` 2.9.0 publishes `manylinux_2_28_x86_64`, which is exactly this platform. What
  still stands is that PyTorch/XLA's GPU backend is being **retired** (it warns on
  initializing the deprecated XLA:CUDA device and has removed its nightly CUDA builds), and
  that plain CUDA is the better experiment anyway — `torch_xla` would make this JAX→XLA vs
  PyTorch→XLA, two frontends over one compiler, where plain PyTorch makes it XLA vs not-XLA.
  **If you re-derive this decision, derive it from those, not from wheel availability.**
- **The instance-size table is entirely different**, and two of its properties are traps: the
  size suffix does not give the GPU count (`16xlarge` has one T4, `12xlarge` has four), and
  vCPU is RAM/4 rather than the G5g rig's RAM/2 — which is why `server.py` carries vCPUs as
  data instead of computing them.
- **Ubuntu 26.04 and PyTorch 2.13**, against the sibling's 24.04 / 2.12, because x86_64 has
  those lines and arm64 does not. That forces `TORCH_PYTHON_VERSION=3.14`: deadsnakes
  publishes `python3.14` for jammy and noble only, so pinning 3.12 against a 26.04 image would
  miss the system interpreter, take the deadsnakes branch, and fail under `set -e`.

## NOT inherited — the runtime

Dropped at the first fork, and this rig removed what the first fork left behind.

- **`ports/gemma4/`** — the vendored clean-room JAX model port. Not here, so neither is
  `PLE_BITS`, `INT8_LM_HEAD`, `PREFILL_CHUNK_SIZE`, `check_w4a16_fits_scoped_memory()`, the
  fused W4A16 Pallas path, or the hand-written KV ring.
- **`docs/padding-window-eviction.md`** — a correctness bug in that hand-written ring, where
  right-padding to a bucket evicted real tokens from the 512-slot sliding window. There is no
  ring here; transformers owns the cache.
- **`docs/bf16-weights-on-turing.md`** and the 54.0%-dtype-tax profile — JAX-engine artifacts.
  Their **question** is why this rig exists; their numbers are the parent's.
- **The XLA compilation cache.** `gpu-pytorch-g5g-2b` still carries
  `JAX_COMPILATION_CACHE_DIR`, `JAX_CACHE_S3_URI`, `XLA_PYTHON_CLIENT_MEM_FRACTION` and a
  systemd timer pushing `/opt/jax-cache` to S3 every ten minutes — **into a rig with no
  compiler.** Nothing on this path compiles: there is no `torch.compile` in
  `torch_openai_server.py`, by an argument its own docstring makes. So the directory stays
  empty and the timer reports a successful sync of nothing, forever. Both halves working
  correctly against a path nothing writes to is precisely the shape of the JAX rig's own
  cache bug. **Removed here**, with `tests/test_server.py::NoCompilationCacheTests` asserting
  it stays removed. If `torch.compile` is adopted, the knob is `TORCHINDUCTOR_CACHE_DIR`.
- **`tests/test_engine.py` was rewritten, not carried.** The inherited copy's 22 tests all
  imported `ports.gemma4.jax_e_model` and skipped on every run — a suite that always skips
  reads as coverage in the summary line and asserts nothing. It now tests the torch device
  policy, and it runs.

## What is quantised, and why, is a different argument here

Both rigs serve the dense reference checkpoint and **the reasons are not the same one**.

The JAX rigs cannot run the fused W4A16 Pallas kernel: it is tiled for TPU VMEM, lowers
through Triton on GPU, and needs 550 KiB – 1.1 MiB of shared memory per block against
Turing's 64 KiB. `check_w4a16_fits_scoped_memory()` raises at startup with the arithmetic
attached.

Here there is no Pallas and no fused path at all. `AutoModelForCausalLM` has nowhere to put
w4a16 weights without bitsandbytes or torchao, neither of which is installed. Same outcome,
entirely different mechanism — **do not repeat the Pallas argument in this rig.**

## The A/B partner, and what it already settled

**`gpu-jax-g4dn-2b` first served on 2026-08-29** — same instance type, same region, same
checkpoint, same dtype policy, different runtime. Two things follow.

**It closed the host axis.** Decode **13.1 tok/s against the G5g rig's 13.10**,
`tpu_jax_weight_bytes` **6.155 GB against 6.155 GB**, and an xprof profile reproducing 54.4%
dtype conversion / 32.8% fp32 GEMV / 0.0% TensorCore with roofline peaks identical to three
decimals. Its own conclusion: *the host architecture contributes nothing measurable to decode.*
So the 86.9% tax is a **Turing** property, not a Graviton2 one — which means **this rig's
comparison is no longer confounded by the host at all.**

**It gives this rig a baseline on identical hardware**, which no other pair in the tree has:

| input tokens | 41 | 521 | 2,057 |
| --- | ---: | ---: | ---: |
| `tpu_jax_decode_tokens_per_second` | 13.1 | 13.2 | 13.1 |
| end-to-end `output_tok_per_s` | 12.671 | 11.654 | 8.748 |

Median of 3, 64 output tokens, warmed at the measured shape. **Compare on the gauge** — it is
flat in context, while end-to-end falls because it carries prefill and HTTP.

Two operational notes from that run, inherited because they are about this instance type:

- **Spot capacity was NOT available in `us-east-1a`**, and 1a was not the cheapest AZ. Price
  remains a bad proxy for capacity; retry across AZs.
- **The instance role's S3 policy is granted per rig prefix.** That run was nearly blocked
  because `gpu-jax-g4dn-2b` had no prefix on `jax-compilation-cache-rw`. This rig writes
  nothing to S3 today, but anything that starts to will need its own prefix added in kind.

## Numbers you will be offered and must not use

- **13.1 / 13.2 / 13.1 tok/s** — `gpu-jax-g4dn-2b`. The A/B partner above: a baseline to
  **report against**, never this rig's own result.
- **12.4 – 13.10 tok/s** — `gpu-jax-g5g-2b`. Different runtime *and* different host.
- **43.1 / 44.24 tok/s** — `gpu-vllm-g5g-2b`, and obtained with hand-reduced Triton tiles.
  **CORRECTED 2026-08-30 — neither figure is a benchmark, do not compare against either.**
  `43.1` is one sample from the 2026-08-12 first-serve run, whose own report says "single-run,
  single-stream, no repeats and no variance figure", taken with a 19-token prompt. `44.24` has
  **no benchmark artifact anywhere in the tree** — it survives only in `gpu-vllm-g5g-2b/server.py`'s
  swap comment and `tests/test_server.py`, where it was measured 2026-08-13 to show that
  `g5g.xlarge` + a 16 GiB swapfile reaches a healthy endpoint at all. The tile-clamp caveat is real
  but does not distinguish them: it applies to every vLLM-on-T4G number, the good ones included.
  **Compare against `gpu-vllm-g5g-2b/benchmarks/runs/2026-08-14-rust-frontend-g5g/`** — `vllm bench
  serve`, three runs, one `g5g.4xlarge`: c=1 TPOT 31.44 ms (~31.8 tok/s decode), c=4 ~97 tok/s,
  c=8 168.33 tok/s.
- **~44 tok/s on one Inferentia core** — different harness, different silicon.
- **Anything from `~/gemma4-tips`** — that tree duplicated its own artifacts and its
  directory names misattribute both model and chip.

Record this rig's own results in `benchmarks/runs/<date>-<what>-g4dn/`.
