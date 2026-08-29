# What came from the JAX sibling, and what did not

Forked from `gpu-jax-g5g-2b` on 2026-08-28. **No measurement in this rig is its own yet.**

Inherited and still true, because they are about the *chip* or the *host*:

- **Turing has no bf16 and no fp8**, so float16 is the compute dtype, and asking for bf16
  emulates rather than failing. Ported as `resolve_compute_dtype()`.
- **The AMI must be arm64 and carry the NVIDIA driver.** AWS also ships ARM64 DLAMIs built for
  Graviton *CPU* inference; they boot fine and have no GPU.
- **`g5g.2xlarge` has exactly 16 GiB of host RAM** and needs a swapfile — inclusive threshold.
- **Warm up at the shape you measure.** Cold 18.77 s against warm 4.35 s on the JAX rig; torch
  has its own first-call costs and the rule holds.

Deliberately **not** inherited:

- `docs/turing-aarch64-gap.md` — kept in the parent. Its Part 1 is directly relevant (it is
  where the `torch 2.12.0+cu132 sm_75` measurement comes from) but its Part 2, the Triton
  shared-memory blocker, is a vLLM problem this rig may or may not share.
- `docs/bf16-weights-on-turing.md` — the dtype-tax investigation is a JAX-engine artifact. Its
  *question* is why this rig exists; its numbers are the parent's.
- `docs/padding-window-eviction.md` — a correctness bug in the hand-written JAX KV ring. There
  is no hand-written ring here; transformers owns the cache.

## Why this rig is CUDA PyTorch and not `torch_xla` — settled 2026-08-28

The obvious question for a PyTorch rig in a repo full of XLA is whether it should lower through
XLA too, which would make it a frontend-only A/B against `gpu-jax-g5g-2b`. **It should not, and
it cannot.** Written down with the evidence so nobody spends an afternoon re-deriving it.

### It cannot: there is no wheel for this platform

`torch_xla` 2.9.0 publishes for exactly one platform — `manylinux_2_28_x86_64`, on cp310–cp313.
No aarch64, no arm, no CUDA-tagged build. This host is Graviton2, so there is nothing to install.
That is RIG-ANALYSIS step 7 answered before step 9 costs a launch.

### It is not that XLA can't reach this chip — the sibling rig disproves that daily

The asymmetry is the whole point, because both frameworks sit on the same compiler:

| package | latest | aarch64 wheel? |
| --- | --- | :---: |
| `jaxlib` | 0.11.1 | **yes** — `manylinux_2_27_aarch64` |
| `jax-cuda13-pjrt` | 0.11.1 | **yes** |
| `jax-cuda13-plugin` | 0.11.1 | **yes** |
| `torch_xla` | 2.9.0 | **no** — x86_64 only |

`jax-cuda13-pjrt` *is* the XLA CUDA runtime for aarch64, shipping today, through the same PJRT
plugin architecture `torch_xla` uses. And `gpu-jaxrust-g5g-2b/docs/rust-jax-runtime-survey.md`
found it a third way: `elixir-nx/xla` publishes `xla_extension-0.10.0-aarch64-linux-gnu-cuda13.tar.gz`
as a release asset. **Two independent projects ship aarch64+CUDA XLA binaries.** The compiler is
not the obstacle; the PyTorch-flavoured route to it is.

### Why PyTorch/XLA does not build it

- **Its market is x86_64.** `torch_xla` exists for Cloud TPU VMs, which are x86_64 hosts — which
  is also why the `tpu-pytorch-*` rigs in this repo work at all.
- **The GPU path is being retired, not extended.** PyTorch/XLA now warns on initializing the
  deprecated XLA:CUDA device and has removed its nightly CUDA builds. Nobody adds an
  aarch64+CUDA target to a backend they are withdrawing.
- **A wheel bundles the XLA runtime**, so a new platform is a new CI runner, toolchain and
  permanent test-matrix entry. JAX pays that because aarch64 is first-class for it (Graviton,
  GH200); PyTorch/XLA has no such pull.

### And plain CUDA is the better experiment anyway

`torch_xla` would have made this **JAX→XLA vs PyTorch→XLA** — two frontends over one compiler,
which is a narrow question. Plain PyTorch makes it **XLA vs not-XLA**, which is the question
actually open: the JAX sibling spends 54.0% of decode in dtype conversion at 0.0% TensorCore and
nobody knows whether that is the chip or the compiler.

- Same conversion tax here → it is Turing, and XLA is exonerated.
- No conversion tax here → it is something XLA does, and the JAX rig has a real target.

`torch_xla` would lower through XLA too and would likely reproduce the tax for reasons that teach
nothing. **Slot 2 is documentation of which stack loads the weights, not a performance claim**
(NAMING.md) — and `NAMING.md` already glosses `pytorch` as "via `torch_xla` on TPU or CUDA on
GPU", so the name is correct either way and needs no amendment.

**Sources:** PyPI release metadata for the four packages above, read 2026-08-28 ·
`gpu-jaxrust-g5g-2b/docs/rust-jax-runtime-survey.md` · https://github.com/pytorch/xla/releases
