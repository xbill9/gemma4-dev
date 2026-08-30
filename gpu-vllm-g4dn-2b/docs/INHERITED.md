# Inherited documentation — what is NOT here, and why

This directory was a copy of `gpu-jax-g4dn-2b` before it was a vLLM rig, and the vLLM side
was forked from `gpu-vllm-g6-2b` on 2026-08-29. Both lineages leave things a reader might
expect to find here and which were **deliberately not carried over**.

## From the JAX copy — the whole engine, and its documentation

The directory arrived holding `jax_engine.py`, `jax_openai_server.py`, the vendored
`ports/gemma4/` model port, `tune_loop.py`, `profile_decode.py`, `profile_prefill.py` and
three docs. **All of it described a runtime this rig does not use.** It was removed rather
than left in place, and `test_no_jax_module_survived_the_runtime_change` keeps it out.

Three of those docs are real findings and still live in the JAX rig:

- **`bf16-weights-on-turing.md`** — the loader storing bf16 while compute is float16, costing
  54% of decode. **A property of that rig's JAX loader**, which converts at every *use*, per
  step. vLLM converts once at load. **Do not quote the 54% here**; the mechanism does not
  transfer even though the chip does.
- **`padding-window-eviction.md`** — a correctness bug in the vendored port's KV ring. vLLM
  does not use that port at all.
- **`profiling-recipes.md`** — xprof against JAX traces. There is no JAX here.

Read them at `gpu-jax-g4dn-2b/docs/` and `gpu-jax-g5g-2b/docs/`.

## From the vLLM lineage — `turing-aarch64-gap.md` stays in the G5g rig

That document is the measured write-up of why `gpu-vllm-g5g-2b` needs a ~67-minute
from-source build: G5g requires **aarch64 and SM 7.5 together**, and no published CUDA
artifact provides both.

**Its first half is a fact about Graviton2, not about this rig** — G4dn is x86_64, so the
packaging gap simply does not exist here. The monorepo rule is to file a fact by what it is
true of, and keeping a copy would put an aarch64 finding inside an x86_64 rig.

**Its second half does apply**, because the chip is the same Turing silicon, and that half is
re-stated here in `turing-shared-memory.md` — with the framing that matters for this rig,
which is that G4dn keeps the Turing blocker while deleting the aarch64 one.

Read the original at `gpu-vllm-g5g-2b/docs/turing-aarch64-gap.md`.

## `benchmarks/runs/` — empty on purpose

Nothing was copied from either parent. Benchmark JSON has travelled with forks in this tree
before, leaving rigs carrying numbers measured on hardware they are not.
`test_benchmarks_carries_no_other_rigs_runs` keeps this directory empty until something is
measured **here**.

## Articles — not carried

`gpu-jax-g4dn-2b` ships `devto-*.md`, `medium-*.md` and `make-medium.py`. Those are the JAX
rig's articles about a pure-JAX engine on Graviton2. This rig has nothing to publish yet.
