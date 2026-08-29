# Inherited documentation — what is NOT here, and why

This rig was forked from `gpu-vllm-g5g-2b` on 2026-08-28. Two documents that a reader might
expect to find here were **deliberately not copied**.

## `turing-aarch64-gap.md` — lives in the sibling only

That document is the measured write-up of why `gpu-vllm-g5g-2b` needs a ~67-minute
from-source build and an unlanded Triton patch: G5g requires **aarch64 and SM 7.5 together**,
and no published CUDA artifact provides both.

**It is a fact about the T4G and Graviton2, not about this rig**, and the monorepo rule is to
file a fact by what it is true of. Keeping a copy here would put a Turing measurement inside
an Ada rig, which is precisely the misattribution this tree keeps getting bitten by.

Read it at **`gpu-vllm-g5g-2b/docs/turing-aarch64-gap.md`**.

What carries forward from it into this rig is summarised in `CLAUDE.md`:

- the amd64/arm64 arch-list table, which is *why there is no build here*; and
- the Triton 96 KiB shared-memory requirement, which is the one thing that **might** still
  bite — Ada allows ~99 KiB against Turing's 64, a real but narrow margin, and **unverified**.

## `benchmarks/runs/` — empty on purpose

The sibling's `2026-08-12-first-serve-g5g` and `2026-08-14-rust-frontend-g5g` were not
copied. Benchmark JSON has travelled with forks in this tree before, leaving rigs carrying
numbers measured on hardware they are not. `test_benchmarks_carries_no_other_rigs_runs` keeps
this directory empty until something is measured **here**.
