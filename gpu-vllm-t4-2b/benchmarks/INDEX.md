# gpu-vllm-t4-2b — benchmark index

**GENERATED FILE.** Regenerate from the monorepo root with `make benchmarks-rollup`.
Do not hand-maintain it: a hand-written per-rig index is how the old run-index table
came to claim other rigs' results.

**This rig has measured nothing.** `runs/` is empty on purpose, and no report was
copied at the fork — deliberately, because benchmark JSON travelling with a fork is
how several rigs in this tree ended up carrying numbers measured on hardware they
are not.

## The closest measured data is NOT in this rig

`gpu-vllm-g4dn-2b/benchmarks/runs/2026-08-30-first-serve-g4dn/` measured this exact
serving configuration on the same T4 silicon (x86_64 host, vLLM 0.28.0, float16,
patched for Turing). Read it there. **Do not copy it here**, and do not difference a
future number from this rig against it without noting that the host differs: 2 vCPU
against 4, 7.80 GB of RAM against 16 GiB, and no swap.

When this rig does measure something, the filename is date-first per `@NAMING.md`:
`benchmarks/reports/<date>-gemma4-e2b-t4.json` — `e2b`, not the `2b` of the model
slot, and `t4`, the same value as the hardware slot.
