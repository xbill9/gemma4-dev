# CLAUDE.md — `tpu-vllm-v5e1-4b-w8a8`

## This is an artifact rig, not a serving rig

No `server.py`, no MCP server, no skill, no plugin manifest, no `tpu.env`, and none is owed. It holds
sweep measurements for `/work/models/gemma-4-E4B-it-qat-w8a8-int8` on TPU v5e-1 (`v5litepod-1`) and nothing else. Do not
scaffold it into a full rig. `NAMING.md` has the rule for artifact rigs.

## Where the runs came from

Each run under `benchmarks/runs/` is one arm of a sweep run by the harness named in its `README.md`
(`jev-tpu`, `jev-tpu-31b` or `jev-tpu-v5e1`), filed here on 2026-09-30 because this directory names its
chip and exact checkpoint. The sweep keeps a relative symlink at the arm's old `results/` path, so its
scripts and comparison files still resolve; `benchmarks/sweep-moves.json` at the monorepo root maps
every old path to its new one. Run-wide logs (run.log, patch logs, checksums) stayed with the sweep.

**The directory name is a claim about the weights measured.** Two checkpoints with the same numeric
format are different cells when their values differ: `w4a16` (Google's min-max re-rounding) and
`q4w4a16` (the QAT grid kept exactly) differ by 1.3 to 2.4 suite points at E2B and E4B.
