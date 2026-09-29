# Inherited documentation — what is NOT here, and why

This rig was forked from `gpu-vllm-g6-2b` on 2026-09-29.

- **`benchmarks/runs/` and `benchmarks/reports/` are empty.** The parent's
  `2026-08-30-first-serve-g6` measured E2B bf16 on vLLM 0.28.0; it is not a measurement of this
  rig's checkpoint or settings, and benchmark files that travel with forks are how this tree has
  misattributed results before.
- **The G6 background is in the parent.** Why G6 needs no build, the SM 8.9 image coverage,
  the Triton shared-memory margin and the image-tag history are facts established on
  `gpu-vllm-g6-2b`; read `../gpu-vllm-g6-2b/CLAUDE.md` and `../gpu-vllm-g6-2b/docs/INHERITED.md`.
