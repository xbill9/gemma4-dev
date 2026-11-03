# Drafter builds for the 2026-11-03 engine comparison

The E2B QAT assistant drafter rebuilt today as bf16 and Q8_0 GGUF, with the llama.cpp converter at
`d1044bc82` (`converter-commit.txt`, `build-*.log`). The hashes are in `sha256.txt`. These files are 96 bytes
smaller than October's builds and hash differently, from the newer converter. The exact Q4_0 drafter is the
rig's `SPEC_DRAFT_MODEL`, unchanged.

The measurements of every drafter on this rig and on `local-vllm-3050oem-2b-q4w4a16emb4` are in
`local-vllm-3050oem-2b-q4w4a16emb4/benchmarks/runs/2026-11-03-drafters-3050oem/REPORT.md`. They come from one
script, which needed both engines.
