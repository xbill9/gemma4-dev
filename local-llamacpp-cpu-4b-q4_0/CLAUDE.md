# CLAUDE.md — local-llamacpp-cpu-4b-q4_0

The **CPU arm of the E4B pair**, forked on 2026-09-30 from `local-llamacpp-cpu-2b-q4_0`; slot 4 is the
only slot that differs. Its GPU twin is `local-llamacpp-1650ti-4b-q4_0`. Both serve the same bytes
(`xbill9/gemma-4-E4B-it-qat-q4_0-exact-gguf`, SHA-256 `5462cc10…`) from the same llama.cpp commit
(`f95b0d9`) on the same port, and their `make serve` differs only in binary (`build-cpu/`), the hidden
CUDA device and `-ngl 0`. `sweep.py` is byte-identical in both arms.

**Read `local-llamacpp-cpu-2b-q4_0/CLAUDE.md`** for everything about the CPU build, the thread
derivation, the refusal of a GPU binary, attestation and the host's thermal behaviour. Those are
properties of this laptop and this binary, and every number in them — and in this rig's `tpu.env`
comments — was **measured on E2B**, not here.

What is E4B-specific: 4.22 GB file, 1.59 GB of it the lazily read `per_layer_token_embd`; the rest is
read per token from the page cache on a 15 GiB host. KL divergence against bf16 was measured on this
CPU through a separate `llama-perplexity` binary: 0.00103 against Google's E4B file's 0.0353
(`local-llamacpp-1650ti-4b-q4_0/benchmarks/runs/2026-09-30-exact-gguf-e4b-1650ti/kld/`).

Measurements: `benchmarks/runs/2026-09-30-paired-sweep-cpu`, the ABBA pair (driver and analysis in the
GPU twin's run directory of the same date). GPU **4.20x** decode, **3.84x** prefill, **3.96x** end-to-end;
this arm reads 8.3–9.5 tok/s decode, ~0.51x its E2B parent's. **This arm drifted −8% between its two
passes** (the second started 8 °C warmer): never read a single CPU pass as the arm's speed.
