# CLAUDE.md — local-pytorch-3050oem-12b-q4w4a16

Gemma 4 12B, the QAT W4A16 repack, in plain PyTorch + transformers on the RTX 3050 OEM (GA106, sm_86, 8192 MiB) in the
i7-5820K desktop `gpu`. Local: nothing to provision. The engine of `local-pytorch-3050oem-2b-q4w4a16` widened to the 12B;
read that rig's `CLAUDE.md` and the 1650 Ti rig's first.

**State as of 2026-11-01: an experiment that does not serve. Use `local-llamacpp-3050oem-12b-q4_0` for the 12B.**

- 27.5 tok/s decode, level with llama.cpp without a drafter (28.6) and half of it with one (56.4).
- Weights 6,388 MiB, so 1.8 GiB of the card is left for cache, prompt reads and the Triton kernels. A 4,096-token prompt
  runs out of memory at `max_ctx` 8192 and 16384; 32768 runs out on a short prompt. The default is 4096.
- At 4096 the server answered 46 BFCL prompts and then ran out of memory on every later request. A restart is the only
  way back. Cause of the failure at request 47 not investigated.
- The model class is `Gemma4UnifiedTextAttention` / `Gemma4UnifiedTextMLP` in `gemma4_unified`; the other Gemma 4 class
  loads and produces garbage digits. Full layers keep V apart from K (K is `k_norm` + RoPE, V is `v_norm` of raw
  `k_proj`), 16 KiB/token.
- `w4attn.py` decodes on a 2-D grid: Kh KV heads, G query heads per KV head.
- Installed: `transformers==5.16.1`, `compressed-tensors==0.17.0` in the user site, torch 2.14.1+cu130, Triton 3.8.0.
- The card is shared with every `local-llamacpp-3050oem-*` rig; stop their servers first.

Runs: `benchmarks/runs/2026-10-31-first-light-3050oem`, `benchmarks/runs/2026-10-31-accuracy-3050oem` (46 of 400, no score).
