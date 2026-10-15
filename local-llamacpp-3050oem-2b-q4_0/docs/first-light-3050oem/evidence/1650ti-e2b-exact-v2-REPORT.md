# 2026-09-29 — exact Q4_0 GGUF v2 (`per_layer_model_proj` at Q4_0), GTX 1650 Ti

Follows `../2026-09-29-exact-gguf-1650ti`, and repeats the i7-1360P rig's
`2026-09-29-exact-gguf-v2-i71360p` on this machine. Same host, binaries and flags as that run:
llama.cpp `f95b0d9`, rig binary for bench/serve, worktree `llama-perplexity`.

## Findings

1. **v2 rebuilds bit-identically here too.** `gguf_exact.py` (the i7 v2 script, unchanged), 305 s,
   SHA-256 `419db9a6bf3bc15d770c85ccf9216827aa88c64fe8f3d72fbe5674f48efe2dc8`, 2,620,370,912 B;
   `build_report.json` identical to the i7's.
2. **Quality: v1 and v2 tie on CUDA.** Against the v2 bf16 reference (its `per_layer_model_proj` is
   the source's bf16, not Google's F16), wikitext-2 test 16 × 512, base logits on CPU, quantized
   files on CUDA `-ngl 99` (`kld/`):

   | | Google | v1 | v2 |
   |---|---|---|---|
   | Mean KL divergence | 0.054264 ± 0.001359 | 0.001482 ± 0.000053 | 0.001680 ± 0.000072 |
   | 99th percentile KLD | 0.406903 | 0.011296 | 0.016059 |
   | Same top token | 87.181 ± 0.523 % | 98.088 ± 0.214 % | 98.260 ± 0.205 % |
   | PPL / PPL(bf16) | 1.092092 | 1.025420 | 1.021563 |

   v1 is ahead on mean KLD and v2 on top token and PPL, each by about two standard errors or less.
3. **v2 is faster on the card, unlike on the i7-1360P's CPU.** `bench_abba.sh`, pp512/tg128, r=5,
   ABBA, cooled to ≤50 °C before each pass (`bench-abba.md`):

   | pass | order | tg128 G / v1 / v2 | v2/v1 | v2/G | pp512 v2/v1 |
   |---|---|---|---:|---:|---:|
   | 1 | G,v1,v2 | 73.46 / 80.32 / 81.65 | 1.017 | 1.111 | 1.012 |
   | 2 | v2,v1,G | 73.10 / 80.26 / 81.82 | 1.019 | 1.119 | 1.019 |
   | 3 | v2,v1,G | 73.06 / 80.36 / 81.92 | 1.019 | 1.121 | 1.019 |
   | 4 | G,v1,v2 | 73.42 / 80.30 / 81.59 | 1.016 | 1.111 | 1.016 |

   The i7-1360P measured v2/v1 at 0.997 and 1.014 for generation. Here it holds at 1.016-1.019 in
   every pass and order, and prefill also gains 1.2-1.9%. The 7.7 MB tensor saves ~1.6% of the bytes
   decode reads. It also replaces an F16 matmul, which this card runs without tensor cores, with a
   Q4_0 one; which of the two accounts for the gain was not separated.
4. **VRAM:** CUDA0 model buffer 1223.91 MiB (v1 1242.78, Google 1341.78); `nvidia-smi` 1480 MiB.
   KV and compute buffers are unchanged (`serve/alloc-v2.log`).
5. **Greedy chat** (`serve/chat-v2.json`, CUDA): identical to bf16 on four prompts. The long French
   answer first differs at char 202, the same place as Google's file on CUDA.

## What is left in the file

`gguf-py` over v2: 2,604,554,380 tensor bytes, of which **2,603,409,408 are Q4_0** and 1,144,972
are F32 (norms, `rope_freqs`, `layer_output_scale`). Nothing else is stored in any other type.
Q4_0 (4-bit levels plus an fp16 step per 32 values) is the format the QAT checkpoint was trained
onto, so any smaller llama.cpp type takes the weights off that grid and costs KLD. Resident on the
card: 1224 MiB, of which `token_embd` (the tied output projection) is 216 MiB.

## Files

`gguf_exact.py`, `build.log`, `build_report.json`, `bench_abba.sh` → `bench-abba.md`, `kld/`,
`serve/`. The bf16 base logits are not kept.
