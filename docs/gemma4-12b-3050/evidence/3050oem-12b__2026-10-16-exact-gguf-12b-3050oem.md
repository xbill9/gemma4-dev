# 2026-10-16 — exact Q4_0 GGUF of Gemma 4 12B, built on the 3050 host

**Answer: the 12B QAT weights rebuild exactly onto Q4_0 like E2B's and E4B's, and the result is
smaller, closer to bf16 and faster than Google's GGUF.** Published as
[`xbill9/gemma-4-12B-it-qat-q4_0-exact-gguf`](https://huggingface.co/xbill9/gemma-4-12B-it-qat-q4_0-exact-gguf)
(commit `85e797e`); the card is `hf/README.md`.

## The rebuild

`gguf_exact.py` is the E4B build's script
(`local-llamacpp-1650ti-4b-q4_0/benchmarks/runs/2026-09-30-exact-gguf-e4b-1650ti`), unchanged but
for its docstring. Its name map already covers the 12B: 329 tensors rebuilt, every one bf16 in the
source, and the 8 full-attention layers have no `attn_v` in either file (`attention_k_eq_v`).

- Source `google/gemma-4-12B-it-qat-q4_0-unquantized` at `b6ed862`; template Google's
  `gemma-4-12b-it-qat-q4_0.gguf` at `29d0977` (SHA-256 `93567e57…`).
- 1,039 s, peak RSS 20.50 GiB (mostly the memory-mapped 23.9 GB source). **No group off the
  4-bit grid.**
- **6,716,356,736 B** against Google's 6,975,879,296. SHA-256 `038337ee…` (`sha256.txt`).
- Google's file is 6.131 GB Q4_0 and 0.826 GB Q6_K, the Q6_K being `token_embd` alone (787.5 MiB).
  The rebuild stores it as Q4_0 (540 MiB).
- 96.77–96.88% of values per tensor kind reconstruct to the bf16 source exactly
  (`build_report.json`).

## Divergence from bf16 (`kld/run_kld.sh`)

CPU, llama.cpp `7182ef71e`, wikitext-2 test, 16 × 512 tokens, against `--bf16` output of the same
script (Google's metadata, the source weights unchanged). 39 minutes for all three.

| | Google's GGUF | exact rebuild |
| --- | ---: | ---: |
| mean KLD | 0.543225 ± 0.019046 | **0.124504 ± 0.009525** |
| 99th percentile KLD | 6.217015 | **1.914886** |
| same top token | 71.77 ± 0.71 % | **89.14 ± 0.49 %** |
| PPL / PPL(bf16) | 1.3855 ± 0.0449 | **1.2567 ± 0.0248** |

**The rebuild is 4.4x closer by mean KLD.** These figures are not comparable with the E4B build's
(mean KLD 0.0010): bf16 perplexity on this text is 312 for the 12B against 40 for E4B. The 12B
IT checkpoint degrades on text without its chat format (`MODELS.md`, §12B), and the divergence is
measured in that regime. Answer quality is in `../2026-10-16-accuracy-12b-3050oem`.

## Speed and memory

In `../2026-10-16-first-light-3050oem`: 1.035x Google's file on generation, prefill level,
247.5 MiB less VRAM, and 40,960 tokens of f16 context on the 8 GB card against 28,672.
