# 2026-09-29 — exact Q4_0 GGUF of Gemma 4 E2B-it QAT, GTX 1650 Ti

The re-pack from `local-llamacpp-i71360p-2b-q4_0` (`2026-09-29-exact-gguf-i71360p`), repeated on
this machine and measured on the card.

**Host:** i7-10750H + GTX 1650 Ti Max-Q (4096 MiB, sm_75, no tensor cores), Debian sid, driver
615.71.09. **Runtime:** llama.cpp `f95b0d9` (build 318). Throughput and serving use this rig's own
binary (`~/llama.cpp/build`). `llama-perplexity` is not in that build, so it came from a separate
worktree at the same commit (`~/llama.cpp-f95b0d9/build`, same cmake flags, `-j6`); `~/llama.cpp`
was not touched. **Weights:** `google/gemma-4-E2B-it-qat-q4_0-gguf@675cff4` (SHA-256 `fa401b55…`,
checked against the Hub) and the rebuild from `google/gemma-4-E2B-it-qat-q4_0-unquantized@6befbac`.

## Findings

1. **The rebuild is bit-identical across machines.** `gguf_exact.py` (copied unchanged) ran in 303 s
   here and wrote SHA-256 `25f21f14a99d313fb3d017e3350ad7be11b1b17dd3ba591c333c23c09762a390`, the
   hash the i7-1360P produced and published. `build_report.json` is identical to that run's.
2. **The quality gap holds on CUDA.** Against a bf16 GGUF with identical metadata, wikitext-2 test,
   16 × 512 (`kld/`):

   | | Google, CUDA | exact, CUDA | Google, CPU | exact, CPU |
   |---|---|---|---|---|
   | Mean KL divergence | 0.054244 ± 0.001357 | **0.001485 ± 0.000053** | 0.054303 ± 0.001385 | 0.001753 ± 0.000075 |
   | 99th percentile KLD | 0.412949 | 0.011322 | 0.427643 | 0.016382 |
   | Same top token | 87.206 ± 0.523 % | **98.088 ± 0.214 %** | 87.181 ± 0.523 % | 98.064 ± 0.216 % |
   | PPL / PPL(bf16) | 1.092450 | 1.025756 | 1.089218 | 1.021293 |

   The CPU columns match the i7-1360P's to every printed digit, on a different CPU and a different
   llama.cpp commit. The base logits were computed on the CPU (bf16 does not fit in 4 GiB).
3. **Generation is 1.10x faster; prefill is unchanged.** `bench_abba.sh`: llama-bench, `-ngl 99 -fa 1
   -t 6`, pp512/tg128, r=5, four passes in ABBA order, with the card cooled to ≤50 °C before each pass
   (`bench-abba.md`):

   | pass | first | tg128 Google | tg128 exact | ratio | pp512 ratio |
   |---|---|---:|---:|---:|---:|
   | 1 | Google | 73.63 | 80.45 | 1.093 | 0.999 |
   | 2 | exact | 73.22 | 80.57 | 1.100 | 1.003 |
   | 3 | exact | 73.14 | 80.47 | 1.100 | 1.007 |
   | 4 | Google | 73.54 | 80.44 | 1.094 | 1.004 |

   Unlike the i7-1360P's CPU prefill, nothing here depends on order beyond ~0.7%. `token_embd` is the
   tied output projection and is read in full every token; at Q4_0 it is 31.4% smaller than at Q6_K.
   Decode at B=1 on this card is bandwidth-bound, so that is the expected direction. Memory traffic
   was not measured.
4. **VRAM falls by 99 MiB.** Same flags as `make serve`, from llama-server's allocation log
   (`serve/alloc-*.log`, `-lv 4`):

   | buffer | Google | exact |
   |---|---:|---:|
   | CUDA0 model | 1341.78 MiB | 1242.78 MiB |
   | CPU_Mapped model (includes lazy PLE) | 2152.50 MiB | 1476.00 MiB |
   | CUDA0 KV (non-SWA + SWA) | 48 + 12 MiB | 48 + 12 MiB |
   | CUDA0 compute | 122.52 MiB | 122.52 MiB |
   | `nvidia-smi` process total | 1614 MiB | 1514 MiB |

   99.0 MiB is `token_embd` 330.3 MB → 226.5 MB, the only resident tensor whose type changed.
5. **Greedy chat: one CUDA-only divergence, and it is not the file's.** Five prompts, thinking off
   (`serve/chat-*.json`). On this machine's CPU, bf16, Google's file and the rebuild give identical
   answers, as they did on the i7-1360P. On CUDA, Google's file gives the same answers; the rebuild
   gives the same four and indents the Python answer with four spaces instead of two. The
   rebuild's CPU answer matches bf16, so the flip comes from the CUDA backend (it quantizes
   activations for its Q4_0 matmuls) and not from the weights. Its CUDA KLD is the lowest of the four
   columns above.

## Not changed

This rig still serves Google's file: `tpu.env` is unchanged. Moving this rig alone would put the
weights inside the GPU-vs-CPU delta that `local-llamacpp-cpu-2b-q4_0` is paired against.

## Files

- `gguf_exact.py`: the i7-1360P run's script, unchanged. `build.log` and `build_report.json` are its output.
- `bench_abba.sh` → `bench-abba.md`.
- `kld/`: `llama-perplexity` logs. The bf16 base logits (2.1 GB) are not kept.
- `serve/`: allocation logs, `vram.txt`, chat outputs (`chat-{google,exact}.json` on CUDA,
  `chat-cpu-*.json` on CPU). `chat_check.py` is the i7 run's script with a port argument.

## Rebuild

```bash
python3 gguf_exact.py <google/gemma-4-E2B-it-qat-q4_0-unquantized@6befbac> \
    <google/gemma-4-E2B-it-qat-q4_0-gguf@675cff4>/gemma-4-E2B_q4_0-it.gguf out.gguf
python3 gguf_exact.py <same> <same> ref-bf16.gguf --bf16   # KLD reference
```
