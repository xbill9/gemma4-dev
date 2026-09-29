# 2026-09-29 — exact Q4_0 GGUF of Gemma 4 E2B-it QAT, CPU, i7-1360P

**Host:** Lenovo Yoga 9 14IRP8, i7-1360P (4 P-cores + 8 E-cores, AVX2 + AVX-VNNI), 15 GiB,
bare metal, no discrete GPU. **Runtime:** llama.cpp `fc07d781e` (build 11243), CPU only.
**Weights:** `xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf` against Google's
`google/gemma-4-E2B-it-qat-q4_0-gguf` (revision `675cff4`).

## Findings

1. **Both E2B embedding tables are QAT 4-bit data**, like the transformer layers: every block
   of 32 along a row holds at most 16 distinct values on a scale-times-integer grid, levels
   -8..7 (`qat_grid.py`, `qat_lattice.py` on `google/gemma-4-E2B-it-qat-q4_0-unquantized`).
   The audio tower's attention and feed-forward weights are 2-bit QAT (levels -2..1); the
   vision tower is full precision. `MODELS.md` holds the embedding result for the
   safetensors path; this run adds the GGUF side.
2. **Google's GGUF moves the QAT values.** Its Q4_0 layers use llama.cpp's `amax/8` step, and
   in blocks whose peak is below level 8 values shift by up to one full step: 51% of
   `blk.0.attn_q` values equal the source. Its embeddings are Q6_K, off by up to 28% of a
   grid step.
3. **Rebuilt on the trained step, every tensor lands on the grid.** `gguf_exact.py` keeps
   Google's metadata byte for byte and writes the 275 layer matrices and both embedding
   tables as Q4_0. No block is off the grid; 96.84% of values are bit-identical to the
   source after bf16 rounding and the rest are within 4% of a step (fp16 scale rounding).
   `build_report.json` has the per-kind counts.
4. **The rebuilt file is smaller and closer to bf16.** 2,640,154,592 B against 3,349,516,256.
   Against a bf16 GGUF of the same QAT weights with identical metadata (wikitext-2 test,
   16 × 512 tokens):

   | | Google | Rebuilt |
   |---|---|---|
   | Mean KL divergence | 0.054303 ± 0.001385 | 0.001753 ± 0.000075 |
   | 99th percentile KL divergence | 0.427642 | 0.016390 |
   | Same top token | 87.181 ± 0.523 % | 98.064 ± 0.216 % |
   | PPL / PPL(bf16) | 1.089218 ± 0.009458 | 1.021292 ± 0.005067 |

5. **Generation is 7% faster; prefill shows no stable difference.** Native build, `-t 8`,
   pp512/tg128, r=5, run in both orders (`bench-native-abba.md`):

   | order | pp512 ratio (rebuilt/Google) | tg128 ratio |
   |---|---|---|
   | Google first, 76 → 80 °C | 1.018 | 1.069 |
   | rebuilt first, 53 → 67 °C | 1.305 | 1.074 |

   The generation ratio holds in both orders and matches the Docker run (1.075,
   `docker/bench.md`). The prefill ratio follows the order and is not claimed. `token_embd`
   is also the output projection, read in full every token, and is 31.4% smaller at Q4_0 than
   Q6_K; memory traffic was not measured.
6. **Five greedy chat prompts gave identical answers** from Google's file, the rebuilt file
   and bf16, thinking off (`docker/chat-*.json`).

## Method

- **KL divergence and chat:** `ghcr.io/ggml-org/llama.cpp:full` at the same commit, `-t 8`.
  `llama-perplexity --kl-divergence-base` on the bf16 reference, then `--kl-divergence` for each
  quantized file. The reference is `gguf_exact.py --bf16`: Google's metadata with the source
  bf16 bytes, so the three files differ only in weight storage. The base logits (2.1 GB)
  are not kept; the logs are in `docker/`.
- **Throughput:** this rig's native build (`~/llama.cpp-fc07d78/build-cpu`, `GGML_NATIVE=ON`,
  gcc 16.2), `CUDA_VISIBLE_DEVICES=` empty. Package temperature from the `x86_pkg_temp` zone,
  logged around each pass; 90 s pause between passes.
- **Threads** (`threads-raw.md`, one invocation, 60 → 67 °C): generation at `-t` 4 / 8 / 12 is
  22.60 / 21.82 / 22.62 tok/s, inside one another's spread; `-t 16` is 16.77, 23.1% below
  `-t 8`. Prefill 118.61 / 110.28 / 112.90 / 106.62, with the 4-thread cells first and
  coolest. `THREADS=8` matches every other measurement here; this is a spot-check, not a sweep.
- **Native against Docker:** at `-t 8` the native build measured 1.46x (pp512) and 1.47x
  (tg128) the Docker image's figures for the rebuilt file. The two runs were about an hour
  apart and not interleaved, so this is a direction, not a ratio.
- **First light** (`first-light.md`): started, attested, queried and stopped through this
  rig's MCP tool functions. Attested `cpu`, no GPU library mapped, `-ngl 0`. `RssAnon`
  1,433,372 kB with the model loaded after one query; `RssFile` 2,578,532 kB with the page
  cache warm, so it says nothing about how much of the lazy table inference touches.

## Rebuild

```bash
python3 gguf_exact.py <google/gemma-4-E2B-it-qat-q4_0-unquantized@6befbac> \
    <google/gemma-4-E2B-it-qat-q4_0-gguf@675cff4>/gemma-4-E2B_q4_0-it.gguf out.gguf
sha256sum out.gguf   # 25f21f14a99d313fb3d017e3350ad7be11b1b17dd3ba591c333c23c09762a390
```

numpy only; about 4 minutes on this host. Published with this evidence at
https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf.
