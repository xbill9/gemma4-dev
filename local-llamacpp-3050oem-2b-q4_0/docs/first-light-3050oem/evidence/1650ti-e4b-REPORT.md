# 2026-09-30 — exact Q4_0 GGUF of Gemma 4 E4B, GTX 1650 Ti

First light of `local-llamacpp-1650ti-4b-q4_0`, the E4B parallel of `local-llamacpp-1650ti-2b-q4_0`.
Same host, binary and flags as that rig's `2026-09-29-exact-gguf-v2-1650ti`: llama.cpp `f95b0d9`,
`-ngl 99 -c 8192 -ctk f16 -ctv f16 -fa 1 -t 6 -tb 12`.

## Question: does E4B fit in 4096 MiB?

**Yes, with ~820 MiB to spare.** Predicted from `config.json` and Google's tensor table before loading;
measured from the `-lv 4` allocation log (`serve/alloc-*.log`) and `nvidia-smi` (`serve/vram-*.txt`).

| | predicted | Google's GGUF | exact rebuild |
|---|---:|---:|---:|
| CUDA0 model buffer | 2493 MiB (exact) | 2696.06 MiB | **2493.32 MiB** |
| KV, 4 full layers × 8192 cells | 128 MiB | 128.00 | 128.00 |
| KV, 20 sliding layers × 1024 cells | 40 MiB | 40.00 | 40.00 |
| CUDA0 compute buffer | — | 110.02 | 110.02 |
| **`nvidia-smi`, whole process** | ~2.85 GiB | **3052 MiB** | **2848 MiB** |

3671 MiB was free before load (the card reports 3732 usable). E2B's exact file is 1480 MiB on the same
flags, so **the two rigs cannot share the card** — and they share port 8080 too.

It fits for the reason E2B does: `per_layer_token_embd` (`[10752, 262144]`, 1.585 GB at Q4_0, 38% of the
file) is `TENSOR_READ_LAZY` and never reaches VRAM. Resident, E4B would need ~4.4 GiB and could not load.
The KV figures confirm `MODELS.md`'s E4B derivation (2 KV heads, 24 cache-owning layers, 4 of them full)
against an allocation log for the first time, with llama.cpp's cap of the sliding layers at 1024 cells.

## The rebuild

`gguf_exact.py` is the E2B rig's v2 script, unchanged except its docstring: E4B's GGUF has the same
tensor set (666 tensors; `attn_k`/`attn_v` on only the 24 cache-owning layers) and the same HF names.

- 441 s, peak RSS 6.2 GiB. **No group off the 4-bit grid** in any tensor (the script stops otherwise).
- 4,215,695,712 B against Google's 5,154,941,280. SHA-256
  `5462cc102b7d22c4748f9b8e82091f2ae064e10a45525151af6f9a7d0da7caf9`.
- 96.7–97.0% of values reconstruct to the source bf16 exactly, per tensor kind (`build_report.json`);
  E2B's v2 was 96.7–97.0% too.
- `make info`: 4.198 GB of Q4_0, 0.002 GB of F32 norms, nothing else. Google's file is 2.86 GB Q6_K
  (both embedding tables), 2.22 GB Q4_0 and 55 MB F16 (`per_layer_model_proj`).

## Speed: exact vs Google, ABBA

`bench_abba.sh`, pp512/tg128, r=5, cooled to ≤50 °C before each pass (`bench-abba.md`).

| pass | order | tg128 G / X | X/G | pp512 G / X | X/G |
|---|---|---|---:|---|---:|
| 1 | G,X | 37.56 / 39.75 | 1.058 | 170.63 / 172.11 | 1.009 |
| 2 | X,G | 37.58 / 39.94 | 1.063 | 170.31 / 173.45 | 1.018 |
| 3 | X,G | 37.56 / 39.92 | 1.063 | 169.01 / 173.46 | 1.026 |
| 4 | G,X | 37.55 / 39.75 | 1.059 | 170.34 / 172.18 | 1.011 |

**Decode 1.058–1.063x Google's file in every pass and order**, prefill 1.01–1.03x. That is about half
E2B's gain (1.11–1.12x), which fits where the bytes are: E4B's transformer body, already Q4_0 in Google's
file, is a larger share of what decode streams, so replacing the Q6_K `token_embd` (the tied LM head) moves
a smaller fraction. Not separated further.

**E4B against E2B on this card:** tg128 ~39.8 against E2B v2's ~81.7 (0.49x); pp512 ~173 against ~345.
Different sessions, same binary and flags — a rough ratio, not a paired result.

## Greedy chat (`serve/chat-*.json`)

Five prompts at temperature 0, exact file (`chat-exact.json`): Canberra, 391, a correct string-reverse
function, Io/Europa/Ganymede, and a coherent three-sentence French Rayleigh-scattering answer. ~37–39 tok/s
per request. Google's file gave a correct one-sentence TPU answer at 36.7 tok/s.

## Quality: KL divergence against bf16 (`kld/`)

Added the same day, before publishing. `kld/run_kld.sh`: `gguf_exact.py --bf16` builds the reference
(Google's metadata, the source's bf16 bytes, 14.9 GB), then `llama-perplexity` from
`~/llama.cpp-f95b0d9` on wikitext-2 test, 16 × 512, **CPU only** (`CUDA_VISIBLE_DEVICES=` empty,
`-t 6`), so the E2B server stayed up. 11m32 for all three passes on 15 GiB of RAM: the bf16 file
streams from the page cache.

| | Google | exact |
|---|---|---|
| Mean KL divergence | 0.035254 ± 0.000773 | **0.001031 ± 0.000053** |
| 99th percentile KLD | 0.215774 | 0.008396 |
| Same top token | 90.637 ± 0.456 % | **98.407 ± 0.196 %** |
| PPL / PPL(bf16) | 1.081619 ± 0.008910 | 1.025568 ± 0.005199 |

34x lower mean KLD. Google's E4B file is closer to bf16 than its E2B file was (0.035 against 0.054), and
the exact rebuild lands in the same place as E2B's (0.0010 against 0.0017). Not measured on CUDA.

## Published

`xbill9/gemma-4-E4B-it-qat-q4_0-exact-gguf`, public, Apache 2.0, laid out like the E2B repo: the GGUF,
`gguf_exact.py`, Google's card as `ORIGINAL_README.md`, and `evidence/` (build report, bench, allocation
logs, chat, KLD logs).

## NOT DONE

- KLD on CUDA; no `sweep.py` run, no concurrency sweep, no thinking-on check, no context above 8192.
- Rebuilt on one machine only (E2B's was reproduced bit-identically on two).

## Files

`gguf_exact.py`, `build.log`, `build_report.json`, `bench_abba.sh` → `bench-abba.md` / `.err`,
`kld/`, `serve/` (allocation logs, `nvidia-smi` readings, chats, and the E2B server's command line as it was
before it was stopped for this run).
