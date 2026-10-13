# 2026-10-13 — first light, RTX 3050 OEM

`local-llamacpp-3050oem-2b-q4_0`, the exact Q4_0 E2B GGUF (v2, SHA-256 `419db9a6…`, checked after
download) on one RTX 3050 OEM, llama.cpp `77c4540ce` built with `make build` (CUDA 13.4, sm_86,
gcc 16.2.0).

## Host

| | |
| :--- | :--- |
| GPU | NVIDIA GeForce RTX 3050 OEM, PCI `10de:2508` (GA106), compute capability 8.6, 8192 MiB, 120 W limit, SM max 2100 MHz (`gpu-static.txt`) |
| Driver | 615.78.08 |
| CPU | Intel i7-5820K, 6C/12T, AVX2, no AVX-512 |
| Board / RAM | ASUS X99-A, 46 GiB |
| OS | Debian sid, gcc 16.2.0, CUDA 13.4 (V13.4.92) |

## Method

`bench.sh`: `llama-bench -p 512 -n 128 -r 5 -ngl 99`, five cells in fixed order, each started once
the GPU read ≤40 °C, with temperature, SM clock and power recorded before and after
(`results/cells.txt`). The last cell repeats the first. Raw JSON in `results/`, means and standard
deviations in `results/summary.txt`.

## Results

| cell | pp512 t/s | tg128 t/s |
| :--- | ---: | ---: |
| `-fa 1`, f16 KV, `-t 6` (serving config) | 3859.96 ± 111.73 | **116.14** ± 0.25 |
| `-fa 0`, f16 KV, `-t 6` | 3408.19 ± 110.65 | 108.46 ± 0.22 |
| `-fa 1`, q8_0 KV, `-t 6` | 3708.48 ± 118.09 | 107.57 ± 0.23 |
| `-fa 1`, f16 KV, `-t 4` | 3838.78 ± 111.16 | 115.87 ± 0.22 |
| `-fa 1`, f16 KV, `-t 6`, repeat | 3835.54 ± 102.85 | 115.90 ± 0.22 |

- **Flash attention pays:** 1.133x prefill, 1.071x decode. Kept on.
- **q8_0 KV costs:** 0.961x prefill, 0.926x decode. Kept at f16. The direction matches the 1650 Ti,
  which has no tensor cores; this card has them and the KV is still too small for the saving to matter.
- **Threads do not matter:** `-t 4` is 0.995x / 0.998x of `-t 6`. Decode runs on the card.
- **Repeat:** the last cell reads 0.994x / 0.998x of the first. The card held 1890–1905 MHz and
  reached 44 °C at most; no sign of throttling.

## Serving

`make serve` (tpu.env flags: `-ngl 99 -c 8192 -ctk f16 -ctv f16 -fa 1 -t 6 -tb 12 --parallel 1
--reasoning off --metrics`):

- **VRAM: 1510 MiB of 8192** (`serve/vram.txt`). `per_layer_token_embd` (1.321 GB, 51% of the
  file) stays lazy in the mmap, so only 1.20 GiB of weights must be resident (`make info`).
- One chat completion, temperature 0, "In one sentence, what is a TPU?": correct one-sentence answer,
  34 tokens, llama-server timings 102.56 tok/s decode, 272.35 t/s prompt on an 18-token prompt
  (`serve/chat.json`).
- `attest.py`: `device=gpu`, `-ngl 99`, `ggml-cuda`, `libcublas`, `libcuda`, `libcudart` mapped,
  no mismatch (`serve/attest.txt`). `model_server_status` and `gpu_status` answered through the
  MCP server's code path.

## Against the parent

`local-llamacpp-1650ti-2b-q4_0` measured the same file at tg128 81.59–81.92 and pp512 344–345 on
`f95b0d9` (`2026-09-29-exact-gguf-v2-1650ti`). Read as indicative only — different host, different
llama.cpp commit, different day — that puts this card at about **1.42x decode and 11x prefill**.
The prefill gap is far wider than the decode gap. A likely reading, untested here: prefill is
matmul work, which this card can run on tensor cores and the 1650 Ti cannot, while single-stream
decode is limited by memory bandwidth and kernel launches, where the two cards are closer.

## Not measured

Concurrency, context sweep (`sweep.py`), `GGML_CUDA_FORCE_MMQ`, n-gram speculative decoding, and a
CPU-only arm on this host.
