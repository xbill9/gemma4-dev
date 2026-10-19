# What loads: llama-server, Gemma 4 26B A4B QAT Q4_0 (Google's GGUF), RTX 3050 OEM 8 GiB

2026-10-18, llama.cpp 7182ef71e. `fit.sh N CTX KV`: llama-server -ngl 99 -ncmoe N -c CTX
-ctk KV -ctv KV -fa 1 -t 8 -tb 12 --parallel 1 --reasoning off, no drafter, default -b/-ub.
OK = /health answered; VRAM = nvidia-smi memory.used while loaded. FAIL = the first
`cudaMalloc failed` line in fit-n<N>-c<CTX>-<KV>.log (the allocation that did not fit).

| N (expert layers in RAM) | ctx | KV | result |
|---:|---:|---|---|
| 17 | 32768 | f16 | FAIL 474.89 MiB (compute buffer) |
| 18 | 32768 | f16 | FAIL 474.89 MiB |
| 18 | 131072 | f16 | FAIL 2560.00 MiB (KV) |
| 18 | 262144 | f16 | FAIL 5120.00 MiB (KV) |
| 18 | 32768 | q4_0 | OK 7256 MiB |
| 18 | 65536 | q4_0 | OK 7468 MiB |
| 18 | 98304 | q4_0 | OK 7808 MiB |
| 19 | 32768 | f16 | OK 7520 MiB |
| 19 | 131072 | f16 | FAIL 2560.00 MiB |
| 19 | 65536 | q8_0 | OK 7452 MiB |
| 19 | 131072 | q8_0 | FAIL 826.91 MiB |
| 19 | 65536 | q4_0 | OK 7058 MiB |
| 19 | 131072 | q4_0 | OK 7738 MiB |
| 19 | 262144 | q4_0 | FAIL 1466.91 MiB |
| 20 | 32768 | f16 | OK 7110 MiB |
| 20 | 131072 | f16 | FAIL 2560.00 MiB |
| 20 | 65536 | q8_0 | OK 7042 MiB |
| 20 | 131072 | q8_0 | FAIL 826.91 MiB |
| 20 | 65536 | q4_0 | OK 6648 MiB |
| 20 | 131072 | q4_0 | OK 7328 MiB |
| 20 | 262144 | q4_0 | FAIL 1466.91 MiB |
| 21 | 32768 | f16 | OK 6702 MiB |
| 21 | 131072 | f16 | FAIL 300.00 MiB |
| 21 | 65536 | q8_0 | OK 6636 MiB |
| 21 | 131072 | q8_0 | OK 7636 MiB |
| 21 | 262144 | q8_0 | FAIL 2720.00 MiB |
| 21 | 65536 | q4_0 | OK 6242 MiB |
| 21 | 131072 | q4_0 | OK 6922 MiB |
| 21 | 262144 | q4_0 | FAIL 1466.91 MiB |
| 22 | 65536 | q8_0 | OK 6228 MiB |
| 22 | 131072 | q8_0 | OK 7228 MiB |
| 22 | 262144 | q8_0 | FAIL 1466.91 MiB |
| 22 | 65536 | q4_0 | OK 5834 MiB |
| 22 | 131072 | q4_0 | OK 6514 MiB |
| 22 | 262144 | q4_0 | OK 7874 MiB |
| 23 | 65536 | q8_0 | OK 5820 MiB |
| 23 | 131072 | q8_0 | OK 6820 MiB |
| 23 | 262144 | q8_0 | FAIL 1466.91 MiB |
| 23 | 65536 | q4_0 | OK 5426 MiB |
| 23 | 131072 | q4_0 | OK 6106 MiB |
| 23 | 262144 | q4_0 | OK 7466 MiB |

Also: N=17 at 65536/131072/262144 f16 and 131072/262144 q8_0 fail. Earlier N=17 attempts'
FAIL text in the console was a `common_fit_params` warning line; the logs show the real
failure was the 474.89 MiB compute buffer (fit.sh's grep was then corrected).

The f16 KV failures read 2560 MiB at 131072 and 5120 at 262144: 20 KiB/token for the
full-attention layers. The compute buffer grows with ctx (474.89 MiB at 32768, 1466.91 at 262144).
Expert weights per layer: 12,251 MiB / 30 = 408 MiB (gguf reader over the file's tensors).
