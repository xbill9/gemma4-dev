# 2026-10-13 — first light, E4B on the RTX 3050 OEM

`local-llamacpp-3050oem-4b-q4_0`: the exact Q4_0 E4B GGUF (`xbill9/gemma-4-E4B-it-qat-q4_0-exact-gguf`,
revision `8b7c081e`, SHA-256 `5462cc10…`, checked after download), llama.cpp `77c4540ce`, the same
binary, card and host as the E2B sister rig `local-llamacpp-3050oem-2b-q4_0`, measured the same
day. Script: `bench.sh`; raw output `results.md`; allocation logs `alloc-*.log`.

## Memory at the serving context (-c 131072)

`make info`: 4.20 GB file, 2.43 GiB must be resident, `per_layer_token_embd` 1.585 GB (38%).

| | table in mmap | table on card | on card + drafter |
| --- | ---: | ---: | ---: |
| CUDA0 model | 2493.32 MiB | 4005.32 MiB | 4005.32 + 79.05 MiB |
| CUDA0 KV (full-attention + sliding) | 2048 + 40 MiB | 2048 + 40 MiB | same |
| CUDA0 compute | 256.02 MiB | 236.02 MiB | 236.02 + 149.00 MiB |
| `nvidia-smi` | 4938 MiB | 6430 MiB | **6662 MiB** of 8192 |

The KV cache at 131,072 tokens is 2,088 MiB, against 780 MiB for E2B: E4B's full-attention KV is
2.67x E2B's at this context.

## Per-layer table placement (llama-bench r=5, ABBA, cooled to ≤40 °C)

| test | mmap (passes 1, 4) | card (passes 2, 3) | card / mmap |
| --- | ---: | ---: | ---: |
| pp512 | 2201.97, 2190.32 | 2245.78, 2245.29 | 1.023 |
| pp2048 | 2177.44, 2179.16 | 2202.28, 2214.74 | 1.014 |
| tg128 | 60.45, 60.57 | 60.46, 60.61 | 1.000 |

Both card passes beat both mmap passes on prefill; decode is level. Adopted, as for E2B, at a cost
of 1,492 MiB.

## Speed by conversation length (table on the card, r=3)

| tokens already in context | prompt t/s | generate tok/s | E4B / E2B, prompt | E4B / E2B, generate |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2230.50 | 60.41 | 0.563 | 0.521 |
| 8,192 | 1860.37 | 56.00 | 0.642 | 0.523 |
| 32,768 | 1263.10 | 47.05 | 0.747 | 0.519 |
| 65,536 | 883.36 | 38.80 | 0.818 | 0.517 |
| 120,000 | 591.45 | 30.04 | 0.884 | 0.517 |

E2B figures from `local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-13-long-context-3050oem`,
same binary, card and day. Without drafting, E4B generates at 0.52x E2B's speed at every length;
its prompt-reading gap narrows as the conversation grows. GPU ended at 59 °C, 1882 MHz.
