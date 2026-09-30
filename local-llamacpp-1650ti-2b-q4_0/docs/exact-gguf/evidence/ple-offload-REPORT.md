# 2026-09-29 — `per_layer_token_embd` in VRAM vs in the mmap, GTX 1650 Ti

**Question:** v2 of the exact GGUF stores `per_layer_token_embd` at Q4_0 (1.32 GB, against 1.93 GB
at Google's Q6_K), so for the first time the whole model fits on the card. Does keeping it there
help?

**Answer: no.** It fits and it costs 1252 MiB of VRAM, and decode and prefill are unchanged within
0.7%.

**Setup:** v2 exact GGUF (`419db9a6…`), this rig's binary (llama.cpp `f95b0d9`), `-ngl 99 -fa 1
-t 6`. The only difference between the arms is `-ot 'per_layer_token_embd\.weight=CUDA0'`.
`ple_ab.sh` → `ple-ab.md`; allocation logs in `alloc-{host,card}.log`.

## It really moved

From llama-server's allocation log with the `make serve` flags:

| buffer | host (default) | card (`-ot`) |
|---|---:|---:|
| CUDA0 model | 1223.91 MiB | **2483.91 MiB** (+1260) |
| CPU_Mapped model | 1476.00 MiB | 216.00 MiB |
| CUDA0 KV | 48 + 12 MiB | 48 + 12 MiB |
| CUDA0 compute | 122.52 MiB | 115.52 MiB |
| `nvidia-smi` process | 1480 MiB | **2732 MiB** of 3732 usable |

The 216 MiB left on the host is `token_embd`'s input-lookup copy, which llama.cpp keeps on the
CPU by design.

## It does nothing for speed

llama-bench, r=5, ABBA (host, card, card, host), with the card cooled to ≤50 °C before each pass:

| | host p1 | card p2 | card p3 | host p4 |
|---|---:|---:|---:|---:|
| pp512 | 347.96 | 345.68 | 345.84 | 345.00 |
| pp2048 | 319.43 | 319.57 | 319.68 | 319.36 |
| tg128 | 82.17 | 82.16 | 82.05 | 81.96 |

Host mean against card mean: pp512 +0.2%, pp2048 −0.1%, tg128 −0.0%. Pass 1 started with the
card still clocked up from the previous job (1890 MHz at 44 °C), which is the most likely reason
its pp512 is the highest cell.

## Why

The table is read by `GGML_OP_GET_ROWS`, one row of 8960 values (about 5 KB at Q4_0) per token.
Decode streams about 1.2 GB of weights per token, so the lookup is noise, and even a
2048-token prompt fetches only about 10 MB of rows. Moving the table saves a host-side gather
and copy that was never on the critical path.

## Consequence

Leave it lazy. The KV cache was never what was at stake here: at `-c 8192` it is 60 MiB, because
the 12 sliding layers are capped at their 1024-cell window. The 1.25 GiB freed by keeping the
table in the mmap would matter only for a much longer context or many parallel slots. Neither
binds on this card for this model; `MODELS.md` has the KV geometry.
