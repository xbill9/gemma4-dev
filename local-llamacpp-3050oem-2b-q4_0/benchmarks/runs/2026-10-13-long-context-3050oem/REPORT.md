# 2026-10-13 — single-user context length, RTX 3050 OEM

**Question:** this rig serves one user. The model was trained to 131,072 tokens (`n_ctx_train`
in the load log) and the rig served 8,192. What does the full context cost?

**Answer: 840 MiB, and nothing for short chats.** Adopted: `CONTEXT_SIZE=131072`.

## Memory

`run.sh`, serving flags with `per_layer_token_embd` on the card (`alloc-c*.log`):

| `-c` | KV, full-attention layers | KV, sliding layers | compute | `nvidia-smi` |
| ---: | ---: | ---: | ---: | ---: |
| 8,192 | 48 MiB | 12 MiB | 128.02 MiB | 2770 MiB |
| 32,768 | 192 MiB | 12 MiB | 152.02 MiB | 2938 MiB |
| 131,072 | 768 MiB | 12 MiB | 248.02 MiB | **3610 MiB** of 8192 |

Only the 3 full-attention layers grow with context. The 12 sliding-window layers stay at
1,024 cells.

## Short chats are unaffected

`short_chat.sh`: the same request six times against a server at each `-c`, exactly 256 tokens
generated (`ignore_eos`), prompt cache off. Runs 2–6 (run 1 of each is the cold first request):

| `-c` | decode tok/s, mean | runs |
| ---: | ---: | --- |
| 8,192 | 108.30 | 108.36 / 108.02 / 108.32 / 108.48 / 108.31 |
| 131,072 | 107.99 | 107.87 / 107.85 / 107.98 / 108.09 / 108.17 |

0.997x. Raw lines in `short_chat.txt`.

## Long conversations slow down

llama-bench, 512-token prompt and 128 generated tokens on top of `d` tokens already in the
cache, r=3 (`results.md`):

| tokens already in context | prompt t/s | vs empty | generate tok/s | vs empty |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 3959.63 | 1.000 | 116.02 | 1.000 |
| 8,192 | 2898.48 | 0.732 | 107.06 | 0.923 |
| 32,768 | 1690.19 | 0.427 | 90.68 | 0.782 |
| 65,536 | 1079.92 | 0.273 | 75.02 | 0.647 |
| 120,000 | 669.25 | 0.169 | 58.14 | 0.501 |

Generation stays above 100 tok/s up to roughly 8k tokens of conversation and halves by 120k.
The GPU ended at 61 °C, 1882 MHz.
