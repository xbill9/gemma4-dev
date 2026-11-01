# 2026-10-31 — answer quality through the OpenAI server, against llama.cpp on the same card

GSM8K (1,319, 2,048-token limit) and BFCL simple (400) through `jev-tpu-v5e1/gen_eval.py`, greedy, one request at a time,
against `openai_server.py --max-ctx 8192`. Reference: llama.cpp `77c4540ce` on this card with the exact Q4_0 GGUF, no
drafter (`local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-accuracy-fullctx-3050oem`, arm `e2b-none`). Paired with
`jev-tpu-v5e1/gen_compare.py`, `COMPARE.txt`.

| | llama.cpp | PyTorch | Difference (95% range) |
| --- | ---: | ---: | ---: |
| GSM8K | 0.9014 | 0.8992 | −0.002 (−0.014 to +0.008) |
| BFCL simple | 0.922 | 0.9225 | +0.000 (−0.007 to +0.007) |

Level on both. GSM8K took 4,377 s, 1.55x sooner than the 1650 Ti's 6,775 s through the same server. The fix for the
torch 2.13 SDPA fault (K/V copied on sliding layers) stays in; torch 2.14.1 was not tested without it.
