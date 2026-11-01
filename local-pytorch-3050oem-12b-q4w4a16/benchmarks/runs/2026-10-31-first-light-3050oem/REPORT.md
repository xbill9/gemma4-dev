# First light: 12B int4 PyTorch on the RTX 3050 OEM, 2026-10-31

Engine `e12b_engine.py`, torch 2.14.1+cu130, transformers 5.16.1, driver 615.78.08. fp16 compute, greedy.

| Measure | Result | Source |
|---|---|---|
| Weights | 6,388 MiB | `short.jsonl` |
| Load | 277 s (16k cache), 295 s (8k cache) | `short.jsonl`, `short8k.jsonl` |
| Decode at 65-token prompt | 27.48 tok/s (16k cache), 27.42 (8k cache) | same |
| Prompt read at 65 tokens | 172 tok/s | `short.jsonl` |
| Peak memory | 7,275 MiB (16k cache), 7,147 (8k cache) | same |
| One decode step, profiled | 36.271 ms, 27.6 tok/s | `profile.json` |

Code at 65 tokens was recalled correctly in both runs.

llama.cpp `7182ef71e` on this card (2026-10-16, a different date and engine run): 28.6 tok/s without a drafter, 56.4 with.
So decode is 0.96x without a drafter and 0.49x with one.

**Depth was not measured.** The 4,096-token prompt ran out of memory at `MAX_CTX` 8192 (`short8k.log`: 7.13 GiB allocated,
16 MiB requested, 9 MiB free) and at 16384, whose log was not kept. `MAX_CTX` 32768 ran out on the short prompt, also with
no log kept. llama.cpp holds 131,072 tokens of this model on the same card without a drafter.
