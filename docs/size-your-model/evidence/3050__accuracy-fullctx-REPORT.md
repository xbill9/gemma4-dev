# 2026-10-15 — answer quality as the rigs serve: full context, one request at a time, RTX 3050 OEM

**Question:** every accuracy run on these rigs so far used `-c 32768 --parallel 8` with 8
requests at once. The rigs serve one user at `-c 131072 --parallel 1`. Are the scores the same
there, with and without the drafter?

**Answer: yes.** Every difference against the 8-at-once runs, and every drafter-against-none
difference, has a 95% range that includes zero. The drafter finishes GSM8K 2.23x faster on E4B
and 1.76x on E2B, one request at a time.

| arm | GSM8K, full context, 1 at a time | at `-c 32768`, 8 at a time | BFCL simple, full / 8 at a time |
|---|---:|---:|---:|
| E4B, drafter (serving setting) | 0.9295 | 0.9325 | 0.9125 / 0.9150 |
| E4B, no drafter | 0.9310 | 0.9340 | 0.9125 / 0.9100 |
| E2B, drafter (serving setting) | 0.8969 | 0.9022 | 0.9250 / 0.9225 |
| E2B, no drafter | 0.9014 | 0.8984 | 0.9225 / 0.9250 |

- **Full context against 8 at a time**, GSM8K: E4B −0.0030 (both arms), E2B −0.0053 (drafter)
  and +0.0030 (none). Widest range: E2B drafter, −0.0159 to +0.0053.
- **Drafter against none at full context**: E4B GSM8K −0.0015 (range −0.0053 to +0.0015), BFCL
  0.0000; E2B GSM8K −0.0045 (−0.0159 to +0.0068), BFCL +0.0025.
- **One request at a time, E4B's drafted replies match no drafter word for word on 1,200 of
  1,319 GSM8K problems and all 400 BFCL questions**, against 366 of 1,319 between two no-drafter
  runs at 8 at a time. That points to 8 requests sharing the card as the main source of the earlier
  runs' word-level variation (inferred; no one-at-a-time repeat was run). E2B's drafted replies match on 162 and 399, as expected: E2B's
  greedy text changes with drafting even on short prompts.

**Time** (`wall_s`, one request at a time):

| | E4B no drafter | E4B drafter | E2B no drafter | E2B drafter |
|---|---:|---:|---:|---:|
| GSM8K, 1,319 problems | 130.9 min | 58.7 min (2.23x) | 83.2 min | 47.2 min (1.76x) |
| BFCL simple, 400 | 4.6 min | 2.6 min (1.81x) | 2.5 min | 1.6 min (1.56x) |

**What this means:** the scores measured at 8 at a time stand for the serving setup, and the
drafter costs nothing in quality where the rigs actually serve.

## Method

`run.sh` (job B of `../2026-10-14-overnight-3050oem`), 01:03–06:41. llama.cpp `77c4540ce`,
`-c 131072 --parallel 1`, `-fa 1`, f16 KV, `per_layer_token_embd` on the card, `--reasoning off`;
drafted arms add the exact Q4_0 drafter at 2 tokens, the rigs' setting. `jev-tpu-v5e1/gen_eval.py`
at concurrency 1, GSM8K 1,319 at 2,048 tokens and BFCL simple 400, prompt cache at the server
default (on). Order: E4B drafter, E4B none, E2B drafter, E2B none; each arm was committed as it
finished. The 8-at-once comparison is `2026-10-14-drafter-q4exact-3050oem/accuracy`. `compare.py`
→ `COMPARE.txt` (95% ranges from a paired bootstrap, 2,000 resamples).
