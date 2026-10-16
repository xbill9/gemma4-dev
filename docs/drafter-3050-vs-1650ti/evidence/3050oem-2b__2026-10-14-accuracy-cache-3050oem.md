# 2026-10-14/15 — does the prompt cache change answer quality? RTX 3050 OEM, E2B and E4B

**Question:** `2026-10-14-drafter-divergence-3050oem` showed that llama.cpp's prompt cache
changes greedy text on its own: a prompt partly served from the cache reads differently from the
same prompt read in full. Every accuracy run on these rigs used the server default, cache on.
Does turning it off change the scores?

**Answer: no.** All eight paired differences (2 models × drafter off/on × GSM8K/BFCL) have a 95%
range that includes zero, and the number of problems that flip is no larger than between two
cache-on runs.

| arm | GSM8K, cache on → off | BFCL simple, cache on → off |
|---|---|---|
| E4B, no drafter | 0.9340 → 0.9333 (−0.0008) | 0.9100 → 0.9125 (+0.0025) |
| E4B, drafter | 0.9325 → 0.9325 (0.0000) | 0.9150 → 0.9125 (−0.0025) |
| E2B, no drafter | 0.8984 → 0.8901 (−0.0083) | 0.9250 → 0.9200 (−0.0050) |
| E2B, drafter | 0.9022 → 0.8992 (−0.0030) | 0.9225 → 0.9250 (+0.0025) |

- **Flips against run-to-run variation.** GSM8K problems scored differently, cache off against
  cache on: E4B 25 (no drafter) and 20 (drafter); E2B 51 and 28. Two cache-on no-drafter runs a
  day apart differed on 20 (E4B) and 59 (E2B) (`2026-10-14-drafter-q4exact-3050oem`). The cache
  adds nothing measurable to what 8 requests sharing the card already do.
- **The widest gap is E2B GSM8K without the drafter**, −0.0083, range −0.0190 to +0.0023.
- **BFCL barely moves**: 1 to 4 of 400 questions per arm, and 396–398 replies word-for-word the same.
- **Word-for-word identical GSM8K replies**: E4B 349 (no drafter) and 735 (drafter) of 1,319;
  E2B 187 and 479.

**What this means:** past and future accuracy runs at the server default (cache on) stand. The
cache matters only where byte-identical greedy text is the point.

## Method

`run.sh` (job C of `../2026-10-14-overnight-3050oem`), 23:00–01:03. Flags identical to
`2026-10-14-drafter-q4exact-3050oem/accuracy`: llama.cpp `77c4540ce`, `-c 32768 --parallel 8`,
`-fa 1`, f16 KV, `per_layer_token_embd` on the card, `--reasoning off`; drafted arms add the exact
Q4_0 drafter at 2 tokens. `jev-tpu-v5e1/gen_eval.py` at concurrency 8, GSM8K 1,319 problems at
2,048 tokens and BFCL simple 400, with `JEV_CACHE_PROMPT=0`, which adds `"cache_prompt": false`
to every request. Each arm pairs record for record with its cache-on twin in that run.
`compare.py` → `COMPARE.txt` (95% ranges from a paired bootstrap, 2,000 resamples).
