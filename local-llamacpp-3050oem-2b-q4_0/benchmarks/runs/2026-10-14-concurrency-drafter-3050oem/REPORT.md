# 2026-10-14 — concurrency with and without the drafter, RTX 3050 OEM, E2B and E4B

**Question:** the rigs serve with the exact Q4_0 drafter at 2 drafted tokens. Every drafter
speed figure so far was one request at a time. What happens when several users share the card?

**Answer: the drafter is a clear win up to 8 requests at once and shrinks after.** E2B is level
at 32–64 requests (0.95–1.10x); E4B still gains 1.09–1.16x at 32 (64 did not fit with the
drafter). Time to first token is the same or lower with the drafter.

| drafter / no drafter | c=1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E2B, 512 in / 128 out | 1.56 | 1.19 | 1.27 | 1.29 | 1.05 | 0.99 | 0.95 |
| E2B, 128 in / 512 out | 1.79 | 1.33 | 1.44 | 1.53 | 1.21 | 1.10 | 1.00 |
| E4B, 512 in / 128 out | 1.86 | 1.39 | 1.33 | 1.40 | 1.17 | 1.09 | — |
| E4B, 128 in / 512 out | 2.06 | 1.46 | 1.45 | 1.77 | 1.30 | 1.16 | — |

**Read the ratios for their trend, not their size.** `sweep.py`'s prompt is one paragraph about
the model repeated to length, then "Summarize the text above." The drafter guesses that kind of
reply unusually well: acceptance over each drafted sweep was 0.745 / 0.829 (E2B) and 0.786 /
0.726 (E4B), from the server's counters, against 0.475 / 0.481 on the six mixed prompts of
`2026-10-14-drafter-gaps-3050oem`. So one request runs 1.56–2.06x here against 1.40x / 1.72x on
mixed prompts; the gain at higher concurrency on real traffic is likely smaller than the table
by a similar factor (inferred, not measured).

**E4B with the drafter does not fit 64 slots** (`-c 65536 --parallel 64`) in 8 GiB: no drafter
used 7,798 MiB there; the drafted server failed to start and ran at `-c 32768 --parallel 32`
(1,024 tokens per slot either way), 6,146 MiB. E2B: 3,854 MiB without, 4,132 with, at 64 slots.

**No serving change**: the rigs serve one user, where the drafter is worth 1.4–1.7x.

## Method

`run.sh` (job A of `../2026-10-14-overnight-3050oem`), llama.cpp `77c4540ce`, 22:00–22:59.
One llama-server per (model, arm) on port 8080 with the rigs' flags (`-fa 1`, f16 KV,
`per_layer_token_embd` on the card, `--reasoning off`, `--metrics`) at `-c 65536 --parallel 64`;
drafted arms add `-md <exact Q4_0 drafter> --spec-type draft-mtp -ngld all --spec-draft-n-max 2`.
`sweep.py` as in `2026-10-13-concurrency-3050oem`: c = 1, 2, 4, 8, 16, 32, 64, 5 repeats per
level, unique prompts with the prompt cache defeated, two shapes. Aggregate output is total
tokens over wall time, median of the repeats. Stream-gap TPOT is not used: accepted drafted
tokens arrive together, which bends gap timing. `summarize.py` → `summary.txt`.

## Aggregate output tok/s (spread over 5 repeats)

E2B, 512 in / 128 out:

| c | no drafter | drafter | ratio |
|---:|---:|---:|---:|
| 1 | 93.7 (0.4%) | 146.4 (3.2%) | 1.56x |
| 2 | 158.1 (0.7%) | 188.4 (1.2%) | 1.19x |
| 4 | 197.8 (0.9%) | 251.8 (2.6%) | 1.27x |
| 8 | 225.9 (22.0%) | 290.9 (23.0%) | 1.29x |
| 16 | 317.8 (26.1%) | 335.1 (16.0%) | 1.05x |
| 32 | 347.7 (14.7%) | 344.5 (8.2%) | 0.99x |
| 64 | 361.1 (1.0%) | 342.9 (3.8%) | 0.95x |

E2B, 128 in / 512 out:

| c | no drafter | drafter | ratio |
|---:|---:|---:|---:|
| 1 | 104.7 (0.1%) | 187.3 (2.4%) | 1.79x |
| 2 | 189.7 (0.2%) | 252.6 (1.2%) | 1.33x |
| 4 | 258.8 (4.7%) | 372.7 (0.6%) | 1.44x |
| 8 | 309.7 (0.7%) | 473.0 (1.0%) | 1.53x |
| 16 | 508.8 (0.6%) | 613.5 (1.8%) | 1.21x |
| 32 | 590.1 (0.5%) | 650.7 (1.1%) | 1.10x |
| 64 | 625.1 (0.5%) | 622.1 (0.4%) | 1.00x |

E4B, 512 in / 128 out:

| c | no drafter | drafter | ratio |
|---:|---:|---:|---:|
| 1 | 50.8 (0.2%) | 94.4 (2.6%) | 1.86x |
| 2 | 90.0 (0.5%) | 124.7 (3.2%) | 1.39x |
| 4 | 125.2 (0.9%) | 165.9 (35.8%) | 1.33x |
| 8 | 144.4 (22.9%) | 202.5 (30.6%) | 1.40x |
| 16 | 199.3 (26.1%) | 233.4 (19.2%) | 1.17x |
| 32 | 221.6 (15.1%) | 241.6 (1.3%) | 1.09x |
| 64 | 237.9 (0.9%) | — | |

E4B, 128 in / 512 out:

| c | no drafter | drafter | ratio |
|---:|---:|---:|---:|
| 1 | 56.4 (0.1%) | 116.2 (1.9%) | 2.06x |
| 2 | 108.1 (5.9%) | 158.2 (6.1%) | 1.46x |
| 4 | 168.3 (5.4%) | 243.5 (1.0%) | 1.45x |
| 8 | 190.6 (6.2%) | 337.4 (3.4%) | 1.77x |
| 16 | 333.9 (8.6%) | 432.4 (4.1%) | 1.30x |
| 32 | 404.2 (10.3%) | 466.7 (4.2%) | 1.16x |
| 64 | 443.7 (3.1%) | — | |

- The 512-in shape spreads 15–36% at c = 4–32 on both arms (as in the 2026-10-13 sweep); the
  128-in shape stays near or under 10%.
- Time to first token, median: the drafted arm is never more than 2.4% slower (E2B 3,933 against
  3,841 ms at c=64, 128 in) and often faster, most at c=32 (E2B 4,522 against 5,265 ms; E4B 7,551
  against 9,289 at 512 in) and on E4B's 128-in shape (176 against 215 ms at c=1).
- E2B no-drafter numbers reproduce the 2026-10-13 sweep (c=1 93.7 against 93.28, c=64 361.1
  against 358.05), though that sweep had `per_layer_token_embd` on the host.
