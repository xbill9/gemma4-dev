# 2026-10-26 — what a drafted step costs, and the drafter with several requests at once, NVIDIA L4

**Answer: the card sets the draft length through the cost of a step, not through acceptance.** On all three cards a
step yields the same number of tokens at each draft length (about 1.60 / 1.94 / 2.15 at 1 / 2 / 3 drafted tokens);
what changes is how much longer a drafted step takes than a plain one. On the L4, checking up to 4 drafted tokens costs
the target no more than reading one token (≤1.04x), so nearly all the extra time is the drafter itself, about
0.8–1.2 ms per drafted token. That is a smaller share of an E4B step (13.5 ms) than of an E2B step (7.5 ms), which is
why E4B gains more from the same drafter. With several requests at once the L4 drafter at 2 tokens pays up to 16
requests (E4B 1.15–1.34x, E2B 1.06–1.19x at 2–16) and loses at 64 (0.87x for both).

## Setup

The instance, llama.cpp `77c4540ce` build, models (SHA-256 in `models.sha256`, equal to both rigs' `tpu.env`) and flags
of `../2026-10-26-drafter-g6` and `gpu-llamacpp-g6-4b-q4_0/benchmarks/runs/2026-10-26-drafter-g6`, run after them on
the same spot `g6.2xlarge`. `run.sh` drives everything; `driver.log` has the timeline.

## 1. Tokens per step and cost per step, three cards (`step_cost.py`, `step_cost.txt`)

No new run: the greedy replies of each card's existing speed run (six prompts, 256 tokens, `77c4540ce`), split into
steps. A step is one pass of the target over the last token plus the drafted ones; steps per reply = tokens − accepted
drafted tokens; step time = decode time / steps. Pooled over all replies of an arm.

| card | model | tokens per step, 1 / 2 / 3 drafted | step cost vs plain, 1 / 2 / 3 | best |
| --- | --- | --- | --- | --- |
| GTX 1650 Ti | E2B | 1.596 / 1.924 / 2.116 | 1.352 / 1.768 / 2.119 | 1 token, 1.181x |
| GTX 1650 Ti | E4B | 1.629 / 1.996 / 2.223 | 1.092 / 1.391 / 1.669 | 1 token, 1.491x |
| RTX 3050 OEM | E2B | 1.602 / 1.939 / 2.132 | 1.162 / 1.391 / 1.628 | 2 tokens, 1.394x |
| RTX 3050 OEM | E4B | 1.619 / 1.950 / 2.179 | 1.048 / 1.136 / 1.287 | 2 tokens, 1.717x |
| L4 | E2B | 1.599 / 1.982 / 2.195 | 1.150 / 1.256 / 1.356 | 3 tokens, 1.619x |
| L4 | E4B | 1.613 / 1.952 / 2.174 | 1.116 / 1.185 / 1.248 | 3 tokens, 1.741x |

The speed-up is tokens per step divided by step cost. Tokens per step agree across cards to within 0.03 at 1 drafted token and 0.11 at 3;
step cost differs by up to 1.56x at 3 drafted tokens (E2B, 1650 Ti 2.119 against L4 1.356). Each card's best draft length is where
one more drafted token adds less in tokens per step than it adds in step cost. On the L4 at 4 drafted tokens: E2B 2.174
tokens per step at 1.508x, E4B 2.355 at 1.359x.

## 2. What a step is made of, L4 (`verify/`, `summarize_verify.py`, `verify/summary.txt`)

`llama-bench`, one pass of the target over k = 1–6 tokens at 0 / 8,192 / 32,768 tokens held, the rigs' flags, 10
repeats each.

| k tokens in one pass | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E2B, ms at depth 0 | 7.70 | 7.60 | 7.60 | 7.71 | 8.28 | 8.23 |
| E4B, ms at depth 0 | 13.66 | 13.90 | 14.11 | 14.24 | 14.97 | 15.71 |
| E2B, x vs k = 1 at 0 / 8k / 32k | 1.00 | 0.98–0.99 | 0.99–1.01 | 1.00–1.02 | 1.07–1.08 | 1.07 |
| E4B, x vs k = 1 at 0 / 8k / 32k | 1.00 | 1.01–1.02 | 1.02–1.03 | 1.03–1.04 | 1.07–1.10 | 1.08–1.15 |

Checking up to 3 drafted tokens (k = 4) costs the target at most 4% more than one token, at every depth; the first
clear step is at k = 5. Repeats vary by 0.9–2.4 ms (standard deviation), so differences under about 0.3 ms are within
noise; the pattern holds at all three depths.

The step time of section 1, less the target's pass over 1 + n tokens, leaves the drafter and bookkeeping:

| drafted tokens | E2B step ms | of it target pass | rest | rest per drafted token | E4B step ms | of it target pass | rest | rest per drafted token |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8.62 | 7.60 | 1.02 | 1.02 | 15.08 | 13.90 | 1.18 | 1.18 |
| 2 | 9.41 | 7.60 | 1.81 | 0.91 | 16.02 | 14.11 | 1.90 | 0.95 |
| 3 | 10.17 | 7.71 | 2.46 | 0.82 | 16.87 | 14.24 | 2.63 | 0.88 |
| 4 | 11.30 | 8.28 | 3.02 | 0.75 | 18.37 | 14.97 | 3.40 | 0.85 |

The drafter proposes its tokens one after another, so its time grows with each drafted token: 0.75–1.18 ms each, on
both models. A plain step is 7.50 ms for E2B and 13.51 ms for E4B, so 3 drafted tokens add 33% to an E2B step and 19% to
an E4B step. That difference, not acceptance, is why E4B gains more from its drafter here (1.741x against 1.619x).

The same split on the 3050 and 1650 Ti was not run (the hosts were not reachable from this session); section 1 shows
their step costs but not how those divide.

## 3. Several requests at once, L4 (`mixed/`, `summarize_mixed.py`, `mixed/summary.txt`)

The 3050's `2026-10-15-concurrency-mixed-3050oem` run, `mixed.py` unchanged: six chat prompts, 256 tokens, prompt cache
off, c requests at once, 5 repeats, greedy, one shared KV cache (`-kvu`) at `-c 32768 --parallel 64`. Aggregate output
tok/s, median of repeats; x vs no drafter.

| requests | E2B none | 2 tokens | 3 tokens | E4B none | 2 tokens | 3 tokens |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 131.4 | **1.53x** | 1.59x | 73.5 | 1.63x | **1.74x** |
| 2 | 230.3 | **1.19x** | 1.16x | 129.3 | **1.29x** | 1.28x |
| 4 | 351.1 | **1.13x** | 1.13x | 193.0 | **1.34x** | 1.33x |
| 8 | 505.2 | **1.06x** | 1.03x | 309.7 | **1.15x** | 1.13x |
| 16 | 667.7 | **1.07x** | 0.98x | 441.3 | **1.17x** | 1.13x |
| 32 | 850.9 | 0.93x | 0.87x | 649.9 | 1.00x | 0.95x |
| 64 | 918.9 | 0.87x | 0.80x | 778.1 | 0.87x | 0.81x |

Acceptance holds at 0.46–0.52 (2 tokens) and 0.38–0.42 (3 tokens) at every level. 3 drafted tokens lead only for one
request; from 2 requests on, 2 tokens are level or ahead, and 3 lose more at 32–64. Median time to first token is
lower with the drafter at 2–64 requests for E2B and at 2–16 and 64 for E4B (for example E2B at 8 requests: 511 ms none, 286 ms
with 2 tokens). Spreads between repeats are 16–43% at 2–4 requests and under 10% at 16–64; the one-request
drafter cells spread 43–64% because the six prompts differ in acceptance.

Against the 3050 run (same script, same prompts, 2 drafted tokens), x vs none at 2 / 4 / 8 / 16 / 32 / 64 requests:

| | 3050 | L4 |
| --- | --- | --- |
| E2B | 0.93 / 1.09 / 1.17 / 0.95 / 0.86 / 0.79 | 1.19 / 1.13 / 1.06 / 1.07 / 0.93 / 0.87 |
| E4B | 1.09 / 1.18 / 1.24 / 1.05 / 0.92 / 0.79 | 1.29 / 1.34 / 1.15 / 1.17 / 1.00 / 0.87 |

On the L4 the drafter is ahead at every level from 1 to 16 requests on both models; on the 3050, E2B fell behind at 2
and 16 requests and E4B at 32.

## Notes

- `verify e4b` ran with the previous E2B server still loaded and idle on the card (3,836 MiB used at its start); it
  served no requests during the run.
- The cool-down before each step waits up to 90 s for ≤45 °C; the card started steps at 47–67 °C.

## Files

`step_cost.py`, `step_cost.txt`, `step_cost.json`; `verify/{e2b,e4b}.jsonl` (llama-bench), `verify/*.log`,
`summarize_verify.py`, `verify/summary.txt`; `mixed/<model>/<arm>/` (results, greedy log, server log, `/metrics` at
the end), `mixed.py`, `summarize_mixed.py`, `mixed/summary.txt`; `run.sh`, `driver.log`, `run.out`, `models.sha256`.
