# 2026-10-14 — Q8_0, exact Q4_0 and bf16 drafters, GTX 1650 Ti Max-Q

**Answer: the exact Q4_0 drafter is as fast as Q8_0 on this card too, and bf16 is slower, at the
same acceptance.** Same result as the RTX 3050 run: drafter precision does not change which
tokens are accepted, only what each drafted step costs.

## Setup

`run.sh` on the laptop `yogagpu` (GTX 1650 Ti Max-Q, 4096 MiB, driver 615.71.09), llama-server
build 318 (`f95b0d9`), run locally.

- **Drafters**: the six files built and hashed in
  `local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-drafter-q4exact-3050oem/`, copied
  from the 3050 host; all six pass that run's `sha256.txt` (`sha256.log`).
- **Targets and flags** as in `2026-10-14-drafter-1650ti`: each rig's `tpu.env` (exact Q4_0 E2B
  and E4B), `-c 8192 --parallel 1`, six prompts, greedy ×3 plus one sampled pass (temperature
  0.8, seed 1234), 256 tokens, prompt cache off, card cooled to ≤50 °C before each arm.
- **Draft lengths**: each rig's serving setting (E2B 1, E4B 2), plus E2B at 2 to line up with
  the 3050. No drafter and Q8_0 run first and again last.

## Results (`e2b/summary.txt`, `e4b/summary.txt`)

| greedy, geometric mean of 6 prompts | E2B tok/s | x | acceptance | E4B tok/s | x | acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no drafter | 78.9 | 1.000 | — | 38.4 | 1.000 | — |
| Q8_0, 1 token | 92.1 | 1.167 | 0.614 | | | |
| exact Q4_0, 1 token | **92.7** | **1.175** | 0.619 | | | |
| bf16, 1 token | 84.6 | 1.072 | 0.614 | | | |
| Q8_0, 2 tokens | 86.2 | 1.092 | 0.490 | 53.8 | 1.401 | 0.493 |
| exact Q4_0, 2 tokens | 87.3 | 1.107 | 0.495 | **54.5** | **1.419** | 0.491 |
| bf16, 2 tokens | 76.1 | 0.964 | 0.493 | 49.8 | 1.296 | 0.502 |
| Q8_0, repeated last | 92.2 | 1.169 | 0.614 | 53.7 | 1.398 | 0.488 |
| no drafter, repeated last | 78.2 | 0.992 | — | 38.7 | 1.008 | — |

Sampled: E2B at 1 token Q8_0 1.173x, exact Q4_0 1.165x, bf16 1.075x; E4B at 2 tokens 1.359x,
1.403x, 1.286x.

- **Exact Q4_0 against Q8_0**: 1.007x and 1.013x on E2B (1 and 2 tokens), 1.013x on E4B. The Q8_0
  arm repeated last is within 0.2% of the first.
- **bf16 against Q8_0**: 0.919x and 0.883x on E2B, 0.926x on E4B. On E2B at 2 tokens bf16 is
  slower than no drafter (0.964x); on the 3050 it stayed ahead (1.231x).
- **Acceptance** is within 0.012 across the three precisions at every length, on both models.
- **VRAM at `-c 8192`**: E2B 1,485 / 1,595 / 1,557 / 1,663 MiB and E4B 2,853 / 2,965 / 2,927 /
  3,035 MiB for none / Q8_0 / Q4_0 / bf16. The exact Q4_0 drafter uses 38 MiB less than Q8_0 and
  106–108 MiB less than bf16.
- **Spread**: within-cell spread is under 3% on every E2B arm; E4B's drafted arms reach 6.0–12.9%,
  as in `2026-10-14-drafter-1650ti`. The card ran 48–64 °C.

## Greedy text

- **E2B**: at a given draft length, Q8_0, exact Q4_0 and bf16 give the same 18 greedy replies.
  1 and 2 drafted tokens give different replies from each other, and both differ from no drafter
  on all 18. The two no-drafter arms match on all 18. So E2B's text depends on whether and how far
  it drafts, never on the drafter's precision.
- **E4B** repeats differently at temperature 0 with or without a drafter on this card (17 of 18
  no-drafter replies differ between the first and last arm), so its text cannot be compared
  across arms here.

## Against the 3050 (indicative: different card, host and llama.cpp build)

| 2 drafted tokens, x vs no drafter | 1650 Ti E2B | 3050 E2B | 1650 Ti E4B | 3050 E4B |
| --- | ---: | ---: | ---: | ---: |
| Q8_0 | 1.092 | 1.390 | 1.401 | 1.724 |
| bf16 | 0.964 | 1.231 | 1.296 | 1.585 |

The ordering is the same on both cards: exact Q4_0 level with Q8_0, bf16 last.
