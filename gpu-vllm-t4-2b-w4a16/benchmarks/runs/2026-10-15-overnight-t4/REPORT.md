# 2026-10-15 — T4 drafter gaps: several requests at once, GSM8K, W4A16 drafters

**Answers:**

1. **The drafter's gain falls as requests share the card, faster for E2B.** At the serving
   setting (QAT drafter, 3 tokens), aggregate output against no drafter: E2B 1.28x / 1.09x /
   1.01x at 1 / 4 / 8 requests in flight; E4B 1.59x / 1.43x / 1.17x. The rigs serve up to 8.
2. **The drafter leaves GSM8K unchanged on E2B**: 1,188 of 1,319 right with and without it
   (0.9007), 8 problems right only without and 8 only with; 1,002 of 1,319 completions identical.
   It finished 1.20x faster at 8 at a time (958.8 s against 1,147.6 s).
3. **The W4A16 drafters serve and match the bf16 QAT drafter, without being faster** (results
   in `../2026-10-15-drafter-sweep-t4/e?b-q4ct/`, see that run's REPORT.md).

## Method

`trimmed.sh`, then `trimmed2.sh` (`trimmed.log`), on the T4 of the GCE `n1-standard-2` VM, vLLM
0.29.0, each rig's `tpu.env` serving flags via `t4serve.py`. The card was cooled to 50 °C before
every server in steps 1 and 2. `trimmed.sh` ran its first job (the W4A16 drafters) without the
cool-down; it was stopped there and `trimmed2.sh` ran the rest with it, because the no-drafter
server run first on a 52–63 °C card decoded 113.8 tok/s and the same server run last at 71–81 °C
104.8 tok/s (studied in its own repo, `xbill9/t4-thermal-drift`).

- **`conc.py`** (`conc-e?b/`): per arm one server; at 1, 4 and 8 requests in flight,
  max(16, 4 x C) greedy chat requests of 256 tokens drawn in a fixed order from 26 real prompts;
  one pass per cell (`CONC=1,4,8 REPEATS=1`). Aggregate output tok/s is output tokens over the
  cell's wall time; acceptance from the server's spec-decode counters.
- **`quality.py`** (`quality/`): GSM8K, all 1,319 test problems, `jev-tpu-v5e1/gen_eval.py`,
  2,048-token limit, greedy, 8 at a time, E2B only (`TASKS=gsm8k`), no drafter then QAT at 3
  tokens. Scoring is gen_eval.py's own `right` field. BFCL, E4B and the Nimble suite were cut
  from the plan to save time.
- `conc.py`'s and `longctx.py`'s full plans and `queue.sh` remain from the larger plan; only the
  steps above ran. `longctx.py`, `build_suite.sh` and the bf16-target job have not run.

## Several requests at once (`conc-e?b/summary.txt`)

| aggregate output tok/s (x vs none) [acceptance] | 1 | 4 | 8 |
| --- | ---: | ---: | ---: |
| E2B no drafter | 109.3 | 339.8 | 576.7 |
| E2B QAT, 3 tokens | 139.5 (1.28x) [0.38] | 369.2 (1.09x) [0.38] | 584.3 (1.01x) [0.38] |
| E4B no drafter | 62.8 | 194.3 | 327.3 |
| E4B QAT, 3 tokens | 100.0 (1.59x) [0.42] | 277.7 (1.43x) [0.42] | 382.4 (1.17x) [0.42] |

- Acceptance does not change with load, so the shrinking gain is the verification step costing
  more once the card is busy, not the drafter guessing worse.
- One pass per cell of 16–32 requests: no repeat spread, so differences of a few percent are not
  resolved.
- The same pattern as the RTX 3050 under llama.cpp (`local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-concurrency-drafter-3050oem`),
  where the gain held to 8 and was level for E2B at 32–64. On the T4 E2B is level already at 8.

## GSM8K, E2B (`quality/gsm8k-e2b-*.jsonl`)

| | right of 1,319 | accuracy | truncated at 2,048 | wall time, 8 at a time | output tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| no drafter | 1,188 | 0.9007 | 3 | 1,147.6 s | 534,578 |
| QAT, 3 tokens | 1,188 | 0.9007 | 2 | 958.8 s | 530,330 |

The 3050's exact Q4_0 E2B scored 0.898 (`local-llamacpp-3050oem-4b-q4_0/benchmarks/runs/2026-10-13-quality-e2b-vs-e4b-3050oem`).
