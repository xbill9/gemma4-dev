# SGLang 0.5.23 accuracy with the drafter on the RTX 3050 OEM — 2026-11-04

The GSM8K and BFCL simple run of `../2026-11-04-accuracy-3050oem` repeated with Google's E2B QAT drafter as W4A16
at 4 drafted tokens, SGLang's best for one request (`../2026-11-04-engines-3050oem`). Four drafted tokens load
only with 16 running requests; this run sends one at a time. Same harness, sets, limits and thinking off.
`run.sh` drives it; `compare.py` prints `COMPARE.txt`.

#### Result

| | GSM8K (1,319) | BFCL simple (400) |
|---|---|---|
| SGLang, drafter at 4 tokens | 0.8992 (1,186) | 0.9175 (367) |
| SGLang, no drafter | 0.8992 (1,186) | 0.915 (366) |
| vLLM, the same drafter at 4 tokens | 0.8992 (1,186) | 0.9225 (369) |

**The drafter changes no score.** Paired against SGLang without it: GSM8K +0.0000 [−0.0076, +0.0083], BFCL
+0.0025 [+0.0000, +0.0075] (one question gained, none lost; 398 of 400 replies identical). Against vLLM with the
same drafter: GSM8K +0.0000, BFCL −0.0050 [−0.0150, +0.0050], level.

#### Wall time

BFCL finished 1.77x sooner than without the drafter (75.0 against 132.6 s) and took 1.30x vLLM's time with its
drafter. The GSM8K wall time (2,601 s) overlapped an Android build on the same host, 12 jobs and 46 GB of RAM
from about 15:54, so it is not a speed result.

#### Files

`gsm8k-sglang-w4a16n4.{jsonl,log}`, `bfcl_simple-sglang-w4a16n4.{jsonl,log}`, `driver.log`, `server.log`,
`command.txt`.
