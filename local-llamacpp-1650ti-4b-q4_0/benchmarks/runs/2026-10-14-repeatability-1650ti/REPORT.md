# 2026-10-15 — why E4B's greedy replies change between identical requests, GTX 1650 Ti

**Question:** the 2026-10-14 1650 Ti drafter runs found E4B giving 2–3 different texts over 3
repeats of a prompt at temperature 0, with or without the drafter. E2B on this card and E4B on the
RTX 3050 repeat exactly. What causes it?

**Answer: flash attention in the llama.cpp build the rigs serve (`f95b0d9`).** With flash
attention off, or on the newer build `77c4540ce`, E4B repeats exactly. CPU threads, CUDA graphs
and the drafter make no difference.

| arm (E4B unless noted), 10 identical requests each | story: distinct texts | code: distinct texts | tok/s |
|---|---:|---:|---:|
| `f95b0d9`, `-fa 1` (the rig's setting) | 10 | 6 | 38.9 |
| `f95b0d9`, **`-fa 0`** | **1** | **1** | 37.5 / 37.2 |
| `f95b0d9`, `-fa 1`, `-t 1 -tb 1` | 10 | 5 | 38.8 |
| `f95b0d9`, `-fa 1`, `GGML_CUDA_DISABLE_GRAPHS=1` | 10 | 5 | 39.0 |
| `f95b0d9`, `-fa 1`, exact Q4_0 drafter at 2 tokens | 10 | 4 | 45.4 / 69.4 |
| **`77c4540ce`**, `-fa 1` | **1** | **1** | 38.0 / 37.9 |
| E2B, `f95b0d9`, `-fa 1` (control) | 1 | 1 | 78.7 / 78.5 |

- **Where repeats part:** the story at word 12 in most repeats, the code at word 37 or 83. The
  same near-tie words flip, so a small run-to-run difference in the attention result decides them.
- **E2B on the same build and flags repeats exactly.** The fault depends on E4B's shapes (hidden size
  2,560 against E2B's 1,536), not on the card alone.
  Which kernel in `f95b0d9` is responsible was not traced.
- **Cost of each fix on this card:** `-fa 0` 0.96x decode on E4B here (37.4 against 38.9 tok/s,
  mean of the two prompts); the 2026-09-03 sweep measured flash attention worth +4.8% decode on E2B.
  `77c4540ce` decodes E4B at 38.0 tok/s, 0.98x, at the same setting.

**What this means:** on the 1650 Ti rigs, E4B greedy text was not reproducible on `f95b0d9`,
so any E4B comparison there that relied on exact text is unsound; scores are unaffected in kind
(each reply is still a greedy reply, only near-ties differ). Moving the 1650 Ti rigs to
`77c4540ce` fixes it at about 2% decode. That is a rig change (both arms of the CPU/GPU control
build from one commit), so it is not made here.

## Method

`repeat.py`, run on the laptop `yogagpu` (GTX 1650 Ti Max-Q, 4096 MiB, driver 615.71.09)
2026-10-15 06:41–06:58. One llama-server per arm on port 8099 at `-c 8192 --parallel 1`, f16 KV,
`--reasoning off`, the rig's other flags (`-t 6 -tb 12`, no `-ot`); one setting changed per arm.
Prompt cache off. 10 repeats each of a story prompt and a code prompt, greedy, 256 tokens; card
cooled to ≤50 °C before each arm. Builds: `~/llama.cpp` at `f95b0d9` (the rigs' `LLAMA_SERVER_BIN`)
and `~/llama.cpp-77c4540ce` (`host.txt`). `summarize.py` → `summary.txt`; every reply's text is in
`results.jsonl`.

**Started late.** `run.sh` was to wait for the laptop's other queue
(`local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-10-14-drafter-gaps-1650ti`) to finish. That
queue waits in turn for `2026-10-14-drafter-accuracy-1650ti`, which stopped at 01:42 during its
last task (BFCL, E4B with the drafter) without writing DONE, so neither would have started. The
card was idle; this job was started at 06:41 without the wait (`driver.log`, `run.sh` comment).
The other queue was left as it was.
