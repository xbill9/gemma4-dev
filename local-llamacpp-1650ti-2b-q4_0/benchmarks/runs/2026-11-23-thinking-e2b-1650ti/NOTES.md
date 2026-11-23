# E2B thinking study on the GTX 1650 Ti — run notes

The 3050's E2B thinking run (`local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-11-16-thinking-e2b-3050oem`)
on the `yogagpu` laptop (GTX 1650 Ti Max-Q, 4 GB), started 2026-11-23. Unattended, on AC power, under
`systemd-inhibit` so the laptop does not sleep.

## What it measures

16 cells: 4 server arms x thinking off/on x GSM8K (1,319) and AIME 2025 (30).

| Arm | Weights | Drafter |
| --- | --- | --- |
| `repack-none` | xbill9 exact Q4_0 (`gemma-4-E2B-it-qat-q4_0-exact-v2`) | none |
| `repack-q4x` | same | exact Q4_0 E2B QAT assistant, **1** drafted token |
| `google-none` | `google/gemma-4-E2B-it-qat-q4_0-gguf` @ `675cff4` | none |
| `google-q4x` | same | same drafter |

Same as the 3050 run: llama.cpp `7182ef7`, the same three files by SHA-256 (`weights.sha256`), the same cells
in the same order (per arm: GSM8K off, AIME off, AIME on, GSM8K on), `-c 131072 --parallel 1`, f16 cache,
temperature 0, 32,768-token output limit, 1,800 s per-request timeout.

Two settings differ, both this rig's own: the drafter drafts 1 token (`SPEC_DRAFT_N_MAX=1` in `tpu.env`, the
best length on this card; the 3050 drafted 2), and the per-layer table stays in system RAM (the 3050 put it on
the card). Measured before the run: 2,331 MiB of 4,096 in use at 131,072 tokens without the drafter, 2,523 MiB
with it.

The 3050 run took 18.8 hours.

## Is it running?

```bash
cd ~/gemma4-dev/local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-11-23-thinking-e2b-1650ti
pgrep -af 'run\.sh$'; pgrep -af gen_eval.py; tail -5 driver.log
wc -l *.jsonl.partial 2>/dev/null
```

## If it dropped

`pkill -x llama-server`, then from this directory:

```bash
setsid nohup systemd-inhibit --what=sleep:idle:handle-lid-switch --why="E2B thinking run" ./run.sh >> nohup.out 2>&1 < /dev/null &
```

`run.sh` is itself the resume: it skips a cell whose `.jsonl` holds every problem and an arm whose four cells
are done, and re-runs any other cell with `GEN_EVAL_RESUME=1`, which keeps the saved replies in
`<cell>.jsonl.partial` and asks only the rest. It refuses to start beside another live `run.sh` here.

When it finishes it runs `analyze.py` (writes `RESULTS.md`, `results.json`) and pushes. `tokens.py` needs
E2B's `tokenizer.json` passed to it, as on the 3050.
