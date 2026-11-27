# E4B thinking study on the GTX 1650 Ti — run notes

The 3050's E4B thinking run (`local-llamacpp-3050oem-4b-q4_0/benchmarks/runs/2026-11-17-thinking-e4b-3050oem`)
on the `yogagpu` laptop (GTX 1650 Ti Max-Q, 4 GB), started 2026-11-25, the E4B twin of
`local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-11-23-thinking-e2b-1650ti`. Unattended, on AC power, under
`systemd-inhibit` so the laptop does not sleep.

## What it measures

16 cells: 4 server arms x thinking off/on x GSM8K (1,319) and AIME 2025 (30).

| Arm | Weights | Drafter |
| --- | --- | --- |
| `repack-none` | xbill9 exact Q4_0 (`gemma-4-E4B-it-qat-q4_0-exact`) | none |
| `repack-q4x` | same | exact Q4_0 E4B QAT assistant, 2 drafted tokens |
| `google-none` | `google/gemma-4-E4B-it-qat-q4_0-gguf` @ `4b4a2c1` | none |
| `google-q4x` | same | same drafter |

Same as the 3050 run: llama.cpp `7182ef7`, the same three files by SHA-256 (`weights.sha256`), the same cells in
the same order (per arm: GSM8K off, AIME off, AIME on, GSM8K on), 2 drafted tokens, `--parallel 1`, f16 cache,
temperature 0, 32,768-token output limit, 1,800 s per-request timeout.

Two settings differ:

- **Context 36,864 tokens** (the 3050 ran 131,072). Google's file with the drafter fails at load at 40,960
  (CUDA out of memory in `cudaGraphInstantiate`) and fits at 36,864. The longest prompt in the 3050 run is 323
  tokens, so every request still has room for its full 32,768-token reply; the context size changes no reply.
- **The per-layer table stays in system RAM** (this rig's setting; the 3050 put it on the card).

Fit, measured 2026-11-25 at 36,864 with one 400-token reply: repack 3,355 MiB at 38.7 tok/s, repack + drafter
3,467 MiB at 56.7, Google + drafter 3,671 MiB at 52.7, of 4,096.

Estimate before the run, from those rates against the 3050's per-cell hours: about 48 hours (the 3050 took
26.83). Arms are committed as they finish, so the repack drafter-on/off pair is complete after roughly the first
half.

## Is it running?

```bash
cd ~/gemma4-dev/local-llamacpp-1650ti-4b-q4_0/benchmarks/runs/2026-11-25-thinking-e4b-1650ti
pgrep -af 'run\.sh$'; pgrep -af gen_eval.py; tail -5 driver.log
wc -l *.jsonl.partial 2>/dev/null
```

## If it dropped

`pkill -x llama-server`, then from this directory:

```bash
setsid nohup systemd-inhibit --what=sleep:idle:handle-lid-switch --why="E4B thinking run" ./run.sh >> nohup.out 2>&1 < /dev/null &
```

`run.sh` is itself the resume: it skips a cell whose `.jsonl` holds every problem and an arm whose four cells
are done, and re-runs any other cell with `GEN_EVAL_RESUME=1`, which keeps the saved replies in
`<cell>.jsonl.partial` and asks only the rest. It refuses to start beside another live `run.sh` here.

When it finishes it runs `analyze.py` (writes `RESULTS.md`, `results.json`) and pushes. `tokens.py` needs
E4B's `tokenizer.json` passed to it.
