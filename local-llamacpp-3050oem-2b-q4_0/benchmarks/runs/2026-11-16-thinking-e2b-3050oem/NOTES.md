# E2B thinking study, wave 1 — run notes

Started 2026-11-16 15:42 on the `gpu` host (RTX 3050 OEM), llama.cpp `7182ef71e`. Unattended.
These notes are for picking the run up if the machine, the server or the run drops.

## What it measures

Gemma 4 E2B only, so the model is held fixed. 16 cells: 4 server arms x thinking off/on x 2 tasks.

| Arm | Weights | Drafter |
| --- | --- | --- |
| `repack-none` | xbill9 exact Q4_0 (`gemma-4-E2B-it-qat-q4_0-exact-v2`, SHA `419db9a6…`) | none |
| `repack-q4x` | same | exact Q4_0 E2B QAT assistant, 2 drafted tokens |
| `google-none` | `google/gemma-4-E2B-it-qat-q4_0-gguf` @ `675cff4` (`~/models/gemma-4-E2B-it-qat-q4_0-google/`) | none |
| `google-q4x` | same | same drafter |

Per arm, in order: `gsm8k` thinking off, `aime2025` off, `aime2025` on, `gsm8k` on (the longest last).
Thinking is `chat_template_kwargs.enable_thinking`; Gemma 4's template has no levels, only on/off.
All cells: `-c 131072 --parallel 1`, f16 cache, one request at a time, temperature 0, prompt cache at
the server default, 32,768-token output limit, 1,800 s per-request timeout. SHA-256 of both weight files
is in `weights.sha256`.

Files per cell: `<task>-<arm>-think<off|on>.jsonl` (one row per problem, with the thinking text in
`reasoning` and llama.cpp `timings` incl. `draft_n` / `draft_n_accepted`) and `.log` (summary line last).
`driver.log` has one line per event with GPU temperature and clock. Each arm is committed and pushed
when its four cells finish.

## Is it running?

On `gpu`:

```bash
cd ~/gemma4-dev/local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-11-16-thinking-e2b-3050oem
pgrep -af '^(/usr)?/bin/bash (\./)?(run|resume)\.sh$'   # the driver
pgrep -af gen_eval.py                                   # the cell in progress
tail -5 driver.log
wc -l *.jsonl.partial 2>/dev/null                       # replies saved so far in the live cell
```

## If it dropped

1. If `nvidia-smi` fails after a reboot, nouveau has taken the card: re-check the nouveau blacklist,
   then reboot again.
2. Make sure nothing is left on port 8080: `pkill -x llama-server`.
3. Resume from this directory:

   ```bash
   setsid nohup ./resume.sh > nohup-resume.out 2>&1 < /dev/null &
   ```

`resume.sh` runs the same cells in the same order with the same settings. It skips a cell whose
`.jsonl` already holds every problem (1,319 GSM8K, 30 AIME), and skips an arm whose four cells are
all done without starting its server. Any other cell runs with `GEN_EVAL_RESUME=1`: `gen_eval.py` keeps
the rows in `<cell>.jsonl.partial` that have no `error`, re-asks the rest, and writes the full `.jsonl`
in dataset order at the end. Rows that errored (a dead server, a timeout) are asked again. It refuses
to start while `run.sh` or another `resume.sh` is live in this directory. It is safe to run more than once.

**One gap:** the first cell (`gsm8k-repack-none-thinkoff`) started before `gen_eval.py` saved replies as
they finish, so if the run drops inside that cell it starts that cell over. Every later cell is a new
`gen_eval.py` process and saves as it goes.

## What a resumed cell changes

Nothing in the settings. A cell finished after a resume holds replies from two server starts; the prompt
cache and the drafter make greedy text differ slightly between starts, which earlier runs measured as
changing no score (`2026-10-14-accuracy-cache-3050oem`). Note any resumed cell in the report.

## Afterwards

- Thinking tokens: tokenize each row's `reasoning` with the served model (`/tokenize`), in code.
- Control: `gsm8k-repack-none-thinkoff` against the 2026-10-14 thinking-off score of 0.9014
  (`2026-10-14-accuracy-fullctx-3050oem`, 2,048-token limit there, 32,768 here).
- Paired tests per problem (McNemar) for thinking, drafter and weights; Wilson intervals on each score.
