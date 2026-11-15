# gen-eval: GSM8K and BFCL scoring against any served model

`gen_eval.py` sends GSM8K (1,319 math word problems) and BFCL simple (400 single
function calls) to an OpenAI-compatible chat endpoint and scores the replies in
code. It is standard-library Python (`urllib`, `json`, `re`), with no accelerator
code, so it scores a model on a desktop GPU, a rented accelerator or a hosted API
the same way.

```
python3 gen_eval.py run <server> <model> <gsm8k|bfcl_simple|aime2025|gpqa_diamond> <out.jsonl> [--limit N] [--concurrency C]
python3 budget_run.py <models.json> <aime2025|gpqa_diamond> <out_dir> --budget USD [--limit N]
python3 gen_eval.py score <out.jsonl>
python3 -m unittest discover -s tests -v
```

| File | Purpose |
| --- | --- |
| `gen_eval.py` | sends the questions and scores the answers; the docstring has the scoring rules |
| `build_gen_set.py` | rebuilds `data/` from the GSM8K and BFCL sources, with SHA-256 checks in `data/gen_manifest.json` |
| `gen_compare.py` | pairs two runs question by question |
| `resend_errors.py` | resends only the questions that failed with HTTP 429/5xx or a timeout |
| `build_hard_set.py` | writes the harder sets: `aime2025` (30, AIME 2025 I and II, pinned revision, committed) and `gpqa_diamond` (198, gated, kept out of git at GPQA's request); SHA-256s in `data/hard_manifest.json` |
| `budget_run.py` | runs one hard task across several models question by question under a dollar budget; refuses a run whose worst case (every reply at its output limit) exceeds the budget, and checks spend plus the next round's worst case before each round |

The hard tasks use a 32,768-token output limit (`GEN_EVAL_HARD_MAX_TOKENS`) and score the
line `Answer: <integer>` (AIME, read like GSM8K) or `Answer: <letter>` (GPQA).

Settings are environment variables: `GEN_EVAL_MAX_TOKENS`, `GEN_EVAL_CHAT_URL`,
`GEN_EVAL_API_KEY_FILE`, `GEN_EVAL_REASONING_EFFORT`, `GEN_EVAL_RETRIES`, `GEN_EVAL_TIMEOUT_S`,
`GEN_EVAL_CACHE_PROMPT`, `GEN_EVAL_ENABLE_THINKING`. The older `JEV_` spellings
(`JEV_GEN_MAX_TOKENS` for the token limit) are still read, so earlier `run.sh`
files and article commands work unchanged; a `GEN_EVAL_` name wins when both are set.

**Moved here 2026-11-15 from `jev-tpu-v5e1/`**, where it was first written. The old
paths are symlinks to these files, so every `run.sh` and article that names
`jev-tpu-v5e1/gen_eval.py` still runs. Called through a symlink, the script reads
`data/` beside the symlink, which is itself a symlink back here.
