# gen-eval: GSM8K and BFCL scoring against any served model

`gen_eval.py` sends GSM8K (1,319 math word problems) and BFCL simple (400 single
function calls) to an OpenAI-compatible chat endpoint and scores the replies in
code. It is standard-library Python (`urllib`, `json`, `re`), with no accelerator
code, so it scores a model on a desktop GPU, a rented accelerator or a hosted API
the same way.

```
python3 gen_eval.py run <server> <model> <gsm8k|bfcl_simple> <out.jsonl> [--limit N] [--concurrency C]
python3 gen_eval.py score <out.jsonl>
python3 -m unittest discover -s tests -v
```

| File | Purpose |
| --- | --- |
| `gen_eval.py` | sends the questions and scores the answers; the docstring has the scoring rules |
| `build_gen_set.py` | rebuilds `data/` from the GSM8K and BFCL sources, with SHA-256 checks in `data/gen_manifest.json` |
| `gen_compare.py` | pairs two runs question by question |
| `resend_errors.py` | resends only the questions that failed with HTTP 429/5xx or a timeout |

Settings are environment variables: `JEV_GEN_MAX_TOKENS`, `JEV_CHAT_URL`,
`JEV_API_KEY_FILE`, `JEV_REASONING_EFFORT`, `JEV_RETRIES`, `JEV_TIMEOUT_S`,
`JEV_CACHE_PROMPT`, `JEV_ENABLE_THINKING`. The `JEV_` prefix is historical.

**Moved here 2026-11-15 from `jev-tpu-v5e1/`**, where it was first written. The old
paths are symlinks to these files, so every `run.sh` and article that names
`jev-tpu-v5e1/gen_eval.py` still runs. Called through a symlink, the script reads
`data/` beside the symlink, which is itself a symlink back here.
