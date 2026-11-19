# 2026-11-19 — Gemini 4 Neon at four thinking levels, AIME 2025 I

The 15 problems, prompt and 32,768-token limit of `docs/gemini-high-vs-low`, at `low`, `medium`, `high` and `xhigh`
(Neon also accepts `none` and rejects `minimal`). `run.sh` drives `budget_run.py` with a $30 stop; `derived.py`
writes `derived.txt`.

| Level | Right | Thinking mean | Median s | Cost | Per right answer |
| :--- | ---: | ---: | ---: | ---: | ---: |
| low | 15 / 15 | 2,499 | 46.8 | $0.7768 | $0.0518 |
| medium | 15 / 15 | 4,032 | 72.9 | $1.1302 | $0.0753 |
| high | 15 / 15 | 5,895 | 106.5 | $1.5291 | $0.1019 |
| xhigh | 15 / 15 | 13,248 | 242.0 | $3.0609 | $0.2041 |

Every level solved every problem; xhigh cost 3.94x low. On 15 problems a ceiling score cannot separate the levels.
