# 2026-11-13 first light — Gemma 4 31B on AMD Token Lab

One streamed request at a time, temperature 0, from this workstation to
`https://api.tokenlab.amd.com/v1`, model `Gemma-4-31B-IT`. Raw records: `probe.jsonl`
(written by `probe.py`). Started 15:39:46 UTC.

| Thinking | max_tokens | Prompt tok | Completion tok | Headers | First token | Total | Decode tok/s | Finish | Pings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| off | 256 | 22 | 97 | 0.46 s | 5.04 s | 17.8 s | 7.55 | stop | 1 |
| off | 256 | 32 | 256 | 0.37 s | 1.22 s | 35.0 s | 7.55 | length | 2 |
| off | 256 | 26 | 35 | 0.39 s | 2.23 s | 12.3 s | 3.41 | stop | 0 |
| on | 1024 | 25 | 390 (291 reasoning) | 0.41 s | 3.35 s | 199.8 s | 1.98 | stop | 13 |

Decode tok/s = (completion tokens − 1) / (last token time − first token time), measured here, so
it includes the network and the service's other load. No request was retried; every stream ended
with `[DONE]`. Points charged (`usage.cost`): 5.75e-05 in total for the four.

What this shows:

- **Decode ran at 2.0–7.6 tok/s for one request**, varying by a factor of 3.8 within five minutes on
  identical settings: the endpoint is shared, so these numbers describe Token Lab at that time, not
  the hardware or SGLang.
- Headers return in under 0.5 s; the wait before the first token (1.2–5.0 s) is spent on `: ping`
  keep-alives, i.e. queued.
- Thinking works through `chat_template_kwargs.enable_thinking` and streams as reasoning deltas
  before the answer.
- Before this run, the first plain request of the day was answered 429
  `model_concurrency_rate_limit_exceeded` (limit 128, `Retry-After: 1`) and the retry succeeded at
  once; an unretried request took 55 s end to end for 8 tokens.

Not measured: concurrency, long prompts, tools, accuracy (GSM8K/BFCL). Any of those against a shared
endpoint needs repeats spread over time to mean anything.
