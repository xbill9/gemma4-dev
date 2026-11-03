# 2026-11-03 — tool calls with and without litert-lm serve's constrained decoding, RTX 3050 OEM

**Question:** on BFCL simple, LiteRT-LM scored 0.825 (Google's `.litertlm`) and 0.8625 (the exact QAT Q4_0
build) against llama.cpp's 0.9225 on the same card, though the exact build holds the same int4 codes and
scales as llama.cpp's GGUF. Where does the gap come from?

**Answer:** from the server's constrained decoding. `litert-lm serve` 0.19.0 turns it on for every chat
request, and with tools present its grammar makes the model write arguments in the order the schema
declares them. The Gemma 4 chat template writes them alphabetically (`dictsort`), and with declared order
forced the model leaves out optional arguments the question asks for. With the constraint off, the exact
build scores 0.9125, level with llama.cpp, and Google's file 0.890.

## Method

BFCL simple (400 prompts, one tool each, `tool_choice` auto), greedy, one request at a time, 512-token
limit, `jev-tpu-v5e1/gen_eval.py`: the harness, set and limits of the llama.cpp and PyTorch accuracy runs on
this card. Drafter off. `serve_noconstraint.py` runs the stock `litert-lm serve` with the handler's
`ConstrainedDecodingConfig` built with `enable=False`; nothing else changes. `proxy.py` moves `max_tokens`
to `max_completion_tokens` (the server reads only the latter). `run.sh int248-off` and
`run.sh q4exact-off` are the two new arms. The constraint-on arms are the stock-server runs in
`local-litertlm-3050oem-2b-q4w4a16emb4/benchmarks/runs/2026-11-02-accuracy-3050oem`, and llama.cpp is
`local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-accuracy-fullctx-3050oem/bfcl_simple-e2b-none.jsonl`
(exact Q4_0 E2B, no drafter). Both new arms ran beside that rig's GSM8K run on the same card. `compare.py`
computes every figure below; ranges are 95% paired bootstrap (2,000 resamples, seed 0).

## Results

| arm | BFCL | right | missing optional argument | keys in declared order / alphabetical |
| --- | ---: | ---: | ---: | ---: |
| LiteRT int248, constraint on (stock) | 0.8250 | 330 | 33 | 198 / 0 |
| LiteRT int248, constraint off | 0.8900 | 356 | 5 | 0 / 230 |
| LiteRT q4exact, constraint on (stock) | 0.8625 | 345 | 24 | 200 / 0 |
| LiteRT q4exact, constraint off | 0.9125 | 365 | 5 | 1 / 226 |
| llama.cpp exact Q4_0 | 0.9225 | 369 | 3 | 1 / 230 |

Key order counts calls with two or more arguments whose order is one and not the other; the rest
(134–160 per arm) are in both orders at once.

| difference | value | 95% range | prompts gained / lost |
| --- | ---: | --- | ---: |
| int248 off − on | +0.0650 | [+0.0375, +0.0975] | +34 / −8 |
| q4exact off − on | +0.0500 | [+0.0275, +0.0750] | +23 / −3 |
| q4exact off − llama.cpp | −0.0100 | [−0.0225, +0.0025] | +1 / −5 |
| int248 off − llama.cpp | −0.0325 | [−0.0600, −0.0075] | +8 / −21 |
| q4exact off − int248 off | +0.0225 | [−0.0025, +0.0475] | +18 / −9 |
| q4exact on − llama.cpp | −0.0600 | [−0.0875, −0.0350] | +3 / −27 |

- **Missing optional arguments are the constraint's failure.** With it on, all 24 of the exact build's
  missing-argument failures name an optional parameter (none a required one), e.g. `species` in
  `ecology.get_turtle_population`, `top_n`, `format`. Off, they fall to 5, near llama.cpp's 3.
- **Off, the exact QAT grid matches llama.cpp** (range includes zero). Google's int2/int4/int8 file stays
  behind llama.cpp, and 0.0225 behind the exact build, a range that just includes zero.
- **The constraint guarantees a parseable call; without it one reply in 400 fails.** `simple_337`
  (q4exact off) wrote single-quoted strings inside a nested object; the server's tool-call parser rejected
  it and returned HTTP 500 (`server-q4exact-off.log`), scored wrong. Off, a client should expect an
  occasional 500 on malformed calls.

## Serving

The stock server has no switch for this: constrained decoding is on for every chat request, in
`litert_lm_cli/commands/serve/openai_chat_completions.py`. Tool-calling clients get the better scores
through `serve_noconstraint.py`.
