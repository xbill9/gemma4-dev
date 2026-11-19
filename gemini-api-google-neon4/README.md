# gemini-api-google-neon4

Gemini 4 Neon (`gemini-4-neon`, announced 2026-11-18) through the **Gemini API**, set up as a sibling of the Gemma 4 rigs so
the same quality checks can be read side by side. Nothing to provision and no MCP server: `tpu.env` names
the endpoint and model, and the key is read from `~/gemini.key`.

Its siblings on the same API and key are `gemini-api-google-flashlite35`, `gemini-api-google-flash38`,
`gemini-api-google-hydrogen4`, `gemini-api-google-carbon4`, `gemini-api-google-argon4`, `gemini-api-google-26b`
and `gemini-api-google-31b`.

#### Runs

- `benchmarks/runs/2026-11-19-accuracy-api/run.sh` — GSM8K (1,319, up to 2,048 tokens) and BFCL simple (400)
  through `benchmarks/gen-eval/gen_eval.py` at the lowest thinking level (`none`), the settings of the
  2026-11-14 seven-model run.
- `benchmarks/runs/2026-11-19-thinking-levels/run.sh` — AIME 2025 I (15 problems) at `low`, `medium`, `high`
  and `xhigh`, 32,768-token output limit, through `budget_run.py` under a dollar cap; the settings of
  `docs/gemini-high-vs-low`.

#### Naming

Slot 1 `gemini` (the Gemini API), slot 2 `api` (serving stack not disclosed), slot 3 `google`
(hardware not disclosed), slot 4 `neon4`. See `NAMING.md`.
