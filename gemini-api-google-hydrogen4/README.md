# gemini-api-google-hydrogen4

Gemini 4 Hydrogen (`gemini-4-hydrogen`) through the **Gemini API**, set up as a sibling of the Gemma 4 rigs so the same
quality checks can be read side by side. Nothing to provision and no MCP server: `tpu.env` names
the endpoint and model, and the key is read from `~/gemini.key`.

Its siblings on the same API and key are `gemini-api-google-flashlite35`, `gemini-api-google-flash38`, `gemini-api-google-hydrogen4`, `gemini-api-google-carbon4`, `gemini-api-google-argon4`.

#### Run the quality checks

```bash
benchmarks/runs/2026-11-13-accuracy-gemini/run.sh
```

GSM8K (1,319, up to 2,048 tokens) and BFCL simple (400) through `jev-tpu-v5e1/gen_eval.py`, the
harness every Gemma rig uses: temperature 0, scored in code. Two arms: the lowest thinking level
the model accepts (`none`) and `low`. Results are in `benchmarks/INDEX.md`.

#### Naming

Slot 1 `gemini` (the Gemini API), slot 2 `api` (serving stack not disclosed), slot 3 `google`
(hardware not disclosed), slot 4 `hydrogen4`. See `NAMING.md`.
