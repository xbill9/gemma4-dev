# gemini-api-google-26b

Gemma 4 (Gemma 4 26B A4B IT, `gemma-4-26b-a4b-it`) through the **Gemini API**, run under the same conditions as the five
`gemini-api-google-*` Gemini rigs. Nothing to provision and no MCP server: `tpu.env` names the
endpoint and model, and the key is read from `~/gemini.key`. Free tier only.

```bash
benchmarks/runs/2026-11-14-accuracy-api/run.sh
```

Slot 1 `gemini`, slot 2 `api`, slot 3 `google`, slot 4 `26b` (the Gemma size). See `NAMING.md`.
