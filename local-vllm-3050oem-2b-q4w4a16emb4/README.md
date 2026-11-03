# local-vllm-3050oem-2b-q4w4a16emb4

vLLM 0.33.0 serving Gemma 4 E2B (`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4`, QAT Q4_0 grid,
int4 throughout) on an RTX 3050 OEM, OpenAI-compatible on `127.0.0.1:8010`.

```
make install     # vLLM on the system python3, keeping the system torch
make serve       # foreground; Ctrl-C stops it
make query
```

Measurements are in `benchmarks/runs/`. `CLAUDE.md` has the details.
