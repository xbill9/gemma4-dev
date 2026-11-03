# local-litertlm-3050oem-2b-q4w4a16emb4

Gemma 4 E2B on an RTX 3050 OEM through [LiteRT-LM](https://ai.google.dev/edge/litert-lm/overview), from a
`.litertlm` built here that keeps Google's quantization-aware-trained (QAT) Q4_0 weights exactly: every 4-bit
value and scale equals the `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4` repack. OpenAI-compatible API
on `http://127.0.0.1:9380/v1`.

| Slot | Value | Means |
| --- | --- | --- |
| platform | `local` | hardware in this machine |
| runtime | `litertlm` | `litert-lm serve` (LiteRT-LM 0.19.0) |
| hardware | `3050oem` | RTX 3050 OEM, reached through WebGPU on Vulkan |
| model | `2b` | Gemma 4 E2B, instruction-tuned |
| encoding | `q4w4a16emb4` | int4 in groups of 32 on the QAT grid, vocabulary tables included |

## Quick start

```bash
make install     # litert-lm + protobuf into the system python3 user site
make build       # convert, write the exact codes, bundle (~20 min, ~45 GB RAM)
make import
make serve       # foreground; Ctrl-C to stop
make query
```

Send `max_completion_tokens`, not `max_tokens`. Measurements: `benchmarks/runs/`. Agent notes: `CLAUDE.md`.
