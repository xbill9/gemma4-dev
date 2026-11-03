# local-litertlm-3050oem-2b-int248

Gemma 4 E2B on an NVIDIA RTX 3050 OEM (8 GB) through [LiteRT-LM](https://ai.google.dev/edge/litert-lm/overview),
Google's on-device runtime, serving Google's own `.litertlm` build of the model with its built-in
speculative-decoding drafter. OpenAI-compatible API on `http://127.0.0.1:9379/v1`.

| Slot | Value | Means |
| --- | --- | --- |
| platform | `local` | hardware already in this machine, nothing provisioned |
| runtime | `litertlm` | `litert-lm serve` (LiteRT-LM 0.19.0) |
| hardware | `3050oem` | RTX 3050 OEM (GA106, Ampere), reached through WebGPU on Vulkan |
| model | `2b` | Gemma 4 E2B, instruction-tuned |
| encoding | `int248` | Google's mobile quantization: int2, int4 and int8 weights, one scale per row |

## Quick start

```bash
make install      # litert-lm + protobuf into the system python3 user site
make download     # ~/models/litert/gemma-4-E2B-it.litertlm, SHA-256 checked
make import       # register it as gemma-4-e2b in the rig's LITERT_LM_DIR
make serve        # foreground; Ctrl-C to stop
make query        # in another shell
```

Send `max_completion_tokens`, not `max_tokens`: the server ignores the latter.

Measurements: `benchmarks/runs/`. Configuration and its provenance: `tpu.env`. Agent notes: `CLAUDE.md`.
