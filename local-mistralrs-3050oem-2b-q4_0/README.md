# local-mistralrs-3050oem-2b-q4_0

Gemma 4 E2B (exact QAT Q4_0 GGUF) served by mistral.rs v0.9.5 on an RTX 3050 OEM (8 GB).

    make install   # clone v0.9.5 and cargo install with CUDA + flash-attn (sm_86)
    make serve     # OpenAI-compatible server on 127.0.0.1:8020
    make query

Settings live in `tpu.env`; `CLAUDE.md` has the memory limits on this card. Measurements are in
`benchmarks/runs/`.
