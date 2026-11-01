# CLAUDE.md — local-pytorch-3050oem-2b-q4w4a16

Gemma 4 E2B, the QAT W4A16 repack, in plain PyTorch + transformers on the RTX 3050 OEM (GA106, sm_86, 8192 MiB) in the
i7-5820K desktop `gpu`. Local: nothing to provision. The engine of `local-pytorch-1650ti-2b-q4w4a16` by way of
`gpu-pytorch-g6-2b-q4w4a16`; read the 1650 Ti rig's `CLAUDE.md` first, every engine rule there holds here.

**State as of 2026-10-31: an engine and an OpenAI-compatible server, no MCP server or skill.** Nothing serves at boot.

- `e2b_engine.py`, `w4attn.py`, `gemma4_tools.py`: unchanged from the G6 rig. `w4gemv.py`: tilings changed where the
  autotuner found 3% or more on this card, worth 0.3% decode. `openai_server.py`: the G6 server, default port 8000.
- Run with the installed `transformers==5.16.1` and `compressed-tensors==0.17.0`; both went into the user site on
  2026-10-31 (the Debian package was 5.12.1), torch 2.14.1+cu130, Triton 3.8.0.
- fp16 compute carried over; fp16 prompt reads beat fp32 here at 4k and 32k tokens (tensor cores).
- The card is shared with every `local-llamacpp-3050oem-*` rig; stop their servers first.

Headline (`benchmarks/runs/2026-10-31-first-light-3050oem/REPORT.md`): 127 tok/s decode at short context, 1.07x llama.cpp
on the same card the same night, 0.92x at 32k tokens held and 0.75x at 131k. Accuracy level with llama.cpp
(`benchmarks/runs/2026-10-31-accuracy-3050oem`). llama.cpp with the drafter stays ahead: this engine has no drafter.
