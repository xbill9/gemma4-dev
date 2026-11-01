# CLAUDE.md — local-pytorch-1650ti-4b-q4w4a16

Gemma 4 E4B, Google's QAT weights as the W4A16 repack `xbill9/gemma-4-E4B-it-qat-q4_0-w4a16-ct-text`, in plain PyTorch +
transformers on the laptop's GTX 1650 Ti (TU117, sm_75, 4096 MiB). Forked 2026-10-31 from `local-pytorch-1650ti-2b-q4w4a16`;
**read that rig's CLAUDE.md too**: its notes on the hook removal, fp16, K/V ownership, the SDPA stride-0 bug and fp32 prompt
reads all apply here. No MCP server or skill.

- `e4b_engine.py` — the engine, with `w4gemv.py` (tilings pinned for E4B's shapes) and `w4attn.py` (unchanged kernels).
- `openai_server.py` — `/v1/chat/completions`, one request at a time, greedy, tool calls via `gemma4_tools.py`.

Headline (`benchmarks/runs/2026-10-31-accuracy-1650ti/REPORT.md`): **43.7 tok/s at short context and 34.8 at 32k, 1.10x and
1.27x llama.cpp without a drafter; 40,960 tokens fit in 3,304 MiB; GSM8K 0.9325 and BFCL 0.910, level with llama.cpp.**
llama.cpp with its drafter does 63.5 tok/s at short context; this engine has no drafter.

What differs from E2B:

- **The per-layer table lives in CPU RAM** (`_CPUEmbedding`): 1.5 GiB even in int4, so all of it does not fit. `load()`
  copies one row into GPU buffers before each graph replay. Prompt reads gather and copy their rows per chunk.
- **Two KV heads.** Each head is its own `_KV` with contiguous buffers, and query heads 0-3 / 4-7 read head 0 / 1 through
  the unchanged kernels, two calls per layer.
- **Prompt reads unpack slices of at most 4096 x 2560 elements** (`PREFILL_ELEMENTS`). Whole merged gate+up (205 MiB) or
  `down_proj` (100 MiB) leave too little room for K/V.
- **`torch.cuda.empty_cache()` before the decode compile and every prompt read** keeps the fp32 slices from fragmenting
  the pool. `expandable_segments` fails: cuBLAS then cannot allocate its handle.
- **40,960 tokens is the ceiling**: 49,152 fails on the first long prompt read. Full layers cost 16,384 B/token here.
- **Tuning found about 1.5% over the default tiling**; the kernel already runs near the card's memory bandwidth.

Do not convert the stock model to fp32 on this host: about 30 GB against 15 GB of RAM, and the session's low-memory reaper
killed the run that tried. Loading takes about 5 minutes because `from_pretrained` reads the fp16 per-layer table first.
