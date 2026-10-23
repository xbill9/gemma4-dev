# CLAUDE.md — local-pytorch-1650ti-2b-q4w4a16

Gemma 4 E2B, Google's QAT weights as the W4A16 repack `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text`, run in plain
PyTorch + transformers on the laptop's GTX 1650 Ti (TU117, sm_75, 4096 MiB). Local: nothing to provision.

**State as of 2026-10-23: an engine and an OpenAI-compatible server, no MCP server or skill yet.**

- `e2b_engine.py` — the engine (`Engine(max_ctx=...)`: load, `render`, `generate`), with `w4gemv.py` (int4 GEMV,
  configs pinned per shape for this card) and `w4attn.py` (split decode attention, chunked prompt attention).
- `openai_server.py` — `/v1/chat/completions`, one request at a time, greedy, tool calls via `gemma4_tools.py`.
- `colab/` — the T4 notebook (jupytext source in `src/`, built into `notebooks/` by `tools/build_notebooks.py`).
- `benchmarks/runs/2026-10-23-*` — every measurement; the engine was promoted from `2026-10-23-long-context-1650ti/`.

Headline (each run's `REPORT.md` has the rest): **87.8 tok/s decode at short context, 1.07x llama.cpp on the same card
the same day; the full 131,072-token context fits in 3,471 MiB peak with every table on the GPU**, decode 51.0 tok/s at
32k (0.90x) and 21.8 at 131k (0.72x); greedy replies token-identical to the fp16-table build.

Read before changing anything:

- **Stock transformers (5.16.1, and 5.20.2 by source) cannot compute on packed int4 here**: it unpacks the whole
  model to bf16 at the first forward, 3.58 GiB of layers, which does not fit. The scripts remove that hook.
- **Vocabulary tables come from `-text-emb4`** (`embed_tokens`, per-layer table, untied `lm_head`, int4 with fp16
  steps recovered from the QAT grid). In fp16 the per-layer table is 4.38 GiB and must sit in CPU RAM (`e2b_w4.py text`);
  never hand it to accelerate's offload, which copies the whole table to the GPU every forward.
- **fp16, not bf16, on the fast paths**: Inductor skips bf16 on sm_75. bf16 runs only as the accuracy reference.
- **`w4gemv.py` (v5) pins one tiling per (rows, packed cols) shape**, tuned on this card; re-tune (`kbench.py` with
  v4 in `2026-10-23-close-gap-1650ti`) on other hardware.
- **`w4gemv*.py` assume the compressed-tensors pack-quantized layout**: symmetric int4, 8 nibbles per int32 low
  first, offset 8, group 32. It handles one row (decode); prefill goes through unpack + matmul.
- **The engine owns the K/V cache, not transformers.** Its `StaticSlidingWindowLayer` keeps a Python-int length that
  a CUDA graph freezes and that makes torch.compile recompile every step. Full layers get `max_ctx` slots; sliding
  layers a 512-slot ring that stores each slot's position; Gemma 4's attention forward is replaced (`_attn_forward`).
  During a chunked prompt read a sliding layer's new K/V is held back until the forward pass ends: the KV-shared
  layers later in the same pass still need the ring's older positions.
- **Decode attention is `w4attn.py`'s split Triton kernel**: all eight query heads per K/V read, no `tl.dot` (this
  card has no tensor cores; fp16 `tl.dot` ran at 24 GB/s). Prompt chunks use memory-efficient SDPA: full layers
  with K/V expanded across heads as views and the built-in bottom-right causal mask; sliding layers with an explicit
  window mask and **K/V copied, never expanded**. With an explicit mask and stride-0 K/V, memory-efficient SDPA (torch
  2.13 on this card) gets the last query row wrong whenever a chunk holds 32k+1 tokens, which gave a wrong first reply
  token on about 1 prompt in 32 (GSM8K: 12 of those 38 right, llama.cpp 36; `benchmarks/runs/2026-10-23-accuracy-1650ti/sdpa_repro.txt`).
- **Prompt reads multiply in fp32 on this card**: cuBLAS fp16 GEMMs run at ~0.4 TFLOP/s on TU117, fp32 at 1.6-1.9.
  On a GPU with tensor cores (a T4) fp16 may win; the Colab notebook times both and picks.
- **Remove accelerate's hooks before compiling, then reapply the int4 forwards** (removing hooks restores the
  compressed-tensors forwards).
- The card is shared with `local-llamacpp-1650ti-*` and `local-ollama-1650ti-2b-q4_0`; stop their servers first.
- System `python3`, never a virtualenv. Tests, when there are some, are `unittest`.
