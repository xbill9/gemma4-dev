# CLAUDE.md — local-pytorch-1650ti-2b-q4w4a16

Gemma 4 E2B, Google's QAT weights as the W4A16 repack `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text`, run in plain
PyTorch + transformers on the laptop's GTX 1650 Ti (TU117, sm_75, 4096 MiB). Local: nothing to provision.

**State as of 2026-10-23: an experiment, not a server.** No `server.py`, MCP server, skill or plugin yet. The current
code is `benchmarks/runs/2026-10-23-close-gap-1650ti/` (`e2b_w4.py`, `w4gemv_v5.py`, `w4attn.py`); each run's
`REPORT.md` holds its numbers. Headline: **89.8–90.6 tok/s decode, 2,793 MiB peak, 1.10x llama.cpp on the same card
the same day (82.0–82.3)**, every weight and vocabulary table int4 on the GPU, greedy replies token-identical to the
fp16-table build on 8 of 8 prompts in two runs (`MERGE=1 ATTN=w4decode STEP_COMPILE=1`, kernel v5 by default). First light was 18.1 tok/s.

Read before changing anything:

- **Stock transformers (5.16.1, and 5.20.2 by source) cannot compute on packed int4 here**: it unpacks the whole
  model to bf16 at the first forward, 3.58 GiB of layers, which does not fit. The scripts remove that hook.
- **Vocabulary tables come from `-text-emb4`** (`embed_tokens`, per-layer table, untied `lm_head`, int4 with fp16
  steps recovered from the QAT grid). In fp16 the per-layer table is 4.38 GiB and must sit in CPU RAM (`e2b_w4.py text`);
  never hand it to accelerate's offload, which copies the whole table to the GPU every forward.
- **fp16, not bf16, on the fast paths**: Inductor skips bf16 on sm_75. bf16 runs only as the accuracy reference.
- **`w4gemv_v5.py` pins one tiling per (rows, packed cols) shape**, tuned on this card; re-tune (`kbench.py` with v4)
  on other hardware.
- **`w4gemv_v*.py` assume the compressed-tensors pack-quantized layout**: symmetric int4, 8 nibbles per int32 low
  first, offset 8, group 32. It handles one row (decode); prefill goes through unpack + matmul.
- **`StaticSlidingWindowLayer` is given `StaticLayer`'s `update`, `get_seq_length` and `get_mask_sizes`**: its
  Python-int length goes stale inside a CUDA graph and makes torch.compile recompile every step. Valid only while
  `MAX_CACHE` (384) is no longer than the 512-token sliding window; the script asserts it.
- **Decode attention is `w4attn.py`'s Triton kernel**, registered as attention implementation `w4decode` (prompt
  reads fall back to SDPA). PyTorch's default memory-efficient kernel took 13 ms per token on sm_75 and the math
  backend about 5.6 ms; this one 0.85 ms.
- **Remove accelerate's hooks before compiling, then reapply the int4 forwards** (removing hooks restores the
  compressed-tensors forwards).
- The card is shared with `local-llamacpp-1650ti-*` and `local-ollama-1650ti-2b-q4_0`; stop their servers first.
- System `python3`, never a virtualenv. Tests, when there are some, are `unittest`.
