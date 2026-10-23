# CLAUDE.md — local-pytorch-1650ti-2b-q4w4a16

Gemma 4 E2B, Google's QAT weights as the W4A16 repack `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text`, run in plain
PyTorch + transformers on the laptop's GTX 1650 Ti (TU117, sm_75, 4096 MiB). Local: nothing to provision.

**State as of 2026-10-23: an experiment, not a server.** No `server.py`, MCP server, skill or plugin yet. The code is
the scripts in `benchmarks/runs/2026-10-23-first-light-1650ti/`; its `REPORT.md` holds every number. Headline:
18.1 tok/s, 1,842 MiB peak, greedy reply token-identical to bf16.

Read before changing anything:

- **Stock transformers (5.16.1, and 5.20.2 by source) cannot compute on packed int4 here**: it unpacks the whole
  model to bf16 at the first forward, 3.58 GiB of layers, which does not fit. The scripts remove that hook.
- **The per-layer embedding table (4.38 GiB) lives in CPU RAM** and is looked up there each step. Do not hand it to
  accelerate's offload: that copies the whole table to the GPU every forward.
- **fp16, not bf16, on the fast paths**: Inductor skips bf16 on sm_75. bf16 runs only as the accuracy reference.
- **`w4gemv.py` assumes the compressed-tensors pack-quantized layout**: symmetric int4, 8 nibbles per int32 low
  first, offset 8, group 32. It handles one row (decode); prefill goes through unpack + matmul.
- **CUDA graphs need the `StaticSlidingWindowLayer.get_seq_length` patch** in `fused_cuda.py`, and it is only valid
  while `max_cache_len` stays below the 512-token sliding window.
- The card is shared with `local-llamacpp-1650ti-*` and `local-ollama-1650ti-2b-q4_0`; stop their servers first.
- System `python3`, never a virtualenv. Tests, when there are some, are `unittest`.
