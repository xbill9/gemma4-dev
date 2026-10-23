# CLAUDE.md — local-pytorch-1650ti-2b-q4w4a16

Gemma 4 E2B, Google's QAT weights as the W4A16 repack `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text`, run in plain
PyTorch + transformers on the laptop's GTX 1650 Ti (TU117, sm_75, 4096 MiB). Local: nothing to provision.

**State as of 2026-10-23: an experiment, not a server.** No `server.py`, MCP server, skill or plugin yet. The current
code is `benchmarks/runs/2026-10-23-lmhead-kernel-1650ti/e2b_w4.py` with `w4gemv_v2.py`; each run's `REPORT.md` holds its
numbers. Headline: **57.3–57.6 tok/s decode, 2,769 MiB peak**, every weight and vocabulary table int4 on the GPU,
greedy replies token-identical to the fp16-table build on 8 of 8 prompts (`STEP_COMPILE=1 DECODE_SDPA=math`).
First light the same day was 18.1 tok/s.

Read before changing anything:

- **Stock transformers (5.16.1, and 5.20.2 by source) cannot compute on packed int4 here**: it unpacks the whole
  model to bf16 at the first forward, 3.58 GiB of layers, which does not fit. The scripts remove that hook.
- **Vocabulary tables come from `-text-emb4`** (`embed_tokens`, per-layer table, untied `lm_head`, int4 with fp16
  steps recovered from the QAT grid). In fp16 the per-layer table is 4.38 GiB and must sit in CPU RAM (`e2b_w4.py text`);
  never hand it to accelerate's offload, which copies the whole table to the GPU every forward.
- **fp16, not bf16, on the fast paths**: Inductor skips bf16 on sm_75. bf16 runs only as the accuracy reference.
- **`w4gemv_v2.py` (and v1) assume the compressed-tensors pack-quantized layout**: symmetric int4, 8 nibbles per int32 low
  first, offset 8, group 32. It handles one row (decode); prefill goes through unpack + matmul.
- **`StaticSlidingWindowLayer` is given `StaticLayer`'s `update`, `get_seq_length` and `get_mask_sizes`**: its
  Python-int length goes stale inside a CUDA graph and makes torch.compile recompile every step. Valid only while
  `MAX_CACHE` (384) is no longer than the 512-token sliding window; the script asserts it.
- **Decode attention runs on `SDPBackend.MATH`**: PyTorch's default memory-efficient kernel took 13 ms per token for
  35 single-query calls on sm_75.
- **Remove accelerate's hooks before compiling, then reapply the int4 forwards** (removing hooks restores the
  compressed-tensors forwards).
- The card is shared with `local-llamacpp-1650ti-*` and `local-ollama-1650ti-2b-q4_0`; stop their servers first.
- System `python3`, never a virtualenv. Tests, when there are some, are `unittest`.
