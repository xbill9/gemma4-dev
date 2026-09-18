# GEMM dtype ranking at E2B decode shapes — MI300X, 2026-09-18

**Question** (from a reader of the dollar-hour article): the fp8 1.77x / int8 0.69x ranking in
`../../../../HARDWARE.md` came from an 8192³ matmul, which decode never runs. Does it survive at
M = 1 and 8 with N x K at the model's real layer shapes?

**Answer: fp8 survives, smaller and uneven; int8 does not survive at all.**

| M | bf16 us/token | fp8 us/token | fp8 x bf16 | int8 |
| ---: | ---: | ---: | ---: | --- |
| 1 | 1965.1 | 1306.4 | **1.50** | cannot run — `_int_mm` needs M > 16 |
| 8 | 2008.2 | 1323.5 | **1.52** | cannot run |
| 64 | 2375.7 | 1592.7 | **1.49** | 0.20x bf16 |

"us/token" is the sum over every E2B linear layer, each shape weighted by how many times a decode
step runs it (141 GEMMs), graph-replayed, weights cold. All figures computed by the script.

## What the per-shape rows say

- **The win tracks weight size.** fp8 per-shape speedup ranges from **0.86x** (`qkv_sliding`,
  2560x1536, M=1, fp8 *slower*) to **1.81x** (`lm_head`, 262144x1536). At M = 1 and 8,
  `qkv_sliding` is the only shape where fp8 loses; at M = 64 none do (min 1.28x).
- **Small shapes are not bandwidth-bound.** bf16 `qkv_sliding` reaches ~37% of the 3.76 TB/s
  measured copy bandwidth; `lm_head` reaches 95%. Halving the bytes only pays when the kernel was
  streaming bytes in the first place.
- **The LM head is not the story.** It is ~12% of per-token GEMM time; excluding it, fp8 is still
  1.47x / 1.49x / 1.50x bf16 at M = 1 / 8 / 64.
- **The control reproduces the original.** 8192³ at M = 8192: fp8 **1.78x** bf16 (original 1.77x),
  bf16 612.7 TFLOP/s (original 664.3; both runs beside a live server).

## int8 is worse than the original table said

- `torch._int_mm` **refuses M ≤ 16**: `self.size(0) needs to be greater than 16`. At M = 1 and 8
  there is no int8 number to report; 20 of 124 cells are that error.
- At M = 64 it runs at **0.10x-0.72x** bf16 per shape, 0.20x per token.
- At the 8192³ control it measured **0.40x**, not the 0.69x recorded 2026-09-16. Same torch
  (2.12.0+rocm10.0.0); the vLLM nightly under the same tag differs (0.3.1.dev85 today). **Unexplained**
  — recorded, not reconciled.

## The measurement trap this run nearly fell into

The first pass timed each GEMM eagerly from Python (`eager-*` files, kept). Every small shape
floored at ~17 us bf16 / ~20 us fp8 regardless of size, and fp8 read **0.97x** bf16 per token —
the opposite conclusion. That floor is host dispatch: vLLM replays decode from HIP graphs, so it
never pays it. Replayed from a graph, the same kernels run 1.46x (M=1) to 1.30x (M=64) faster in
bf16, and fp8 becomes 1.50x. `graph_check.py` was the ad-hoc test that found this; the harness
now defaults to `--mode graph`.

The commenter's premise — small-M, decode-shaped GEMMs rank differently from 8192³ — was right in
kind. The direction of the error depended on how you time them.

## Not measured

- vLLM's per-call **dynamic activation quantisation** for fp8 (one extra small kernel per GEMM).
  The fp8 figures are a GEMM-only upper bound.
- Attention, norms, sampling: the non-GEMM share of a decode step.
- ~~End-to-end tokens/sec with `--quantization fp8`~~ — **measured the same day: fp8 serving is
  0.53x–0.80x bf16.** The GEMM ratio does not survive vLLM's fp8 path. See
  `../2026-09-18-vllm-sweep-mi300x-v0191-fp8/REPORT.md`.

## Provenance

- Droplet `601744248` (created 2026-09-18, Debian 13.5, kernel 6.12.94+deb13-amd64), scaffolded
  by `~/amd-gputools/scaffold-droplet.sh`, card bound after one reboot.
- Run inside the `vllm` container, `vllm/vllm-openai-rocm:nightly-rocm100`
  (vLLM 0.3.1.dev85+gdee37d891, torch 2.12.0+rocm10.0.0), while it served `google/gemma-4-E2B-it`
  idle at `gpu_memory_utilization=0.90`.
- `--rotate-bytes 1073741824` (4x the 256 MB Infinity Cache), 3 repeats, median; worst-cell cv 3.7%.

| file | what |
| --- | --- |
| `gemm.json`, `gemm.md`, `gemm.stderr.jsonl` | the headline run, `gemm_decode_shapes.py --mode graph` |
| `eager-gemm.*` | first pass, eager timing — dispatch-bound, diagnostic only |
| `graph_check.py`, `graph_check.json` | the eager-vs-graph test that caught it (bf16/fp8 only) |
