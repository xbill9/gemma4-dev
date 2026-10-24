# 2026-10-23 — 4-bit lm_head and embeddings, tuned int4 kernel, compiled decode step

Same host, card and stack as `../2026-10-23-first-light-1650ti/` (GTX 1650 Ti Max-Q, torch 2.13.0+cu130,
transformers 5.16.1, Triton 3.7.1). Decoder linears are `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text` at `315aec06`;
vocabulary tables come from `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4` at `db715f83`.

Eight prompts (`PROMPTS` in `e2b_w4.py`), chat template, greedy, up to 128 new tokens, static KV cache of 384
tokens, fp16. Decode tok/s excludes the prompt read; each row is the range over the eight prompts.

#### Result

| Build | Decode tok/s | GPU ms / token (kernels / replay) | Kernels / token | Peak CUDA MiB | Greedy tokens vs `text-v1` | File |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| fp16 tables, per-layer table in CPU RAM, kernel v1 | 20.52–20.60 | — | — | 1,843–1,845 | reference | `text-v1.jsonl` |
| int4 tables on the GPU, kernel v1 | 20.88–20.93 | 44.8 / 47.8 | 3,236 | 2,751–2,753 | 8 of 8 identical | `emb4-v1.jsonl`, `emb4-profile.jsonl` |
| + kernel v2 | 34.35–34.36 | 27.5 / 29.1 | 3,236 | 2,751–2,753 | 8 of 8 identical | `emb4-v2-default.jsonl` |
| + plain-math attention for decode | 41.03–41.12 | 22.0 / 24.4 | 3,726 | 2,767–2,769 | 7 of 8 (prompt 4 from token 88) | `emb4-v2-math.jsonl` |
| **+ decode step compiled with Inductor** | **57.31–57.61** | **16.1 / 17.5** | **776** | **2,767–2,769** | **8 of 8 identical** | `emb4-v2c-math.jsonl` |

The first-light build (`../2026-10-23-first-light-1650ti/`) decoded one prompt at 18.1 tok/s including its prompt read.
`local-llamacpp-1650ti-2b-q4_0` records 67.6–73.8 tok/s single-stream decode on the same card; different engine and
measurement, so the ratio (about 0.8x) is indicative.

Prompt read: 630–673 ms for 15–40 prompt tokens in every build, one outlier at 919 ms (`emb4-v1`, prompt 2). It unpacks every linear to fp16 before multiplying and is
not optimized.

#### The vocabulary tables are on Google's QAT grid

- Google's QAT `embed_tokens` (tied `lm_head`) holds at most 16 distinct values per group of 32 (`embed_grid.jsonl`), and
  the `-text` repack carries it unchanged. Anchoring the step at the group's largest value reproduces only 27.9% of
  groups, so a plain Q4_0 rounding would move most of it; the trained step has to be recovered.
- The `-text-emb4` tables were built that way (`repack/embed_int4.py` in `gpu-vllm-t4-2b-w4a16`, fp16 steps).
  Against the bf16 tables (`emb4_check.jsonl`): `lm_head` and `embed_tokens` 90.9% of values exact after bf16 rounding,
  the per-layer table 89.7%, worst difference 5.3% of one grid step in all three. The linears measured 90.0% and 6.2%.
- int4 tables put everything on the GPU: 2.64 GiB of model tensors, against 1.73 GiB plus 4.38 GiB in CPU RAM before.
  The per-layer lookup and its per-token sync go away, and the embedding lookups run inside the CUDA graph.

#### Kernel v2

`w4gemv_v2.py` loads a 2-D tile of packed words, accumulates one nibble plane at a time against the matching stride-8
slice of the input, sums groups of four words and applies each group's scale once. Timed inside CUDA graphs on E2B's
13 decode shapes weighted by count per token (`kbench-v1-v2.jsonl`, 1,222.8 MiB read per token):

| Kernel | ms / token | Effective GB/s |
| --- | ---: | ---: |
| v1 (`w4gemv_v1.py`) | 29.48 | 43.5 |
| v2 | 9.87 | 129.9 |

The card reads 155.9 GB/s in a plain sum and copies at 144.8 GB/s (`membw.jsonl`), so v2 runs at 83% of measured
bandwidth. Large shapes reach 129–144 GB/s, the 256-row per-layer projections 45–57 GB/s. Relative error against
an fp32 reference is 2–4e-4 on every shape, the same as v1.

#### Attention and the compiled step

- PyTorch's default attention on sm_75 is the memory-efficient kernel (`fmha_cutlassF_f16_aligned_32x128_gmem_sm75`):
  12.96 ms per token for 35 single-query calls. `sdpa_kernel(SDPBackend.MATH)` around the decode step brings attention
  to 0.08–0.13 ms plus 2.1–2.3 ms of cuBLAS matmuls, at the cost of more small kernels.
- Compiling the decode step with Inductor (`mode="default"`) and capturing the compiled step in the same CUDA graph fuses
  norms and elementwise work: 776 kernels per token, elementwise 3.9 ms against 10.2.
- Three things had to change for it: accelerate's device-placement hooks are removed once everything is on the GPU (they
  broke the compiled graph at `accelerate/hooks.py:50`), the int4 forwards are reapplied after that (removing the hooks
  restores the original forwards), and `StaticSlidingWindowLayer` gets `StaticLayer`'s `update`, `get_seq_length` and
  `get_mask_sizes`. That layer keeps a Python-int length that a CUDA graph freezes and that Inductor guards on,
  recompiling every step. With the cache (384) no longer than the sliding window (512) it never rolls, so the two layer
  types behave identically; the script asserts the bound.

#### Prompt 4

Prompt 4 ("List three facts about Jupiter.") reaches a near-tie at its 88th token: "massive" in the reference, "high-pressure"
in the plain-math attention run. Both continue fluently ("...larger than the planet itself" / "...larger than Earth").
The compiled build, also on plain-math attention, gives the reference token, so the choice comes from small
accumulation-order differences between builds rather than from one attention backend.

#### Not measured

Prompts longer than 384 tokens with the cache (the sliding-window substitution assumes it), prompt-read speed after
optimization, accuracy suites, more than one request at a time.

#### Reproduce

From this directory, with the system `python3` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`:

```
python3 membw.py; python3 embed_grid.py; python3 emb4_check.py
python3 kbench.py v1:w4gemv_v1:w4_gemv_v1 v2:w4gemv_v2:w4_gemv
STEP_COMPILE=1 DECODE_SDPA=math python3 e2b_w4.py emb4 out.jsonl profile     # the 57 tok/s build
DECODE_SDPA=math python3 e2b_w4.py emb4 out.jsonl profile                    # without compiling the step
python3 e2b_w4.py text out.jsonl                                             # fp16 tables, per-layer table in CPU RAM
```

`text-v1.jsonl` and `emb4-v1.jsonl` were produced by an earlier revision of `e2b_w4.py` that imported the v1 kernel and
compiled the whole unpack-and-multiply for prompt reading; the logs beside each `.jsonl` are those runs' full output.
