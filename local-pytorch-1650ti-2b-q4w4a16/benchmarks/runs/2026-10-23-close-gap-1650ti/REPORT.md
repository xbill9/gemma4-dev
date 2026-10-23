# 2026-10-23 — past llama.cpp: decode attention kernel, merged projections, int4 kernel v3–v5

Same host, card, stack and checkpoints as `../2026-10-23-lmhead-kernel-1650ti/` (GTX 1650 Ti Max-Q, torch 2.13.0+cu130,
transformers 5.16.1, Triton 3.7.1; decoder linears from `-text`, vocabulary tables int4 from `-text-emb4`). Same eight
prompts, greedy, up to 128 new tokens, 384-token static cache, fp16, decode step compiled with Inductor and replayed as a
CUDA graph. Decode tok/s excludes the prompt read; ranges are over the eight prompts. "Identical" compares greedy tokens
with `reference-text-v1.jsonl` (the fp16-table build).

#### llama.cpp on the same card, same day

`llamacpp-bench.json`: `llama-bench` at commit `7182ef7`, the rig's model (`gemma-4-E2B-it-q4_0-exact.gguf`, SHA-256
`419db9a6…`), `-ngl 99 -fa 1 -t 6`, 128 generated tokens, 5 repeats:

| Context before generating | tok/s | stddev |
| ---: | ---: | ---: |
| 0 | 82.29 | 0.06 |
| 128 | 81.98 | 0.05 |

#### Result

| Build | Decode tok/s | vs llama.cpp | GPU ms / token (kernels / replay) | Kernels / token | Peak MiB | Identical | File |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Previous run: math attention, kernel v2 | 57.31–57.61 | 0.70x | 16.1 / 17.5 | 776 | 2,767–2,769 | 8 of 8 | `../2026-10-23-lmhead-kernel-1650ti/emb4-v2c-math.jsonl` |
| Triton decode attention, kernel v2 | 77.61–78.63 | 0.95x | 11.53 / 12.86 | 578 | 2,751–2,753 | 8 of 8 | `w4decode-c.jsonl` |
| + merged q/k/v and gate/up, kernel v2 | 77.21–78.14 | 0.95x | 11.35 / 12.90 | 513 | 2,788–2,793 | 8 of 8 | `merge-c.jsonl` |
| Triton decode attention, kernel v3 | 78.34–79.12 | 0.96x | 11.30 / 12.71 | 578 | 2,751–2,753 | 8 of 8 | `kv3-merge0.jsonl` |
| + merged q/k/v and gate/up, kernel v3 | 79.29–80.25 | 0.97x | 11.34 / 12.56 | 513 | 2,788–2,793 | 8 of 8 | `kv3-merge1.jsonl` |
| **+ merged, kernel v5, run 1** | **89.84–90.57** | **1.10x** | **10.53 / 11.08** | **513** | **2,788–2,793** | **8 of 8** | `kv5-merge1-r1.jsonl` |
| **+ merged, kernel v5, run 2** | **89.76–90.52** | **1.10x** | **10.56 / 11.08** | **513** | **2,788–2,793** | **8 of 8** | `kv5-merge1-r2.jsonl` |

"vs llama.cpp" divides the range midpoint by 82.1 tok/s. llama-bench starts from 0 or 128 tokens of context; these
prompts start from 15–40 and generate up to 128.

#### Decode attention kernel (`w4attn.py`)

Plain-math SDPA converted the whole 384-slot K/V cache to fp32 and copied it out for all eight query heads in every layer,
then multiplied in fp32: 5.40 ms per token (the sum of those kernels in `../2026-10-23-lmhead-kernel-1650ti/emb4-v2c-math.jsonl`). `w4attn.py` registers a transformers attention function (`w4decode`):
one Triton program per query head streams the fp16 cache once, applies the boolean mask transformers builds and keeps
an online softmax in fp32; prompt reads go to SDPA. Against fp32 SDPA (`attn_check.jsonl`): relative error 2.6–3.2e-4,
22–47 µs per call at head size 256 and 512. In the model: 0.85–0.88 ms per token for 35 calls.

#### Merged projections

`MERGE=1` concatenates gate+up in every MLP (35) and q+k+v in every layer that computes its own K/V (15), one int4 GEMV
each; KV-shared layers keep q alone. Measured in isolation (`fuse_bench.jsonl`): 4.97 → 4.52 ms per token. In the
model it removes 65 kernels per token. With kernel v2 the replay did not get faster; with v3 it gained about 1 tok/s.

#### Int4 kernel v3 and v4

GPU time over E2B's 13 decode shapes weighted by count per token, CUDA-graph timed (`kbench-*.jsonl`):

| Kernel | ms / token | Effective GB/s | Change |
| --- | ---: | ---: | --- |
| v2 | 9.98 | 128.5 | |
| v3 | 9.58–9.83 | 130.4–133.9 | nibble to float by OR-ing into the bits of 2^23 and subtracting 2^23 + 8 (exact), instead of an int-to-float conversion |
| v4 | 9.24 | 138.8 | v3 with a wider autotune search: 6 tile heights, 5 group widths, 1–8 warps, 1–4 pipeline stages |

The card reads 155.9 GB/s (`../2026-10-23-lmhead-kernel-1650ti/membw.jsonl`), a floor of 8.2 ms for these 1,222.8 MiB.
v4's autotuner picks 8 rows by 16 groups with one warp for most tall shapes; `lm_head` reaches 152.5 GB/s. Relative
error is unchanged (1.8–3.7e-4). Two v3 timings differ by 0.25 ms between benchmark processes.

`w4gemv_v5.py` fixes v4's chosen configuration per shape: no start-up autotuning, and the same tiling, so the same
summation order, on every run. Shapes not in its table (the merged projections) use v4's most common choice, 8 rows by
16 groups, one warp, one stage. In isolation v5 takes 9.12 ms per token against v4's 9.30 in the same process
(`kbench-v4-v5.jsonl`). In the model the int4 kernels take 9.03 ms per token against v3's 9.79, and the time the GPU
waits between kernels inside the graph falls from 1.22 to 0.55 ms; both v5 runs give the same tokens.

#### Reproduce

From this directory, system `python3`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`:

```
python3 attn_check.py; python3 fuse_bench.py
python3 kbench.py v2:w4gemv_v2:w4_gemv v3:w4gemv_v3:w4_gemv
python3 kbench.py v3:w4gemv_v3:w4_gemv v4:w4gemv_v4:w4_gemv
MERGE=1 ATTN=w4decode STEP_COMPILE=1 python3 e2b_w4.py emb4 out.jsonl profile    # kernel v5 (default): 90 tok/s
KERNEL=v3 MERGE=1 ATTN=w4decode STEP_COMPILE=1 python3 e2b_w4.py emb4 out.jsonl profile    # 79–80 tok/s
python3 kbench.py v4:w4gemv_v4:w4_gemv v5:w4gemv_v5:w4_gemv
python3 summarize.py out.jsonl
~/llama.cpp/build/bin/llama-bench -m <gguf> -ngl 99 -fa 1 -p 0 -n 128 -d 0,128 -r 5 -t 6 -o json
```

`w4decode-c.jsonl` and `merge-c.jsonl` were produced before `e2b_w4.py` gained the `KERNEL` switch, with kernel v2.
