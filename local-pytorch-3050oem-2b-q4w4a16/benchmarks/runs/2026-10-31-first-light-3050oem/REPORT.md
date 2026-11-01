# 2026-10-31 — first light on the RTX 3050 OEM

The int4 engine (`e2b_engine.py`, as on the 1650 Ti and L4) on the RTX 3050 OEM in the desktop `gpu`: driver 615.78.08,
torch 2.14.1+cu130, Triton 3.8.0, transformers 5.16.1, compressed-tensors 0.17.0. Method: `depth_bench.py`, seeded prompts
from the 1650 Ti run, 63 greedy tokens decoded with stop tokens off.

#### Decode against llama.cpp

llama.cpp `7182ef71e`, the exact Q4_0 GGUF (SHA-256 `419db9a6…`), `-ngl 99 -fa 1 -t 6`, no drafter, same card and night
(`llamacpp-depth.json`: `llama-bench -p 512 -n 64`, 2 repeats, stddev under 1). Depths are nominal for llama-bench.

| Tokens held | This engine | llama.cpp | Ratio | Recall | File |
| ---: | ---: | ---: | ---: | --- | --- |
| 61 | **127.03** | 118.21 | 1.07x | ✅ | `short-tuned*.jsonl` (126.88, 127.19) |
| 4,081 | **119.55** | 112.43 | 1.06x | ✅ | `short-tuned*.jsonl` (119.48, 119.62) |
| 32,761 | **84.59** | 91.83 | 0.92x | ❌ (319550 for 713468) | `short-tuned.jsonl` |
| 130,981 | **42.22** | 56.61 | 0.75x | ❌ | `deep.jsonl` (1650 Ti tilings) |

`llamacpp-bench.json` with `-n 128 -r 5`: 118.81 and 118.62 tok/s, 1.07x. The 32,761-token miss is the same wrong code the
L4 and 1650 Ti gave on that prompt in fp16; the same depth in `deep.jsonl` (a different seeded prompt) recalled it.
Peak allocated 3,398 MiB at 131k; 2,489 MiB of weights, 774 MiB of K/V.

llama.cpp's tokens per second with the exact Q4_0 drafter on this card were measured 2026-10-14 at 151 tok/s on a
different prompt set and commit, so they are not in this table. This engine has no drafter.

#### Prompt reads

| Tokens | fp16 tok/s | fp32 tok/s |
| ---: | ---: | ---: |
| 61 | 96.6–473.6 (first read compiles) | 381.6 |
| 4,081 | 1,174.9–1,786.0 | 856.0 |
| 32,761 | 1,217.1–1,266.9 | 712.9 |
| 130,981 | 532.0 | not run |

llama-bench reads 512 tokens at each depth (3,976.8 / 3,317.0 / 1,670.0 / 620.2 tok/s at 0 / 4k / 32k / 130k); this table
reads the whole prompt from position 0, so the two do not compare directly.

#### The memory ceiling

`cold.jsonl`: a 1 GiB device copy runs at **207.5 GB/s** read plus write. Shapes of 1536×256 packed and larger, with
weights rotated past the cache, run at 151.5–209.1 GB/s.

`profile.json`: 7.878 ms wall per step, 537 kernels: `_w4_gemv` 6.928 ms in 212 calls, decode attention 0.450 ms. At the
1.282 GB per token counted on the 1650 Ti, the GEMV moves 185 GB/s, 0.89 of the copy rate, the same fraction as the
L4's 0.886. At the copy rate a step takes about 7.1 ms, about 140 tok/s: the arithmetic ceiling for this engine.

#### Tilings

`tune.jsonl`: the autotuner beat the 1650 Ti tilings by 0–14% per shape, under 4% on every shape above 512 rows.
Pinning the wins of 3% or more moved short-context decode from 126.71 to 126.88–127.19 and 119.28 to 119.48–119.62.

#### Load time

49.0 s in fp32, 228–229 s in fp16 on two later runs. The first run took 385 s with the downloads. Cause not investigated.

#### Against the other cards

| Card | Copy GB/s | Decode at short context | Decode at 131k |
| --- | ---: | ---: | ---: |
| GTX 1650 Ti Max-Q | 156 | 87.84 | 21.81 |
| RTX 3050 OEM | 207.5 | 127.03 | 42.22 |
| L4 | 241.2 | 141.35 | 51.41 |

Different hosts and torch versions.
