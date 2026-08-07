# KV cache: bf16 vs fp8 on v5e-1

## Capacity

- bf16 `kv_cache_size_tokens`: **321,376**
- fp8 `kv_cache_size_tokens`: **321,376**
- Ratio: **1.000x** — **does not match the 2x prediction**.

A ratio near 1.0 means the flag was accepted but changed nothing.

## Throughput

| ctx | conc | role | KV needed | bf16 out tok/s | fp8 out tok/s | Δ | bf16 TTFT ms | fp8 TTFT ms | Δ |
|---|---|---|---|---|---|---|---|---|---|
| 128 | 1 | control | 256 | 123.26 | 120.13 | -2.5% | 15.71 | 15.85 | 0.9% |
| 128 | 8 | control | 2,048 | 738.28 | 696.73 | -5.6% | 36.55 | 65.8 | 80.0% |
| 1024 | 16 | control | 18,432 | 896.11 | 855.97 | -4.5% | 283.72 | 290.39 | 2.4% |
| 4096 | 64 | bandwidth | 270,336 | 585.92 | 570.47 | -2.6% | 1273.28 | 1302.51 | 2.3% |
| 8192 | 32 | bandwidth | 266,240 | 307.76 | 301.33 | -2.1% | 940.43 | 960.12 | 2.1% |
| 8192 | 64 | key | 532,480 | 314.44 | 307.83 | -2.1% | 11984.94 | 12249.23 | 2.2% |
| 16000 | 32 | key | 516,096 | 166.76 | 163.75 | -1.8% | 10933.51 | 11136.44 | 1.9% |
| 16000 | 64 | saturated | 1,032,192 | 166.69 | 163.69 | -1.8% | 34676.59 | 35314.95 | 1.8% |

Control drift (should be ~0): max |Δ| = **5.6%**.

BANDWIDTH cells (fit in bf16, but KV is ~35-40% of bytes moved): Δ = -2.6%, -2.1% — predicted ~+15-18% from halved KV traffic alone, with no capacity effect.

KEY cells (over bf16 capacity, inside fp8): Δ = -2.1%, -1.8%.

Monotonicity check — mean Δ by role: control -4.2%, bandwidth -2.4%, key -2.0%. Increasing across the three is the signature of a real bytes-moved effect; a flat or non-monotone profile is not.

## Quality

Identical outputs: **8/9** at temperature 0.

| Probe | Identical | First divergence (char) | bf16 needle | fp8 needle |
|---|---|---|---|---|
| `short-factual` | yes | - | - | - |
| `short-arithmetic` | yes | - | - | - |
| `short-code` | yes | - | - | - |
| `short-instruction` | yes | - | - | - |
| `short-multilingual` | yes | - | - | - |
| `short-regional` | no | 0 | - | - |
| `needle-2000` | yes | - | yes | yes |
| `needle-8000` | yes | - | yes | yes |
| `needle-14000` | yes | - | yes | yes |

Long-context recall lost under fp8: **0/3** probes.
