| requests | prompt tokens | AITER tok/s | stock tok/s | AITER / stock | AITER TPOT ms | stock TPOT ms | stock / 2026-10-07 | cv AITER % | cv stock % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 148.3 | 152.2 | 0.974 | 6.7 | 6.55 | 1.004 | 0.21 | 0.16 |
| 1 | 1,024 | 141.0 | 144.3 | 0.977 | 6.97 | 6.82 | 1.001 | 0.18 | 0.18 |
| 1 | 8,192 | 100.6 | 102.7 | 0.980 | 9.02 | 8.83 | 1.002 | 0.2 | 0.11 |
| 8 | 128 | 1,048.0 | 1,063.8 | 0.985 | 7.52 | 7.43 | 0.979 | 0.78 | 0.47 |
| 8 | 1,024 | 896.5 | 917.1 | 0.978 | 8.32 | 8.14 | 1.007 | 0.33 | 0.32 |
| 8 | 8,192 | 477.3 | 480.0 | 0.994 | 11.81 | 11.72 | 0.983 | 0.78 | 0.48 |
| 64 | 128 | 4,780.2 | 4,874.2 | 0.981 | 12.9 | 12.68 | 0.984 | 0.14 | 0.28 |
| 64 | 1,024 | 2,694.7 | 2,731.0 | 0.987 | 21.21 | 20.24 | 0.987 | 0.15 | 0.23 |
| 64 | 8,192 | 849.4 | 857.7 | 0.990 | 69.22 | 68.54 | 0.991 | 0.72 | 0.25 |

AITER/stock: min 0.974 max 0.994 median 0.981; cells slower: 9 of 9
loss: 0.6% to 2.6%
stock vs 2026-10-07 sweep: 0.979-1.007, max deviation 2.1%
max repeat cv: AITER 0.78%, stock 0.48%

untuned AITER GEMM lines: 924; distinct (N, K): 6; distinct M per shape: 77, M from 1 to 262144
- N:8192, K:3840, q_dtype_w:torch.float8_e4m3fnuz: 154
- N:3840, K:4096, q_dtype_w:torch.float8_e4m3fnuz: 154
- N:30720, K:3840, q_dtype_w:torch.float8_e4m3fnuz: 154
- N:3840, K:15360, q_dtype_w:torch.float8_e4m3fnuz: 154
- N:9216, K:3840, q_dtype_w:torch.float8_e4m3fnuz: 154
- N:3840, K:8192, q_dtype_w:torch.float8_e4m3fnuz: 154
