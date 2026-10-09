## bf16 per size

| size | GiB | KV tokens | c1/128 | c8/1024 | c64/1024 | TPOT c1/128 ms |
|---|---:|---:|---:|---:|---:|---:|
| E2B | 9.42 | 9,045,060 | 343 | 1,795 | 8,951 | 2.89 |
| E4B | 14.79 | 2,794,448 | 239 | 1,284 | 5,778 | 4.16 |
| 12B | 22.83 | 439,003 | 134 | 677 | 2,248 | 7.44 |
| 26B-A4B | 48.54 | 551,865 | 224 | 979 | 2,585 | 4.42 |
| 31B | 58.99 | 121,290 | 58 | 306 | 950 | 17.19 |

## fp8 and q4w4a16 against bf16 per size

| size | fp8 c1/128 x | fp8 min | fp8 max | fp8 cells >= 1.00 | fp8 median | q4 c1/128 x | q4 min | q4 max | w8a8 min | w8a8 max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E2B | 0.75 | 0.75 | 1.08 | 5 of 9 | 1.00 | 0.14 | 0.14 | 0.63 | 0.29 | 0.87 |
| E4B | 0.89 | 0.89 | 1.19 | 6 of 9 | 1.12 | 0.14 | 0.14 | 0.54 | 0.31 | 0.82 |
| 12B | 1.13 | 1.12 | 1.41 | 9 of 9 | 1.23 | 0.16 | 0.16 | 0.57 | 0.39 | 0.93 |
| 26B-A4B | 0.99 | 0.99 | 1.23 | 7 of 9 | 1.13 | 0.27 | 0.27 | 0.69 | - | - |
| 31B | 1.23 | 1.20 | 1.45 | 9 of 9 | 1.32 | 0.19 | 0.19 | 0.59 | - | - |

## Every ok arm per size: weights, KV, c1/128, c8/1024, c64/1024, ratio range, weights as % of bf16, KV x bf16

| size | arm | GiB | % bf16 | KV tokens | KV x bf16 | c1/128 | c8/1024 | c64/1024 | min | max | kernels |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E2B | bf16 | 9.42 | 100% | 9,045,060 | 1.00 | 343 | 1,795 | 8,951 | 1.00 | 1.00 | - |
| E2B | q4w4a16 | 6.32 | 67% | 9,070,174 | 1.00 | 48 | 367 | 3,288 | 0.14 | 0.63 | TritonW4A16LinearKernel |
| E2B | w8a8 | 7.07 | 75% | 9,025,606 | 1.00 | 100 | 757 | 4,717 | 0.29 | 0.87 | TritonInt8ScaledMMLinearKernel |
| E2B | fp8 | 7.07 | 75% | 9,017,531 | 1.00 | 259 | 1,827 | 9,700 | 0.75 | 1.08 | RowWiseTorchFP8ScaledMMLinearKernel |
| E2B | q4w4a16ple4 | 3.18 | 34% | 9,449,863 | 1.04 | 48 | 366 | 3,264 | 0.14 | 0.63 | TritonW4A16LinearKernel |
| E2B | q4w4a16emb4 | 2.85 | 30% | 9,475,223 | 1.05 | 47 | 365 | 3,285 | 0.14 | 0.63 | TritonW4A16LinearKernel |
| E2B | w8a8emb4 | 3.60 | 38% | 9,419,957 | 1.04 | 99 | 746 | 4,680 | 0.29 | 0.79 | TritonInt8ScaledMMLinearKernel, TritonW4A16LinearKernel |
| E2B | fp8emb4 | 3.61 | 38% | 9,422,004 | 1.04 | 247 | 1,742 | 9,450 | 0.72 | 1.08 | RowWiseTorchFP8ScaledMMLinearKernel, TritonW4A16LinearKernel |
| E2B | fp8fnuz | 7.07 | 75% | 9,017,531 | 1.00 | 259 | 1,819 | 9,739 | 0.75 | 1.09 | RowWiseTorchFP8ScaledMMLinearKernel |
| E2B | fp8fnuzemb4 | 3.61 | 38% | 9,422,004 | 1.04 | 247 | 1,752 | 9,500 | 0.72 | 1.06 | RowWiseTorchFP8ScaledMMLinearKernel, TritonW4A16LinearKernel |
| E4B | bf16 | 14.79 | 100% | 2,794,448 | 1.00 | 239 | 1,284 | 5,778 | 1.00 | 1.00 | - |
| E4B | q4w4a16 | 8.83 | 60% | 2,835,823 | 1.01 | 33 | 254 | 2,183 | 0.14 | 0.54 | TritonW4A16LinearKernel |
| E4B | w8a8 | 10.46 | 71% | 2,806,038 | 1.00 | 74 | 555 | 3,313 | 0.31 | 0.82 | TritonInt8ScaledMMLinearKernel |
| E4B | fp8 | 10.46 | 71% | 2,803,516 | 1.00 | 212 | 1,456 | 6,662 | 0.89 | 1.19 | RowWiseTorchFP8ScaledMMLinearKernel |
| E4B | q4w4a16ple4 | 5.06 | 34% | 2,985,126 | 1.07 | 33 | 253 | 2,177 | 0.14 | 0.54 | TritonW4A16LinearKernel |
| E4B | q4w4a16emb4 | 4.51 | 30% | 3,001,791 | 1.07 | 33 | 250 | 2,157 | 0.14 | 0.54 | TritonW4A16LinearKernel |
| E4B | w8a8emb4 | failed | | | | | | | | | |
| E4B | fp8emb4 | 6.15 | 42% | 2,970,362 | 1.06 | 193 | 1,355 | 6,436 | 0.81 | 1.18 | RowWiseTorchFP8ScaledMMLinearKernel, TritonW4A16LinearKernel |
| 12B | bf16 | 22.83 | 100% | 439,003 | 1.00 | 134 | 677 | 2,248 | 1.00 | 1.00 | - |
| 12B | q4w4a16 | 8.18 | 36% | 485,062 | 1.10 | 22 | 159 | 1,109 | 0.16 | 0.57 | TritonW4A16LinearKernel |
| 12B | w8a8 | 12.65 | 55% | 471,268 | 1.07 | 53 | 371 | 1,747 | 0.39 | 0.93 | TritonInt8ScaledMMLinearKernel |
| 12B | fp8 | 12.65 | 55% | 470,708 | 1.07 | 152 | 911 | 2,767 | 1.12 | 1.41 | RowWiseTorchFP8ScaledMMLinearKernel |
| 12B | q4w4a16emb4 | 7.36 | 32% | 488,674 | 1.11 | 21 | 157 | 1,102 | 0.16 | 0.57 | TritonW4A16LinearKernel |
| 12B | w8a8emb4 | failed | | | | | | | | | |
| 12B | fp8emb4 | 11.83 | 52% | 474,327 | 1.08 | 139 | 863 | 2,725 | 1.04 | 1.29 | RowWiseTorchFP8ScaledMMLinearKernel, TritonW4A16LinearKernel |
| 26B-A4B | bf16 | 48.54 | 100% | 551,865 | 1.00 | 224 | 979 | 2,585 | 1.00 | 1.00 | - |
| 26B-A4B | q4w4a16 | 14.80 | 30% | 715,856 | 1.30 | 60 | 382 | 1,772 | 0.27 | 0.69 | CompressedTensorsWNA16MoEMethod, TritonW4A16LinearKernel |
| 26B-A4B | fp8 | 24.72 | 51% | 667,691 | 1.21 | 222 | 1,132 | 3,162 | 0.99 | 1.23 | RowWiseTorchFP8ScaledMMLinearKernel |
| 31B | bf16 | 58.99 | 100% | 121,290 | 1.00 | 58 | 306 | 950 | 1.00 | 1.00 | - |
| 31B | q4w4a16 | 18.70 | 32% | 170,112 | 1.40 | 11 | 80 | 512 | 0.19 | 0.59 | TritonW4A16LinearKernel |
| 31B | fp8 | 30.63 | 52% | 155,730 | 1.28 | 71 | 432 | 1,219 | 1.20 | 1.45 | RowWiseTorchFP8ScaledMMLinearKernel |

## fp8 / bf16 per cell, every size

| size | c1/128 | c1/1024 | c1/8192 | c8/128 | c8/1024 | c8/8192 | c64/128 | c64/1024 | c64/8192 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E2B | 0.75 | 0.75 | 0.81 | 1.00 | 1.02 | 1.05 | 1.02 | 1.08 | 0.86 |
| E4B | 0.89 | 0.90 | 0.92 | 1.12 | 1.13 | 1.14 | 1.11 | 1.15 | 1.19 |
| 12B | 1.13 | 1.14 | 1.12 | 1.41 | 1.35 | 1.30 | 1.22 | 1.23 | 1.25 |
| 26B-A4B | 0.99 | 0.99 | 1.00 | 1.13 | 1.16 | 1.09 | 1.23 | 1.22 | 1.15 |
| 31B | 1.23 | 1.24 | 1.20 | 1.45 | 1.41 | 1.36 | 1.34 | 1.28 | 1.32 |

## emb4 against the same format with bf16 tables, per-cell ratio range

- E2B fp8emb4/fp8: 0.954-1.255, median 0.968
- E2B q4w4a16emb4/q4w4a16: 0.990-1.006, median 0.996
- E2B w8a8emb4/w8a8: 0.905-0.992, median 0.986
- E4B fp8emb4/fp8: 0.913-0.986, median 0.935
- E4B q4w4a16emb4/q4w4a16: 0.984-0.996, median 0.986
- 12B fp8emb4/fp8: 0.919-0.988, median 0.948
- 12B q4w4a16emb4/q4w4a16: 0.983-0.997, median 0.987

## Speed per GiB loaded at one request (c1/128 tok/s per GiB of weights)

- E2B: bf16 36.5, fp8 36.6, q4w4a16 7.6
- E4B: bf16 16.2, fp8 20.3, q4w4a16 3.7
- 12B: bf16 5.9, fp8 12.0, q4w4a16 2.6
- 26B-A4B: bf16 4.6, fp8 9.0, q4w4a16 4.1
- 31B: bf16 1.0, fp8 2.3, q4w4a16 0.6

## Boot seconds per arm

- E2B: bf16 205.4, q4w4a16 161.6, w8a8 225.6, fp8 181.9, q4w4a16ple4 181.4, q4w4a16emb4 159.1, w8a8emb4 205.0, fp8emb4 180.7, fp8fnuz 181.7, fp8fnuzemb4 159.9
- E4B: bf16 204.3, q4w4a16 182.5, w8a8 250.8, fp8 182.9, q4w4a16ple4 182.6, q4w4a16emb4 159.9, w8a8emb4 635.4, fp8emb4 183.7
- 12B: bf16 206.1, q4w4a16 183.6, w8a8 228.3, fp8 182.2, q4w4a16emb4 161.0, w8a8emb4 183.3, fp8emb4 183.4
- 26B-A4B: bf16 205.5, q4w4a16 183.7, fp8 206.2
- 31B: bf16 204.5, q4w4a16 205.9, fp8 183.1

## Repeat spread (cv %) per size, every ok arm and cell

- E2B: max cv 15.68% (fp8fnuz c64/8192), cells over 5%: 8 of 90
- E4B: max cv 3.9% (fp8emb4 c64/128), cells over 5%: 0 of 63
- 12B: max cv 1.95% (fp8emb4 c64/128), cells over 5%: 0 of 54
- 26B-A4B: max cv 5.69% (fp8 c8/128), cells over 5%: 1 of 27
- 31B: max cv 0.46% (bf16 c64/128), cells over 5%: 0 of 27

## q4w4a16 / bf16 by request count (mean over prompt lengths) and int4-table savings

- E2B: by requests 1: 0.16, 8: 0.23, 64: 0.44 | by prompt 128: 0.22, 1024: 0.24, 8192: 0.37
- E4B: by requests 1: 0.16, 8: 0.22, 64: 0.41 | by prompt 128: 0.22, 1024: 0.24, 8192: 0.34
- 12B: by requests 1: 0.18, 8: 0.26, 64: 0.48 | by prompt 128: 0.25, 1024: 0.30, 8192: 0.37
- 26B-A4B: by requests 1: 0.30, 8: 0.41, 64: 0.67 | by prompt 128: 0.43, 1024: 0.45, 8192: 0.49
- 31B: by requests 1: 0.20, 8: 0.29, 64: 0.53 | by prompt 128: 0.29, 1024: 0.33, 8192: 0.39
- E4B fp8 - fp8emb4 = 4.31 GiB
- 12B fp8 - fp8emb4 = 0.82 GiB
- cells where w8a8 beats fp8: [('2b', 64, 8192)]
