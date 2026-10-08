## Per arm: weights, KV, output tok/s at c1/in128, c8/in1024, c64/in1024, ratio range vs bf16

| arm | GiB | % bf16 GiB | KV tokens | KV x bf16 | c1/128 | c8/1024 | c64/1024 | ratio min | ratio max | boot s | kernels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| bf16 | 9.42 | 100% | 9,045,060 | 1.000 | 343 | 1,795 | 8,951 | 1.00 | 1.00 | 205 | - |
| fp8 | 7.07 | 75% | 9,017,531 | 0.997 | 259 | 1,827 | 9,700 | 0.75 | 1.08 | 182 | RowWiseTorchFP8ScaledMMLinearKernel |
| fp8fnuz | 7.07 | 75% | 9,017,531 | 0.997 | 259 | 1,819 | 9,739 | 0.75 | 1.09 | 182 | RowWiseTorchFP8ScaledMMLinearKernel |
| fp8emb4 | 3.61 | 38% | 9,422,004 | 1.042 | 247 | 1,742 | 9,450 | 0.72 | 1.08 | 181 | RowWiseTorchFP8ScaledMMLinearKernel, TritonW4A16LinearKernel |
| fp8fnuzemb4 | 3.61 | 38% | 9,422,004 | 1.042 | 247 | 1,752 | 9,500 | 0.72 | 1.06 | 160 | RowWiseTorchFP8ScaledMMLinearKernel, TritonW4A16LinearKernel |
| w8a8 | 7.07 | 75% | 9,025,606 | 0.998 | 100 | 757 | 4,717 | 0.29 | 0.87 | 226 | TritonInt8ScaledMMLinearKernel |
| w8a8emb4 | 3.60 | 38% | 9,419,957 | 1.041 | 99 | 746 | 4,680 | 0.29 | 0.79 | 205 | TritonInt8ScaledMMLinearKernel, TritonW4A16LinearKernel |
| q4w4a16 | 6.32 | 67% | 9,070,174 | 1.003 | 48 | 367 | 3,288 | 0.14 | 0.63 | 162 | TritonW4A16LinearKernel |
| q4w4a16ple4 | 3.18 | 34% | 9,449,863 | 1.045 | 48 | 366 | 3,264 | 0.14 | 0.63 | 181 | TritonW4A16LinearKernel |
| q4w4a16emb4 | 2.85 | 30% | 9,475,223 | 1.048 | 47 | 365 | 3,285 | 0.14 | 0.63 | 159 | TritonW4A16LinearKernel |

## Full grid, output tok/s (ratio to bf16)

| arm | c1/128 | c1/1024 | c1/8192 | c8/128 | c8/1024 | c8/8192 | c64/128 | c64/1024 | c64/8192 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16 | 343 (1.00) | 321 (1.00) | 227 (1.00) | 1,945 (1.00) | 1,795 (1.00) | 1,055 (1.00) | 11,866 (1.00) | 8,951 (1.00) | 2,255 (1.00) |
| fp8 | 259 (0.75) | 242 (0.75) | 184 (0.81) | 1,951 (1.00) | 1,827 (1.02) | 1,111 (1.05) | 12,080 (1.02) | 9,700 (1.08) | 1,935 (0.86) |
| fp8fnuz | 259 (0.75) | 245 (0.77) | 185 (0.81) | 1,956 (1.01) | 1,819 (1.01) | 1,111 (1.05) | 12,129 (1.02) | 9,739 (1.09) | 2,244 (0.99) |
| fp8emb4 | 247 (0.72) | 234 (0.73) | 178 (0.78) | 1,865 (0.96) | 1,742 (0.97) | 1,077 (1.02) | 11,766 (0.99) | 9,450 (1.06) | 2,427 (1.08) |
| fp8fnuzemb4 | 247 (0.72) | 235 (0.73) | 178 (0.78) | 1,876 (0.96) | 1,752 (0.98) | 1,075 (1.02) | 11,759 (0.99) | 9,500 (1.06) | 2,062 (0.91) |
| w8a8 | 100 (0.29) | 98 (0.31) | 87 (0.38) | 781 (0.40) | 757 (0.42) | 585 (0.55) | 5,355 (0.45) | 4,717 (0.53) | 1,962 (0.87) |
| w8a8emb4 | 99 (0.29) | 97 (0.30) | 85 (0.38) | 772 (0.40) | 746 (0.42) | 577 (0.55) | 5,271 (0.44) | 4,680 (0.52) | 1,776 (0.79) |
| q4w4a16 | 48 (0.14) | 47 (0.15) | 44 (0.19) | 374 (0.19) | 367 (0.20) | 311 (0.29) | 3,764 (0.32) | 3,288 (0.37) | 1,416 (0.63) |
| q4w4a16ple4 | 48 (0.14) | 47 (0.15) | 44 (0.19) | 374 (0.19) | 366 (0.20) | 311 (0.29) | 3,754 (0.32) | 3,264 (0.36) | 1,426 (0.63) |
| q4w4a16emb4 | 47 (0.14) | 47 (0.15) | 44 (0.19) | 373 (0.19) | 365 (0.20) | 310 (0.29) | 3,762 (0.32) | 3,285 (0.37) | 1,425 (0.63) |

## Pairs, per-cell ratio of output tok/s

- fp8fnuz / fp8: all 9 cells 0.995-1.160; 8 cells without c64/8192 0.995-1.014, median 1.003
- fp8fnuzemb4 / fp8emb4: all 9 cells 0.849-1.006; 8 cells without c64/8192 0.998-1.006, median 1.002
- fp8emb4 / fp8: all 9 cells 0.954-1.255; 8 cells without c64/8192 0.954-0.974, median 0.968
- w8a8emb4 / w8a8: all 9 cells 0.905-0.992; 8 cells without c64/8192 0.984-0.992, median 0.986
- q4w4a16emb4 / q4w4a16: all 9 cells 0.990-1.006; 8 cells without c64/8192 0.990-0.999, median 0.996
- q4w4a16ple4 / q4w4a16: all 9 cells 0.993-1.007; 8 cells without c64/8192 0.993-0.999, median 0.997
- fp8 / w8a8: all 9 cells 0.986-2.585; 8 cells without c64/8192 1.901-2.585, median 2.256
- fp8 / q4w4a16: all 9 cells 1.366-5.420; 8 cells without c64/8192 2.951-5.420, median 4.172
- w8a8 / q4w4a16: all 9 cells 1.385-2.097; 8 cells without c64/8192 1.423-2.097, median 1.957

## bf16 c64/8192 across arms (the noisiest cell)

bf16 2,255, fp8 1,935, fp8fnuz 2,244, fp8emb4 2,427, fp8fnuzemb4 2,062, w8a8 1,962, w8a8emb4 1,776, q4w4a16 1,416, q4w4a16ple4 1,426, q4w4a16emb4 1,425

## TPOT median ms at c1/in128

bf16 2.89, fp8 3.83, fp8fnuz 3.83, fp8emb4 4.03, fp8fnuzemb4 4.02, w8a8 9.94, w8a8emb4 10.11, q4w4a16 20.91, q4w4a16ple4 20.95, q4w4a16emb4 21.12

## Weights against the QAT values

- q4w4a16: 58,650,624 groups, 0 off the grid, 90.0% bit-identical, max rel err 0.0109
- w8a8: 276 tensors, rel err 0.59%-1.71%, mean of means 0.90%
- fp8: 276 modules, rel RMS 2.638%, max row-relative 3.57%
- fp8fnuz: 276 modules, rel RMS 2.642%, max row-relative 3.33%

## Repeat spread (cv %) per cell, from each arm's report

- bf16: c64/8192 cv 9.99%, other cells max cv 1.51%
- fp8: c64/8192 cv 15.08%, other cells max cv 3.1%
- fp8fnuz: c64/8192 cv 15.68%, other cells max cv 1.8%
- fp8emb4: c64/8192 cv 12.41%, other cells max cv 10.74%
- fp8fnuzemb4: c64/8192 cv 10.1%, other cells max cv 4.06%
- w8a8: c64/8192 cv 5.15%, other cells max cv 1.02%
- w8a8emb4: c64/8192 cv 0.97%, other cells max cv 1.21%
- q4w4a16: c64/8192 cv 1.37%, other cells max cv 0.12%
- q4w4a16ple4: c64/8192 cv 0.11%, other cells max cv 0.32%
- q4w4a16emb4: c64/8192 cv 0.86%, other cells max cv 0.71%
