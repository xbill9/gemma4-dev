# GEMM at decode shapes — AMD Instinct MI300X VF (gfx942:sramecc+:xnack-), torch 2.12.0+rocm10.0.0

mode=graph, rotate_bytes=1073741824 (0 = hot weights), repeats=3, median of repeats.

| shape | N x K | M | dtype | status | us/call | TFLOP/s | weight GB/s | % of 3.76 TB/s | x bf16 | cv % |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control_8192 | 8192x8192 | 1 | bf16 | ok | 46.7 | 2.9 | 2876 | 76 | 1.00 | 0.2 |
| control_8192 | 8192x8192 | 1 | fp16 | ok | 46.9 | 2.9 | 2859 | 76 | 0.99 | 0.1 |
| control_8192 | 8192x8192 | 1 | fp8 | ok | 26.1 | 5.1 | 2567 | 68 | 1.79 | 0.3 |
| control_8192 | 8192x8192 | 1 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 1 -->
| control_8192 | 8192x8192 | 8 | bf16 | ok | 44.8 | 24.0 | 2999 | 80 | 1.00 | 0.0 |
| control_8192 | 8192x8192 | 8 | fp16 | ok | 39.0 | 27.5 | 3440 | 91 | 1.15 | 0.4 |
| control_8192 | 8192x8192 | 8 | fp8 | ok | 26.9 | 40.0 | 2498 | 66 | 1.67 | 0.2 |
| control_8192 | 8192x8192 | 8 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 8 -->
| control_8192 | 8192x8192 | 64 | bf16 | ok | 53.1 | 161.6 | 2525 | 67 | 1.00 | 0.0 |
| control_8192 | 8192x8192 | 64 | fp16 | ok | 54.0 | 159.1 | 2485 | 66 | 0.98 | 0.1 |
| control_8192 | 8192x8192 | 64 | fp8 | ok | 31.1 | 276.0 | 2156 | 57 | 1.71 | 0.2 |
| control_8192 | 8192x8192 | 64 | int8 | ok | 275.9 | 31.1 | 243 | 6 | 0.19 | 0.0 |
| control_8192 | 8192x8192 | 8192 | bf16 | ok | 1794.5 | 612.7 | 75 | 2 | 1.00 | 0.5 |
| control_8192 | 8192x8192 | 8192 | fp16 | ok | 1892.0 | 581.1 | 71 | 2 | 0.95 | 0.3 |
| control_8192 | 8192x8192 | 8192 | fp8 | ok | 1007.1 | 1091.8 | 67 | 2 | 1.78 | 0.6 |
| control_8192 | 8192x8192 | 8192 | int8 | ok | 4479.4 | 245.5 | 15 | 0 | 0.40 | 0.2 |
| qkv_sliding | 2560x1536 | 1 | bf16 | ok | 5.6 | 1.4 | 1408 | 37 | 1.00 | 0.3 |
| qkv_sliding | 2560x1536 | 1 | fp16 | ok | 5.5 | 1.4 | 1418 | 38 | 1.01 | 0.3 |
| qkv_sliding | 2560x1536 | 1 | fp8 | ok | 6.5 | 1.2 | 603 | 16 | 0.86 | 0.3 |
| qkv_sliding | 2560x1536 | 1 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 1 -->
| qkv_sliding | 2560x1536 | 8 | bf16 | ok | 5.7 | 11.1 | 1386 | 37 | 1.00 | 0.5 |
| qkv_sliding | 2560x1536 | 8 | fp16 | ok | 5.6 | 11.2 | 1396 | 37 | 1.01 | 0.2 |
| qkv_sliding | 2560x1536 | 8 | fp8 | ok | 6.5 | 9.6 | 601 | 16 | 0.87 | 0.3 |
| qkv_sliding | 2560x1536 | 8 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 8 -->
| qkv_sliding | 2560x1536 | 64 | bf16 | ok | 7.2 | 69.6 | 1088 | 29 | 1.00 | 0.0 |
| qkv_sliding | 2560x1536 | 64 | fp16 | ok | 7.2 | 69.9 | 1091 | 29 | 1.00 | 0.2 |
| qkv_sliding | 2560x1536 | 64 | fp8 | ok | 5.7 | 88.8 | 694 | 18 | 1.28 | 0.0 |
| qkv_sliding | 2560x1536 | 64 | int8 | ok | 58.0 | 8.7 | 68 | 2 | 0.12 | 0.0 |
| qkv_full | 5120x1536 | 1 | bf16 | ok | 9.7 | 1.6 | 1624 | 43 | 1.00 | 0.1 |
| qkv_full | 5120x1536 | 1 | fp16 | ok | 9.7 | 1.6 | 1628 | 43 | 1.00 | 0.1 |
| qkv_full | 5120x1536 | 1 | fp8 | ok | 5.8 | 2.7 | 1365 | 36 | 1.68 | 0.3 |
| qkv_full | 5120x1536 | 1 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 1 -->
| qkv_full | 5120x1536 | 8 | bf16 | ok | 9.8 | 12.9 | 1609 | 43 | 1.00 | 0.7 |
| qkv_full | 5120x1536 | 8 | fp16 | ok | 9.7 | 12.9 | 1616 | 43 | 1.00 | 0.0 |
| qkv_full | 5120x1536 | 8 | fp8 | ok | 5.8 | 21.7 | 1357 | 36 | 1.69 | 0.3 |
| qkv_full | 5120x1536 | 8 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 8 -->
| qkv_full | 5120x1536 | 64 | bf16 | ok | 9.7 | 104.0 | 1625 | 43 | 1.00 | 1.9 |
| qkv_full | 5120x1536 | 64 | fp16 | ok | 9.9 | 101.3 | 1583 | 42 | 0.97 | 0.4 |
| qkv_full | 5120x1536 | 64 | fp8 | ok | 6.9 | 145.5 | 1136 | 30 | 1.40 | 0.0 |
| qkv_full | 5120x1536 | 64 | int8 | ok | 58.2 | 17.3 | 135 | 4 | 0.17 | 0.0 |
| o_sliding | 1536x2048 | 1 | bf16 | ok | 6.1 | 1.0 | 1039 | 28 | 1.00 | 0.4 |
| o_sliding | 1536x2048 | 1 | fp16 | ok | 6.1 | 1.0 | 1036 | 28 | 1.00 | 0.1 |
| o_sliding | 1536x2048 | 1 | fp8 | ok | 5.0 | 1.3 | 632 | 17 | 1.22 | 0.2 |
| o_sliding | 1536x2048 | 1 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 1 -->
| o_sliding | 1536x2048 | 8 | bf16 | ok | 6.2 | 8.2 | 1021 | 27 | 1.00 | 0.2 |
| o_sliding | 1536x2048 | 8 | fp16 | ok | 6.1 | 8.2 | 1024 | 27 | 1.00 | 0.2 |
| o_sliding | 1536x2048 | 8 | fp8 | ok | 5.0 | 10.0 | 623 | 17 | 1.22 | 0.3 |
| o_sliding | 1536x2048 | 8 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 8 -->
| o_sliding | 1536x2048 | 64 | bf16 | ok | 7.2 | 55.6 | 869 | 23 | 1.00 | 0.1 |
| o_sliding | 1536x2048 | 64 | fp16 | ok | 7.2 | 55.6 | 869 | 23 | 1.00 | 0.2 |
| o_sliding | 1536x2048 | 64 | fp8 | ok | 5.6 | 72.4 | 565 | 15 | 1.30 | 0.1 |
| o_sliding | 1536x2048 | 64 | int8 | ok | 42.5 | 9.5 | 74 | 2 | 0.17 | 0.1 |
| o_full | 1536x4096 | 1 | bf16 | ok | 8.0 | 1.6 | 1578 | 42 | 1.00 | 0.1 |
| o_full | 1536x4096 | 1 | fp16 | ok | 8.0 | 1.6 | 1575 | 42 | 1.00 | 0.3 |
| o_full | 1536x4096 | 1 | fp8 | ok | 6.3 | 2.0 | 999 | 27 | 1.27 | 0.2 |
| o_full | 1536x4096 | 1 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 1 -->
| o_full | 1536x4096 | 8 | bf16 | ok | 8.1 | 12.4 | 1553 | 41 | 1.00 | 0.2 |
| o_full | 1536x4096 | 8 | fp16 | ok | 8.1 | 12.4 | 1551 | 41 | 1.00 | 0.1 |
| o_full | 1536x4096 | 8 | fp8 | ok | 6.4 | 15.8 | 987 | 26 | 1.27 | 0.2 |
| o_full | 1536x4096 | 8 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 8 -->
| o_full | 1536x4096 | 64 | bf16 | ok | 10.1 | 79.7 | 1246 | 33 | 1.00 | 0.2 |
| o_full | 1536x4096 | 64 | fp16 | ok | 10.3 | 77.9 | 1218 | 32 | 0.98 | 0.7 |
| o_full | 1536x4096 | 64 | fp8 | ok | 7.1 | 112.9 | 882 | 23 | 1.42 | 0.1 |
| o_full | 1536x4096 | 64 | int8 | ok | 76.6 | 10.5 | 82 | 2 | 0.13 | 0.0 |
| gate_up | 12288x1536 | 1 | bf16 | ok | 15.0 | 2.5 | 2513 | 67 | 1.00 | 0.1 |
| gate_up | 12288x1536 | 1 | fp16 | ok | 15.1 | 2.5 | 2500 | 66 | 0.99 | 0.2 |
| gate_up | 12288x1536 | 1 | fp8 | ok | 8.6 | 4.4 | 2198 | 58 | 1.75 | 0.2 |
| gate_up | 12288x1536 | 1 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 1 -->
| gate_up | 12288x1536 | 8 | bf16 | ok | 15.3 | 19.7 | 2465 | 66 | 1.00 | 0.1 |
| gate_up | 12288x1536 | 8 | fp16 | ok | 15.5 | 19.4 | 2431 | 65 | 0.99 | 0.1 |
| gate_up | 12288x1536 | 8 | fp8 | ok | 8.7 | 34.6 | 2165 | 58 | 1.76 | 0.4 |
| gate_up | 12288x1536 | 8 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 8 -->
| gate_up | 12288x1536 | 64 | bf16 | ok | 17.4 | 138.7 | 2167 | 58 | 1.00 | 0.1 |
| gate_up | 12288x1536 | 64 | fp16 | ok | 17.7 | 136.5 | 2132 | 57 | 0.98 | 0.4 |
| gate_up | 12288x1536 | 64 | fp8 | ok | 12.0 | 202.0 | 1578 | 42 | 1.46 | 0.0 |
| gate_up | 12288x1536 | 64 | int8 | ok | 58.5 | 41.3 | 323 | 9 | 0.30 | 0.0 |
| gate_up_wide | 24576x1536 | 1 | bf16 | ok | 26.0 | 2.9 | 2901 | 77 | 1.00 | 0.1 |
| gate_up_wide | 24576x1536 | 1 | fp16 | ok | 26.1 | 2.9 | 2889 | 77 | 1.00 | 0.1 |
| gate_up_wide | 24576x1536 | 1 | fp8 | ok | 14.5 | 5.2 | 2595 | 69 | 1.79 | 0.2 |
| gate_up_wide | 24576x1536 | 1 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 1 -->
| gate_up_wide | 24576x1536 | 8 | bf16 | ok | 26.6 | 22.7 | 2844 | 76 | 1.00 | 0.2 |
| gate_up_wide | 24576x1536 | 8 | fp16 | ok | 26.7 | 22.6 | 2824 | 75 | 0.99 | 0.3 |
| gate_up_wide | 24576x1536 | 8 | fp8 | ok | 14.9 | 40.7 | 2541 | 68 | 1.79 | 0.2 |
| gate_up_wide | 24576x1536 | 8 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 8 -->
| gate_up_wide | 24576x1536 | 64 | bf16 | ok | 31.4 | 153.6 | 2401 | 64 | 1.00 | 3.7 |
| gate_up_wide | 24576x1536 | 64 | fp16 | ok | 31.6 | 152.9 | 2389 | 64 | 1.00 | 0.4 |
| gate_up_wide | 24576x1536 | 64 | fp8 | ok | 18.7 | 258.0 | 2016 | 54 | 1.68 | 0.8 |
| gate_up_wide | 24576x1536 | 64 | int8 | ok | 59.5 | 81.2 | 634 | 17 | 0.53 | 0.1 |
| down | 1536x6144 | 1 | bf16 | ok | 11.4 | 1.7 | 1655 | 44 | 1.00 | 0.1 |
| down | 1536x6144 | 1 | fp16 | ok | 11.4 | 1.7 | 1657 | 44 | 1.00 | 0.1 |
| down | 1536x6144 | 1 | fp8 | ok | 8.1 | 2.3 | 1164 | 31 | 1.41 | 0.2 |
| down | 1536x6144 | 1 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 1 -->
| down | 1536x6144 | 8 | bf16 | ok | 11.5 | 13.1 | 1644 | 44 | 1.00 | 0.1 |
| down | 1536x6144 | 8 | fp16 | ok | 11.4 | 13.2 | 1649 | 44 | 1.00 | 0.1 |
| down | 1536x6144 | 8 | fp8 | ok | 8.2 | 18.5 | 1157 | 31 | 1.41 | 0.1 |
| down | 1536x6144 | 8 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 8 -->
| down | 1536x6144 | 64 | bf16 | ok | 14.0 | 86.6 | 1352 | 36 | 1.00 | 0.1 |
| down | 1536x6144 | 64 | fp16 | ok | 14.1 | 85.9 | 1342 | 36 | 0.99 | 0.2 |
| down | 1536x6144 | 64 | fp8 | ok | 8.6 | 141.1 | 1102 | 29 | 1.63 | 0.1 |
| down | 1536x6144 | 64 | int8 | ok | 110.7 | 10.9 | 85 | 2 | 0.13 | 0.0 |
| down_wide | 1536x12288 | 1 | bf16 | ok | 18.7 | 2.0 | 2022 | 54 | 1.00 | 0.0 |
| down_wide | 1536x12288 | 1 | fp16 | ok | 18.9 | 2.0 | 1995 | 53 | 0.99 | 0.1 |
| down_wide | 1536x12288 | 1 | fp8 | ok | 11.7 | 3.2 | 1614 | 43 | 1.60 | 0.1 |
| down_wide | 1536x12288 | 1 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 1 -->
| down_wide | 1536x12288 | 8 | bf16 | ok | 19.2 | 15.8 | 1969 | 52 | 1.00 | 0.1 |
| down_wide | 1536x12288 | 8 | fp16 | ok | 19.1 | 15.8 | 1972 | 52 | 1.00 | 0.2 |
| down_wide | 1536x12288 | 8 | fp8 | ok | 11.7 | 25.9 | 1618 | 43 | 1.64 | 0.1 |
| down_wide | 1536x12288 | 8 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 8 -->
| down_wide | 1536x12288 | 64 | bf16 | ok | 22.1 | 109.1 | 1705 | 45 | 1.00 | 0.2 |
| down_wide | 1536x12288 | 64 | fp16 | ok | 22.4 | 108.0 | 1688 | 45 | 0.99 | 0.1 |
| down_wide | 1536x12288 | 64 | fp8 | ok | 14.6 | 165.5 | 1293 | 34 | 1.52 | 0.4 |
| down_wide | 1536x12288 | 64 | int8 | ok | 213.3 | 11.3 | 88 | 2 | 0.10 | 0.0 |
| lm_head | 262144x1536 | 1 | bf16 | ok | 225.6 | 3.6 | 3570 | 95 | 1.00 | 0.1 |
| lm_head | 262144x1536 | 1 | fp16 | ok | 227.9 | 3.5 | 3533 | 94 | 0.99 | 0.1 |
| lm_head | 262144x1536 | 1 | fp8 | ok | 124.6 | 6.5 | 3232 | 86 | 1.81 | 0.3 |
| lm_head | 262144x1536 | 1 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 1 -->
| lm_head | 262144x1536 | 8 | bf16 | ok | 235.2 | 27.4 | 3424 | 91 | 1.00 | 0.3 |
| lm_head | 262144x1536 | 8 | fp16 | ok | 237.6 | 27.1 | 3390 | 90 | 0.99 | 0.2 |
| lm_head | 262144x1536 | 8 | fp8 | ok | 130.1 | 49.5 | 3094 | 82 | 1.81 | 0.4 |
| lm_head | 262144x1536 | 8 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 8 -->
| lm_head | 262144x1536 | 64 | bf16 | ok | 289.5 | 178.0 | 2781 | 74 | 1.00 | 0.8 |
| lm_head | 262144x1536 | 64 | fp16 | ok | 291.1 | 177.0 | 2766 | 74 | 0.99 | 0.7 |
| lm_head | 262144x1536 | 64 | fp8 | ok | 205.5 | 250.7 | 1959 | 52 | 1.41 | 1.7 |
| lm_head | 262144x1536 | 64 | int8 | ok | 402.4 | 128.1 | 1001 | 27 | 0.72 | 0.7 |

## Decode GEMM time per token (sum over all E2B GEMMs, weighted by count)

| M | dtype | us/token | x bf16 |
| ---: | --- | ---: | ---: |
| 1 | bf16 | 1965.1 | 1.00 |
| 1 | fp16 | 1975.3 | 0.99 |
| 1 | fp8 | 1306.4 | 1.50 |
| 1 | int8 | incomplete — a shape did not run |  |
| 8 | bf16 | 2008.2 | 1.00 |
| 8 | fp16 | 2014.3 | 1.00 |
| 8 | fp8 | 1323.5 | 1.52 |
| 8 | int8 | incomplete — a shape did not run |  |
| 64 | bf16 | 2375.7 | 1.00 |
| 64 | fp16 | 2393.5 | 0.99 |
| 64 | fp8 | 1592.7 | 1.49 |
| 64 | int8 | 12155.9 | 0.20 |

## Unsupported cells

- control_8192 M=1 int8: `RuntimeError: self.size(0) needs to be greater than 16, but got 1`
- control_8192 M=8 int8: `RuntimeError: self.size(0) needs to be greater than 16, but got 8`
- qkv_sliding M=1 int8: `RuntimeError: self.size(0) needs to be greater than 16, but got 1`
- qkv_sliding M=8 int8: `RuntimeError: self.size(0) needs to be greater than 16, but got 8`
- qkv_full M=1 int8: `RuntimeError: self.size(0) needs to be greater than 16, but got 1`
- qkv_full M=8 int8: `RuntimeError: self.size(0) needs to be greater than 16, but got 8`
- o_sliding M=1 int8: `RuntimeError: self.size(0) needs to be greater than 16, but got 1`
- o_sliding M=8 int8: `RuntimeError: self.size(0) needs to be greater than 16, but got 8`
- o_full M=1 int8: `RuntimeError: self.size(0) needs to be greater than 16, but got 1`
- o_full M=8 int8: `RuntimeError: self.size(0) needs to be greater than 16, but got 8`
- gate_up M=1 int8: `RuntimeError: self.size(0) needs to be greater than 16, but got 1`
- gate_up M=8 int8: `RuntimeError: self.size(0) needs to be greater than 16, but got 8`
- gate_up_wide M=1 int8: `RuntimeError: self.size(0) needs to be greater than 16, but got 1`
- gate_up_wide M=8 int8: `RuntimeError: self.size(0) needs to be greater than 16, but got 8`
- down M=1 int8: `RuntimeError: self.size(0) needs to be greater than 16, but got 1`
- down M=8 int8: `RuntimeError: self.size(0) needs to be greater than 16, but got 8`
- down_wide M=1 int8: `RuntimeError: self.size(0) needs to be greater than 16, but got 1`
- down_wide M=8 int8: `RuntimeError: self.size(0) needs to be greater than 16, but got 8`
- lm_head M=1 int8: `RuntimeError: self.size(0) needs to be greater than 16, but got 1`
- lm_head M=8 int8: `RuntimeError: self.size(0) needs to be greater than 16, but got 8`

