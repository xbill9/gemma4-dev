# GEMM at decode shapes — AMD Instinct MI300X VF (gfx942:sramecc+:xnack-), torch 2.12.0+rocm10.0.0

mode=graph, rotate_bytes=1073741824 (0 = hot weights), repeats=3, median of repeats.

| shape | N x K | M | dtype | status | us/call | TFLOP/s | weight GB/s | % of 3.76 TB/s | x bf16 | cv % |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control_8192 | 8192x8192 | 1 | bf16 | ok | 45.9 | 2.9 | 2927 | 78 | 1.00 | 0.2 |
| control_8192 | 8192x8192 | 1 | fp16 | ok | 46.3 | 2.9 | 2900 | 77 | 0.99 | 0.1 |
| control_8192 | 8192x8192 | 1 | fp8 | ok | 25.8 | 5.2 | 2604 | 69 | 1.78 | 0.1 |
| control_8192 | 8192x8192 | 1 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 1 -->
| control_8192 | 8192x8192 | 8 | bf16 | ok | 44.1 | 24.3 | 3043 | 81 | 1.00 | 0.1 |
| control_8192 | 8192x8192 | 8 | fp16 | ok | 38.8 | 27.7 | 3464 | 92 | 1.14 | 0.3 |
| control_8192 | 8192x8192 | 8 | fp8 | ok | 26.4 | 40.7 | 2542 | 68 | 1.67 | 0.2 |
| control_8192 | 8192x8192 | 8 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 8 -->
| control_8192 | 8192x8192 | 64 | bf16 | ok | 53.1 | 161.8 | 2528 | 67 | 1.00 | 0.1 |
| control_8192 | 8192x8192 | 64 | fp16 | ok | 54.0 | 159.0 | 2485 | 66 | 0.98 | 0.2 |
| control_8192 | 8192x8192 | 64 | fp8 | ok | 30.9 | 278.0 | 2172 | 58 | 1.72 | 0.0 |
| control_8192 | 8192x8192 | 64 | int8 | ok | 275.6 | 31.2 | 244 | 6 | 0.19 | 0.0 |
| control_8192 | 8192x8192 | 8192 | bf16 | ok | 1791.4 | 613.8 | 75 | 2 | 1.00 | 0.5 |
| control_8192 | 8192x8192 | 8192 | fp16 | ok | 1871.2 | 587.6 | 72 | 2 | 0.96 | 0.2 |
| control_8192 | 8192x8192 | 8192 | fp8 | ok | 995.6 | 1104.4 | 67 | 2 | 1.80 | 0.3 |
| control_8192 | 8192x8192 | 8192 | int8 | ok | 4378.5 | 251.1 | 15 | 0 | 0.41 | 0.1 |
| qkv_sliding | 2560x1536 | 1 | bf16 | ok | 5.6 | 1.4 | 1404 | 37 | 1.00 | 0.3 |
| qkv_sliding | 2560x1536 | 1 | fp16 | ok | 5.6 | 1.4 | 1417 | 38 | 1.01 | 0.1 |
| qkv_sliding | 2560x1536 | 1 | fp8 | ok | 6.5 | 1.2 | 607 | 16 | 0.87 | 0.4 |
| qkv_sliding | 2560x1536 | 1 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 1 -->
| qkv_sliding | 2560x1536 | 8 | bf16 | ok | 5.7 | 11.1 | 1387 | 37 | 1.00 | 0.2 |
| qkv_sliding | 2560x1536 | 8 | fp16 | ok | 5.6 | 11.2 | 1399 | 37 | 1.01 | 0.1 |
| qkv_sliding | 2560x1536 | 8 | fp8 | ok | 6.5 | 9.6 | 602 | 16 | 0.87 | 0.4 |
| qkv_sliding | 2560x1536 | 8 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 8 -->
| qkv_sliding | 2560x1536 | 64 | bf16 | ok | 7.2 | 70.0 | 1093 | 29 | 1.00 | 0.1 |
| qkv_sliding | 2560x1536 | 64 | fp16 | ok | 7.2 | 69.8 | 1090 | 29 | 1.00 | 0.1 |
| qkv_sliding | 2560x1536 | 64 | fp8 | ok | 5.7 | 88.9 | 695 | 18 | 1.27 | 0.0 |
| qkv_sliding | 2560x1536 | 64 | int8 | ok | 58.0 | 8.7 | 68 | 2 | 0.12 | 0.0 |
| qkv_full | 5120x1536 | 1 | bf16 | ok | 9.7 | 1.6 | 1622 | 43 | 1.00 | 0.0 |
| qkv_full | 5120x1536 | 1 | fp16 | ok | 9.7 | 1.6 | 1616 | 43 | 1.00 | 0.7 |
| qkv_full | 5120x1536 | 1 | fp8 | ok | 5.7 | 2.7 | 1369 | 36 | 1.69 | 0.1 |
| qkv_full | 5120x1536 | 1 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 1 -->
| qkv_full | 5120x1536 | 8 | bf16 | ok | 9.8 | 12.8 | 1606 | 43 | 1.00 | 0.0 |
| qkv_full | 5120x1536 | 8 | fp16 | ok | 9.8 | 12.9 | 1608 | 43 | 1.00 | 0.0 |
| qkv_full | 5120x1536 | 8 | fp8 | ok | 5.8 | 21.6 | 1352 | 36 | 1.68 | 0.2 |
| qkv_full | 5120x1536 | 8 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 8 -->
| qkv_full | 5120x1536 | 64 | bf16 | ok | 10.4 | 97.2 | 1519 | 40 | 1.00 | 0.1 |
| qkv_full | 5120x1536 | 64 | fp16 | ok | 10.4 | 96.3 | 1505 | 40 | 0.99 | 0.1 |
| qkv_full | 5120x1536 | 64 | fp8 | ok | 7.1 | 140.9 | 1101 | 29 | 1.45 | 0.2 |
| qkv_full | 5120x1536 | 64 | int8 | ok | 58.1 | 17.3 | 135 | 4 | 0.18 | 0.0 |
| o_sliding | 1536x2048 | 1 | bf16 | ok | 6.1 | 1.0 | 1037 | 28 | 1.00 | 0.2 |
| o_sliding | 1536x2048 | 1 | fp16 | ok | 6.1 | 1.0 | 1033 | 27 | 1.00 | 0.3 |
| o_sliding | 1536x2048 | 1 | fp8 | ok | 4.9 | 1.3 | 637 | 17 | 1.23 | 0.1 |
| o_sliding | 1536x2048 | 1 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 1 -->
| o_sliding | 1536x2048 | 8 | bf16 | ok | 6.1 | 8.2 | 1023 | 27 | 1.00 | 0.1 |
| o_sliding | 1536x2048 | 8 | fp16 | ok | 6.2 | 8.2 | 1023 | 27 | 1.00 | 0.1 |
| o_sliding | 1536x2048 | 8 | fp8 | ok | 5.0 | 10.0 | 624 | 17 | 1.22 | 0.3 |
| o_sliding | 1536x2048 | 8 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 8 -->
| o_sliding | 1536x2048 | 64 | bf16 | ok | 7.2 | 55.7 | 870 | 23 | 1.00 | 0.1 |
| o_sliding | 1536x2048 | 64 | fp16 | ok | 7.3 | 55.1 | 861 | 23 | 0.99 | 0.1 |
| o_sliding | 1536x2048 | 64 | fp8 | ok | 5.6 | 72.4 | 566 | 15 | 1.30 | 0.0 |
| o_sliding | 1536x2048 | 64 | int8 | ok | 42.6 | 9.5 | 74 | 2 | 0.17 | 0.0 |
| o_full | 1536x4096 | 1 | bf16 | ok | 8.0 | 1.6 | 1582 | 42 | 1.00 | 0.3 |
| o_full | 1536x4096 | 1 | fp16 | ok | 8.0 | 1.6 | 1575 | 42 | 1.00 | 0.2 |
| o_full | 1536x4096 | 1 | fp8 | ok | 6.3 | 2.0 | 998 | 27 | 1.26 | 0.1 |
| o_full | 1536x4096 | 1 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 1 -->
| o_full | 1536x4096 | 8 | bf16 | ok | 8.1 | 12.5 | 1556 | 41 | 1.00 | 0.2 |
| o_full | 1536x4096 | 8 | fp16 | ok | 8.1 | 12.5 | 1562 | 42 | 1.00 | 0.0 |
| o_full | 1536x4096 | 8 | fp8 | ok | 6.3 | 15.9 | 992 | 26 | 1.28 | 0.2 |
| o_full | 1536x4096 | 8 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 8 -->
| o_full | 1536x4096 | 64 | bf16 | ok | 10.6 | 76.2 | 1190 | 32 | 1.00 | 0.3 |
| o_full | 1536x4096 | 64 | fp16 | ok | 10.7 | 75.5 | 1180 | 31 | 0.99 | 0.2 |
| o_full | 1536x4096 | 64 | fp8 | ok | 7.2 | 111.7 | 872 | 23 | 1.47 | 0.1 |
| o_full | 1536x4096 | 64 | int8 | ok | 76.6 | 10.5 | 82 | 2 | 0.14 | 0.0 |
| gate_up | 12288x1536 | 1 | bf16 | ok | 15.0 | 2.5 | 2514 | 67 | 1.00 | 0.1 |
| gate_up | 12288x1536 | 1 | fp16 | ok | 15.1 | 2.5 | 2495 | 66 | 0.99 | 0.1 |
| gate_up | 12288x1536 | 1 | fp8 | ok | 8.6 | 4.4 | 2182 | 58 | 1.74 | 0.4 |
| gate_up | 12288x1536 | 1 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 1 -->
| gate_up | 12288x1536 | 8 | bf16 | ok | 15.5 | 19.5 | 2435 | 65 | 1.00 | 0.2 |
| gate_up | 12288x1536 | 8 | fp16 | ok | 15.6 | 19.4 | 2421 | 64 | 0.99 | 0.1 |
| gate_up | 12288x1536 | 8 | fp8 | ok | 9.0 | 33.4 | 2088 | 56 | 1.72 | 0.4 |
| gate_up | 12288x1536 | 8 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 8 -->
| gate_up | 12288x1536 | 64 | bf16 | ok | 17.4 | 139.0 | 2172 | 58 | 1.00 | 0.1 |
| gate_up | 12288x1536 | 64 | fp16 | ok | 17.6 | 137.1 | 2143 | 57 | 0.99 | 0.1 |
| gate_up | 12288x1536 | 64 | fp8 | ok | 11.9 | 203.2 | 1588 | 42 | 1.46 | 0.3 |
| gate_up | 12288x1536 | 64 | int8 | ok | 58.5 | 41.3 | 323 | 9 | 0.30 | 0.0 |
| gate_up_wide | 24576x1536 | 1 | bf16 | ok | 25.8 | 2.9 | 2929 | 78 | 1.00 | 0.0 |
| gate_up_wide | 24576x1536 | 1 | fp16 | ok | 25.8 | 2.9 | 2921 | 78 | 1.00 | 0.0 |
| gate_up_wide | 24576x1536 | 1 | fp8 | ok | 14.5 | 5.2 | 2596 | 69 | 1.77 | 0.1 |
| gate_up_wide | 24576x1536 | 1 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 1 -->
| gate_up_wide | 24576x1536 | 8 | bf16 | ok | 26.4 | 22.9 | 2859 | 76 | 1.00 | 0.1 |
| gate_up_wide | 24576x1536 | 8 | fp16 | ok | 26.4 | 22.9 | 2860 | 76 | 1.00 | 0.3 |
| gate_up_wide | 24576x1536 | 8 | fp8 | ok | 14.9 | 40.7 | 2542 | 68 | 1.78 | 0.1 |
| gate_up_wide | 24576x1536 | 8 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 8 -->
| gate_up_wide | 24576x1536 | 64 | bf16 | ok | 31.6 | 152.9 | 2388 | 64 | 1.00 | 2.1 |
| gate_up_wide | 24576x1536 | 64 | fp16 | ok | 31.9 | 151.3 | 2365 | 63 | 0.99 | 0.6 |
| gate_up_wide | 24576x1536 | 64 | fp8 | ok | 18.7 | 258.2 | 2017 | 54 | 1.69 | 1.1 |
| gate_up_wide | 24576x1536 | 64 | int8 | ok | 59.1 | 81.8 | 639 | 17 | 0.53 | 0.1 |
| down | 1536x6144 | 1 | bf16 | ok | 11.5 | 1.6 | 1637 | 44 | 1.00 | 0.1 |
| down | 1536x6144 | 1 | fp16 | ok | 11.4 | 1.7 | 1656 | 44 | 1.01 | 0.2 |
| down | 1536x6144 | 1 | fp8 | ok | 8.1 | 2.3 | 1164 | 31 | 1.42 | 0.4 |
| down | 1536x6144 | 1 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 1 -->
| down | 1536x6144 | 8 | bf16 | ok | 11.6 | 13.0 | 1627 | 43 | 1.00 | 0.1 |
| down | 1536x6144 | 8 | fp16 | ok | 11.6 | 13.0 | 1631 | 43 | 1.00 | 0.0 |
| down | 1536x6144 | 8 | fp8 | ok | 8.2 | 18.4 | 1151 | 31 | 1.41 | 0.2 |
| down | 1536x6144 | 8 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 8 -->
| down | 1536x6144 | 64 | bf16 | ok | 14.0 | 86.2 | 1347 | 36 | 1.00 | 0.1 |
| down | 1536x6144 | 64 | fp16 | ok | 14.1 | 85.7 | 1338 | 36 | 0.99 | 0.1 |
| down | 1536x6144 | 64 | fp8 | ok | 9.0 | 133.9 | 1046 | 28 | 1.55 | 0.7 |
| down | 1536x6144 | 64 | int8 | ok | 110.7 | 10.9 | 85 | 2 | 0.13 | 0.1 |
| down_wide | 1536x12288 | 1 | bf16 | ok | 18.9 | 2.0 | 1993 | 53 | 1.00 | 0.1 |
| down_wide | 1536x12288 | 1 | fp16 | ok | 19.4 | 1.9 | 1942 | 52 | 0.97 | 0.0 |
| down_wide | 1536x12288 | 1 | fp8 | ok | 11.8 | 3.2 | 1599 | 43 | 1.60 | 0.1 |
| down_wide | 1536x12288 | 1 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 1 -->
| down_wide | 1536x12288 | 8 | bf16 | ok | 19.1 | 15.8 | 1977 | 53 | 1.00 | 0.1 |
| down_wide | 1536x12288 | 8 | fp16 | ok | 19.3 | 15.7 | 1959 | 52 | 0.99 | 0.0 |
| down_wide | 1536x12288 | 8 | fp8 | ok | 11.9 | 25.4 | 1587 | 42 | 1.61 | 0.0 |
| down_wide | 1536x12288 | 8 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 8 -->
| down_wide | 1536x12288 | 64 | bf16 | ok | 22.0 | 109.6 | 1712 | 46 | 1.00 | 0.1 |
| down_wide | 1536x12288 | 64 | fp16 | ok | 22.4 | 108.0 | 1688 | 45 | 0.99 | 0.0 |
| down_wide | 1536x12288 | 64 | fp8 | ok | 14.5 | 166.4 | 1300 | 35 | 1.52 | 0.2 |
| down_wide | 1536x12288 | 64 | int8 | ok | 213.4 | 11.3 | 88 | 2 | 0.10 | 0.0 |
| lm_head | 262144x1536 | 1 | bf16 | ok | 223.3 | 3.6 | 3606 | 96 | 1.00 | 0.2 |
| lm_head | 262144x1536 | 1 | fp16 | ok | 227.1 | 3.5 | 3546 | 94 | 0.98 | 0.1 |
| lm_head | 262144x1536 | 1 | fp8 | ok | 122.1 | 6.6 | 3299 | 88 | 1.83 | 0.2 |
| lm_head | 262144x1536 | 1 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 1 -->
| lm_head | 262144x1536 | 8 | bf16 | ok | 233.8 | 27.6 | 3444 | 92 | 1.00 | 0.3 |
| lm_head | 262144x1536 | 8 | fp16 | ok | 234.9 | 27.4 | 3429 | 91 | 1.00 | 0.2 |
| lm_head | 262144x1536 | 8 | fp8 | ok | 128.4 | 50.2 | 3136 | 83 | 1.82 | 0.2 |
| lm_head | 262144x1536 | 8 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 8 -->
| lm_head | 262144x1536 | 64 | bf16 | ok | 289.1 | 178.2 | 2785 | 74 | 1.00 | 1.0 |
| lm_head | 262144x1536 | 64 | fp16 | ok | 296.5 | 173.8 | 2716 | 72 | 0.98 | 0.8 |
| lm_head | 262144x1536 | 64 | fp8 | ok | 196.6 | 262.1 | 2048 | 54 | 1.47 | 1.0 |
| lm_head | 262144x1536 | 64 | int8 | ok | 400.0 | 128.8 | 1007 | 27 | 0.72 | 0.6 |

## Decode GEMM time per token (sum over all E2B GEMMs, weighted by count)

| M | dtype | us/token | x bf16 |
| ---: | --- | ---: | ---: |
| 1 | bf16 | 1966.2 | 1.00 |
| 1 | fp16 | 1980.8 | 0.99 |
| 1 | fp8 | 1304.2 | 1.51 |
| 1 | int8 | incomplete — a shape did not run |  |
| 8 | bf16 | 2006.6 | 1.00 |
| 8 | fp16 | 2010.2 | 1.00 |
| 8 | fp8 | 1330.8 | 1.51 |
| 8 | int8 | incomplete — a shape did not run |  |
| 64 | bf16 | 2383.6 | 1.00 |
| 64 | fp16 | 2412.6 | 0.99 |
| 64 | fp8 | 1589.4 | 1.50 |
| 64 | int8 | 12146.4 | 0.20 |

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

