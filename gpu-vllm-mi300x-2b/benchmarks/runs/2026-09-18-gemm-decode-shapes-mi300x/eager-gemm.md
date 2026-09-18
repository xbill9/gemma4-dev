# GEMM at decode shapes — AMD Instinct MI300X VF (gfx942:sramecc+:xnack-), torch 2.12.0+rocm10.0.0

rotate_bytes=1073741824 (0 = hot weights), repeats=3, median of repeats.

| shape | N x K | M | dtype | status | us/call | TFLOP/s | weight GB/s | % of 3.76 TB/s | x bf16 | cv % |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control_8192 | 8192x8192 | 1 | bf16 | ok | 46.6 | 2.9 | 2879 | 77 | 1.00 | 0.4 |
| control_8192 | 8192x8192 | 1 | fp16 | ok | 46.8 | 2.9 | 2869 | 76 | 1.00 | 0.5 |
| control_8192 | 8192x8192 | 1 | fp8 | ok | 25.8 | 5.2 | 2597 | 69 | 1.80 | 1.0 |
| control_8192 | 8192x8192 | 1 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 1 -->
| control_8192 | 8192x8192 | 8 | bf16 | ok | 45.1 | 23.8 | 2975 | 79 | 1.00 | 1.3 |
| control_8192 | 8192x8192 | 8 | fp16 | ok | 38.8 | 27.7 | 3458 | 92 | 1.16 | 0.3 |
| control_8192 | 8192x8192 | 8 | fp8 | ok | 26.7 | 40.2 | 2511 | 67 | 1.69 | 0.8 |
| control_8192 | 8192x8192 | 8 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 8 -->
| control_8192 | 8192x8192 | 64 | bf16 | ok | 53.5 | 160.6 | 2509 | 67 | 1.00 | 0.1 |
| control_8192 | 8192x8192 | 64 | fp16 | ok | 53.6 | 160.2 | 2503 | 67 | 1.00 | 1.0 |
| control_8192 | 8192x8192 | 64 | fp8 | ok | 31.2 | 275.1 | 2149 | 57 | 1.71 | 0.6 |
| control_8192 | 8192x8192 | 64 | int8 | ok | 275.9 | 31.1 | 243 | 6 | 0.19 | 0.0 |
| control_8192 | 8192x8192 | 8192 | bf16 | ok | 1784.3 | 616.2 | 75 | 2 | 1.00 | 0.3 |
| control_8192 | 8192x8192 | 8192 | fp16 | ok | 1864.6 | 589.7 | 72 | 2 | 0.96 | 0.3 |
| control_8192 | 8192x8192 | 8192 | fp8 | ok | 1005.7 | 1093.3 | 67 | 2 | 1.77 | 0.1 |
| control_8192 | 8192x8192 | 8192 | int8 | ok | 4466.4 | 246.2 | 15 | 0 | 0.40 | 0.0 |
| qkv_sliding | 2560x1536 | 1 | bf16 | ok | 17.6 | 0.4 | 448 | 12 | 1.00 | 5.2 |
| qkv_sliding | 2560x1536 | 1 | fp16 | ok | 17.3 | 0.5 | 453 | 12 | 1.01 | 0.4 |
| qkv_sliding | 2560x1536 | 1 | fp8 | ok | 20.6 | 0.4 | 191 | 5 | 0.85 | 2.2 |
| qkv_sliding | 2560x1536 | 1 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 1 -->
| qkv_sliding | 2560x1536 | 8 | bf16 | ok | 17.9 | 3.5 | 438 | 12 | 1.00 | 1.9 |
| qkv_sliding | 2560x1536 | 8 | fp16 | ok | 17.5 | 3.6 | 450 | 12 | 1.03 | 6.7 |
| qkv_sliding | 2560x1536 | 8 | fp8 | ok | 20.6 | 3.1 | 191 | 5 | 0.87 | 1.1 |
| qkv_sliding | 2560x1536 | 8 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 8 -->
| qkv_sliding | 2560x1536 | 64 | bf16 | ok | 17.4 | 28.9 | 451 | 12 | 1.00 | 0.7 |
| qkv_sliding | 2560x1536 | 64 | fp16 | ok | 19.9 | 25.2 | 394 | 10 | 0.87 | 4.9 |
| qkv_sliding | 2560x1536 | 64 | fp8 | ok | 20.4 | 24.7 | 193 | 5 | 0.85 | 0.5 |
| qkv_sliding | 2560x1536 | 64 | int8 | ok | 58.0 | 8.7 | 68 | 2 | 0.30 | 0.0 |
| qkv_full | 5120x1536 | 1 | bf16 | ok | 17.2 | 0.9 | 914 | 24 | 1.00 | 0.3 |
| qkv_full | 5120x1536 | 1 | fp16 | ok | 17.2 | 0.9 | 916 | 24 | 1.00 | 0.7 |
| qkv_full | 5120x1536 | 1 | fp8 | ok | 20.4 | 0.8 | 385 | 10 | 0.84 | 1.0 |
| qkv_full | 5120x1536 | 1 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 1 -->
| qkv_full | 5120x1536 | 8 | bf16 | ok | 18.7 | 6.7 | 841 | 22 | 1.00 | 4.3 |
| qkv_full | 5120x1536 | 8 | fp16 | ok | 16.9 | 7.4 | 930 | 25 | 1.11 | 0.9 |
| qkv_full | 5120x1536 | 8 | fp8 | ok | 20.3 | 6.2 | 388 | 10 | 0.92 | 0.4 |
| qkv_full | 5120x1536 | 8 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 8 -->
| qkv_full | 5120x1536 | 64 | bf16 | ok | 17.0 | 59.2 | 925 | 25 | 1.00 | 0.1 |
| qkv_full | 5120x1536 | 64 | fp16 | ok | 17.1 | 58.7 | 917 | 24 | 0.99 | 0.7 |
| qkv_full | 5120x1536 | 64 | fp8 | ok | 20.2 | 49.8 | 389 | 10 | 0.84 | 0.5 |
| qkv_full | 5120x1536 | 64 | int8 | ok | 58.2 | 17.3 | 135 | 4 | 0.29 | 0.0 |
| o_sliding | 1536x2048 | 1 | bf16 | ok | 17.2 | 0.4 | 366 | 10 | 1.00 | 1.2 |
| o_sliding | 1536x2048 | 1 | fp16 | ok | 17.3 | 0.4 | 363 | 10 | 0.99 | 5.2 |
| o_sliding | 1536x2048 | 1 | fp8 | ok | 19.8 | 0.3 | 159 | 4 | 0.87 | 0.2 |
| o_sliding | 1536x2048 | 1 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 1 -->
| o_sliding | 1536x2048 | 8 | bf16 | ok | 17.0 | 3.0 | 370 | 10 | 1.00 | 1.0 |
| o_sliding | 1536x2048 | 8 | fp16 | ok | 17.2 | 2.9 | 365 | 10 | 0.99 | 0.4 |
| o_sliding | 1536x2048 | 8 | fp8 | ok | 20.8 | 2.4 | 151 | 4 | 0.82 | 3.8 |
| o_sliding | 1536x2048 | 8 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 8 -->
| o_sliding | 1536x2048 | 64 | bf16 | ok | 17.1 | 23.6 | 368 | 10 | 1.00 | 0.9 |
| o_sliding | 1536x2048 | 64 | fp16 | ok | 17.1 | 23.5 | 367 | 10 | 1.00 | 0.1 |
| o_sliding | 1536x2048 | 64 | fp8 | ok | 20.4 | 19.7 | 154 | 4 | 0.84 | 3.9 |
| o_sliding | 1536x2048 | 64 | int8 | ok | 42.6 | 9.5 | 74 | 2 | 0.40 | 0.0 |
| o_full | 1536x4096 | 1 | bf16 | ok | 17.3 | 0.7 | 726 | 19 | 1.00 | 0.3 |
| o_full | 1536x4096 | 1 | fp16 | ok | 18.3 | 0.7 | 687 | 18 | 0.95 | 3.5 |
| o_full | 1536x4096 | 1 | fp8 | ok | 20.2 | 0.6 | 312 | 8 | 0.86 | 0.2 |
| o_full | 1536x4096 | 1 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 1 -->
| o_full | 1536x4096 | 8 | bf16 | ok | 17.1 | 5.9 | 737 | 20 | 1.00 | 0.8 |
| o_full | 1536x4096 | 8 | fp16 | ok | 17.0 | 5.9 | 740 | 20 | 1.00 | 0.7 |
| o_full | 1536x4096 | 8 | fp8 | ok | 20.3 | 5.0 | 310 | 8 | 0.84 | 5.5 |
| o_full | 1536x4096 | 8 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 8 -->
| o_full | 1536x4096 | 64 | bf16 | ok | 16.9 | 47.5 | 743 | 20 | 1.00 | 2.0 |
| o_full | 1536x4096 | 64 | fp16 | ok | 17.0 | 47.5 | 742 | 20 | 1.00 | 0.7 |
| o_full | 1536x4096 | 64 | fp8 | ok | 20.1 | 40.1 | 313 | 8 | 0.84 | 3.6 |
| o_full | 1536x4096 | 64 | int8 | ok | 76.6 | 10.5 | 82 | 2 | 0.22 | 0.0 |
| gate_up | 12288x1536 | 1 | bf16 | ok | 17.2 | 2.2 | 2194 | 58 | 1.00 | 0.4 |
| gate_up | 12288x1536 | 1 | fp16 | ok | 17.2 | 2.2 | 2197 | 58 | 1.00 | 0.3 |
| gate_up | 12288x1536 | 1 | fp8 | ok | 19.7 | 1.9 | 957 | 25 | 0.87 | 0.6 |
| gate_up | 12288x1536 | 1 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 1 -->
| gate_up | 12288x1536 | 8 | bf16 | ok | 17.7 | 17.1 | 2135 | 57 | 1.00 | 1.3 |
| gate_up | 12288x1536 | 8 | fp16 | ok | 17.0 | 17.8 | 2220 | 59 | 1.04 | 3.7 |
| gate_up | 12288x1536 | 8 | fp8 | ok | 20.2 | 15.0 | 935 | 25 | 0.88 | 4.6 |
| gate_up | 12288x1536 | 8 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 8 -->
| gate_up | 12288x1536 | 64 | bf16 | ok | 18.0 | 134.3 | 2099 | 56 | 1.00 | 3.6 |
| gate_up | 12288x1536 | 64 | fp16 | ok | 17.8 | 135.8 | 2123 | 56 | 1.01 | 2.4 |
| gate_up | 12288x1536 | 64 | fp8 | ok | 20.2 | 119.8 | 936 | 25 | 0.89 | 5.1 |
| gate_up | 12288x1536 | 64 | int8 | ok | 58.5 | 41.3 | 323 | 9 | 0.31 | 0.0 |
| gate_up_wide | 24576x1536 | 1 | bf16 | ok | 26.4 | 2.9 | 2864 | 76 | 1.00 | 3.4 |
| gate_up_wide | 24576x1536 | 1 | fp16 | ok | 26.3 | 2.9 | 2874 | 76 | 1.00 | 4.3 |
| gate_up_wide | 24576x1536 | 1 | fp8 | ok | 20.0 | 3.8 | 1883 | 50 | 1.31 | 0.3 |
| gate_up_wide | 24576x1536 | 1 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 1 -->
| gate_up_wide | 24576x1536 | 8 | bf16 | ok | 26.9 | 22.4 | 2806 | 75 | 1.00 | 8.6 |
| gate_up_wide | 24576x1536 | 8 | fp16 | ok | 26.9 | 22.5 | 2810 | 75 | 1.00 | 4.0 |
| gate_up_wide | 24576x1536 | 8 | fp8 | ok | 19.9 | 30.4 | 1899 | 51 | 1.35 | 0.1 |
| gate_up_wide | 24576x1536 | 8 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 8 -->
| gate_up_wide | 24576x1536 | 64 | bf16 | ok | 31.7 | 152.6 | 2385 | 63 | 1.00 | 2.2 |
| gate_up_wide | 24576x1536 | 64 | fp16 | ok | 31.6 | 152.9 | 2389 | 64 | 1.00 | 1.9 |
| gate_up_wide | 24576x1536 | 64 | fp8 | ok | 19.9 | 243.3 | 1901 | 51 | 1.59 | 2.3 |
| gate_up_wide | 24576x1536 | 64 | int8 | ok | 59.5 | 81.1 | 634 | 17 | 0.53 | 0.0 |
| down | 1536x6144 | 1 | bf16 | ok | 17.2 | 1.1 | 1099 | 29 | 1.00 | 0.3 |
| down | 1536x6144 | 1 | fp16 | ok | 17.7 | 1.1 | 1064 | 28 | 0.97 | 3.8 |
| down | 1536x6144 | 1 | fp8 | ok | 20.1 | 0.9 | 470 | 13 | 0.86 | 0.5 |
| down | 1536x6144 | 1 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 1 -->
| down | 1536x6144 | 8 | bf16 | ok | 17.0 | 8.9 | 1107 | 29 | 1.00 | 1.0 |
| down | 1536x6144 | 8 | fp16 | ok | 17.1 | 8.8 | 1105 | 29 | 1.00 | 0.4 |
| down | 1536x6144 | 8 | fp8 | ok | 21.1 | 7.1 | 446 | 12 | 0.81 | 2.3 |
| down | 1536x6144 | 8 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 8 -->
| down | 1536x6144 | 64 | bf16 | ok | 17.2 | 70.4 | 1100 | 29 | 1.00 | 0.7 |
| down | 1536x6144 | 64 | fp16 | ok | 17.1 | 70.6 | 1104 | 29 | 1.00 | 0.5 |
| down | 1536x6144 | 64 | fp8 | ok | 20.5 | 58.8 | 459 | 12 | 0.84 | 1.6 |
| down | 1536x6144 | 64 | int8 | ok | 110.8 | 10.9 | 85 | 2 | 0.15 | 0.0 |
| down_wide | 1536x12288 | 1 | bf16 | ok | 18.9 | 2.0 | 1994 | 53 | 1.00 | 1.6 |
| down_wide | 1536x12288 | 1 | fp16 | ok | 18.9 | 2.0 | 2002 | 53 | 1.00 | 1.3 |
| down_wide | 1536x12288 | 1 | fp8 | ok | 20.8 | 1.8 | 906 | 24 | 0.91 | 5.0 |
| down_wide | 1536x12288 | 1 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 1 -->
| down_wide | 1536x12288 | 8 | bf16 | ok | 19.4 | 15.6 | 1947 | 52 | 1.00 | 1.6 |
| down_wide | 1536x12288 | 8 | fp16 | ok | 19.2 | 15.7 | 1964 | 52 | 1.01 | 1.4 |
| down_wide | 1536x12288 | 8 | fp8 | ok | 20.3 | 14.9 | 929 | 25 | 0.95 | 0.6 |
| down_wide | 1536x12288 | 8 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 8 -->
| down_wide | 1536x12288 | 64 | bf16 | ok | 22.3 | 108.1 | 1689 | 45 | 1.00 | 2.9 |
| down_wide | 1536x12288 | 64 | fp16 | ok | 22.1 | 109.3 | 1708 | 45 | 1.01 | 1.1 |
| down_wide | 1536x12288 | 64 | fp8 | ok | 20.1 | 120.4 | 941 | 25 | 1.11 | 0.2 |
| down_wide | 1536x12288 | 64 | int8 | ok | 213.4 | 11.3 | 88 | 2 | 0.10 | 0.0 |
| lm_head | 262144x1536 | 1 | bf16 | ok | 225.5 | 3.6 | 3572 | 95 | 1.00 | 0.1 |
| lm_head | 262144x1536 | 1 | fp16 | ok | 227.6 | 3.5 | 3539 | 94 | 0.99 | 0.1 |
| lm_head | 262144x1536 | 1 | fp8 | ok | 123.9 | 6.5 | 3249 | 86 | 1.82 | 0.2 |
| lm_head | 262144x1536 | 1 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 1 -->
| lm_head | 262144x1536 | 8 | bf16 | ok | 234.5 | 27.5 | 3433 | 91 | 1.00 | 0.2 |
| lm_head | 262144x1536 | 8 | fp16 | ok | 237.3 | 27.1 | 3393 | 90 | 0.99 | 0.3 |
| lm_head | 262144x1536 | 8 | fp8 | ok | 129.4 | 49.8 | 3112 | 83 | 1.81 | 0.2 |
| lm_head | 262144x1536 | 8 | int8 | unsupported | | | | | | | <!-- RuntimeError: self.size(0) needs to be greater than 16, but got 8 -->
| lm_head | 262144x1536 | 64 | bf16 | ok | 282.7 | 182.3 | 2849 | 76 | 1.00 | 0.9 |
| lm_head | 262144x1536 | 64 | fp16 | ok | 289.5 | 178.0 | 2782 | 74 | 0.98 | 0.3 |
| lm_head | 262144x1536 | 64 | fp8 | ok | 196.0 | 262.9 | 2054 | 55 | 1.44 | 0.6 |
| lm_head | 262144x1536 | 64 | int8 | ok | 390.1 | 132.1 | 1032 | 27 | 0.72 | 0.5 |

## Decode GEMM time per token (sum over all E2B GEMMs, weighted by count)

| M | dtype | us/token | x bf16 |
| ---: | --- | ---: | ---: |
| 1 | bf16 | 2862.7 | 1.00 |
| 1 | fp16 | 2873.5 | 1.00 |
| 1 | fp8 | 2955.1 | 0.97 |
| 1 | int8 | incomplete — a shape did not run |  |
| 8 | bf16 | 2910.2 | 1.00 |
| 8 | fp16 | 2878.7 | 1.01 |
| 8 | fp8 | 2997.2 | 0.97 |
| 8 | int8 | incomplete — a shape did not run |  |
| 64 | bf16 | 3093.5 | 1.00 |
| 64 | fp16 | 3163.8 | 0.98 |
| 64 | fp8 | 3031.6 | 1.02 |
| 64 | int8 | 12147.5 | 0.25 |

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

