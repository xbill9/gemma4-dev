
128 in / 512 out, best setting per level:

| Requests | vLLM | SGLang | llama.cpp | mistral.rs |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 🥇 453 (4 drafted) | 283 (4 drafted) | 271 (2 drafted) | 134 (none) |
| 2 | 🥇 856 (4 drafted) | 552 (4 drafted) | 430 (2 drafted) | 254 (none) |
| 4 | 🥇 1,483 (4 drafted) | 875 (4 drafted) | 596 (2 drafted) | 468 (none) |
| 8 | 🥇 2,399 (4 drafted) | 1,223 (4 drafted) | 791 (2 drafted) | 627 (none) |
| 16 | 🥇 2,823 (4 drafted) | 1,673 (none) | 1,021 (2 drafted) | 1,339 (none) |
| 32 | 🥇 3,665 (1 drafted) | 2,257 (none) | 1,132 (2 drafted) | 1,843 (none) |
| 64 | 🥇 4,516 (4 drafted) | 2,967 (none) | 1,170 (2 drafted) | 1,825 (none) |

512 in / 128 out, best setting per level:

| Requests | vLLM | SGLang | llama.cpp | mistral.rs |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 🥇 345 (4 drafted) | 214 (4 drafted) | 224 (2 drafted) | 133 (none) |
| 2 | 🥇 597 (4 drafted) | 386 (4 drafted) | 327 (2 drafted) | 244 (none) |
| 4 | 🥇 900 (4 drafted) | 577 (4 drafted) | 427 (2 drafted) | 439 (none) |
| 8 | 🥇 1,217 (4 drafted) | 711 (4 drafted) | 494 (2 drafted) | 591 (none) |
| 16 | 🥇 1,389 (1 drafted) | 1,026 (none) | 572 (2 drafted) | 1,121 (none) |
| 32 | 🥇 1,606 (1 drafted) | 1,209 (none) | 592 (2 drafted) | 1,031 (none) |
| 64 | 🥇 1,710 (1 drafted) | 1,373 (none) | 631 (none) | 1,132 (none) |

Peak card memory during each sweep, and median time per output token at 64 requests:

| Arm | Peak MiB | 128/512 ms per token at 64 | 512/128 ms per token at 64 |
| :--- | :--- | :--- | :--- |
| vllm | 19,692 | 18.163 | 27.436 |
| vllm-d1 | 19,462 | 26.777 | 42.641 |
| vllm-d4 | 22,370 | 51.176 | 82.339 |
| sglang | 19,411 | 19.232 | 20.984 |
| sglang-d4 | 16,225 | 112.49 | 127.644 |
| llama | 4,008 | 55.947 | 66.116 |
| llama-d2 | 4,298 | 50.064 | 64.651 |
| mistralrs-card | 6,856 | 14.36 | 15.299 |
