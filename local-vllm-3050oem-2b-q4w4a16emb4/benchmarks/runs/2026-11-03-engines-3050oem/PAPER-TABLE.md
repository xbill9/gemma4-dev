#### One request (tok/s, end to end)

| engine | no drafter | vs llama.cpp | best drafter | x no drafter | 4 at once | memory after warm-up |
|---|---|---|---|---|---|---|
| llama.cpp 7182ef71e, exact Q4_0 GGUF | 108.8 | 1.00 | 147.4 (llama-d2) | 1.35 | 277.4 | 2809 MiB |
| vLLM 0.33.0, W4A16 int4 tables | 121.1 | 1.11 | 230.5 (vllm-d4) | 1.90 | 454.6 | 6697 MiB |
| SGLang 0.5.23 (2 patches), W4A16 int4 tables | 119.9 | 1.10 | 158.9 (sglang-d4) | 1.33 | 446.7 | 6352 MiB |
| mistral.rs v0.9.5, exact Q4_0 GGUF | 118.9 | 1.10 | 85.8 (mistralrs-d1) | 0.72 | 316.8 | 4531 MiB |
| PyTorch int4 engine (this repo), W4A16 | 118.9 | 1.09 | none measured | - | 119.1 | 3121 MiB |
| LiteRT-LM 0.19.0, exact Q4_0 | 61.9 | 0.57 | 68.1 (litert-int248-d) | 1.10 | 63.6 | 1846 MiB |

#### Many requests (aggregate output tok/s, no drafter)

| engine | 512 in / 128 out: 1 | 8 | 64 | 128 in / 512 out: 1 | 8 | 64 | peak memory |
|---|---|---|---|---|---|---|---|
| llama.cpp | 95.3 | 228.1 | 367.6 | 107.1 | 312.6 | 633.3 | 2,621 MiB |
| vLLM | 105.4 | 422.2 | 603.1 | 120.0 | 777.2 | 1,651.2 | 7,091 MiB |
| SGLang | 103.4 | 391.2 | 544.0 | 119.6 | 779.7 | 1,711.3 | 6,966 MiB |
| mistral.rs | 107.9 | 270.2 | 258.5 | 111.8 | 308.7 | 308.0 | 4,755 MiB |
| PyTorch, LiteRT-LM | one request at a time | | | | | | |

#### Accuracy (one request at a time, greedy, thinking off)

| engine | GSM8K (1,319) | BFCL simple (400) | GSM8K wall s |
|---|---|---|---|
| llama.cpp | 0.9014 | 0.9225 | 4,993 |
| vLLM | 0.9014 | 0.9250 | 4,360 |
| SGLang | 0.8992 | 0.9150 | 4,491 |
| mistral.rs | 0.9030 | 0.9150 | 4,664 |
| PyTorch | 0.8992 | 0.9225 | 4,377 |
| LiteRT-LM | 0.8992 | 0.9125 | 7,805 |
