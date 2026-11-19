# 2026-11-05 — six serving engines on one NVIDIA L4 (EC2 G6)

The 3050 flight (`local-vllm-3050oem-2b-q4w4a16emb4/benchmarks/runs/2026-11-03-engines-3050oem`) moved onto
one L4: the same six engines, the same Gemma 4 E2B weights on Google's QAT Q4_0 grid, the same six prompts,
statistics and arm names. One engine served at a time on the card; every arm ran in one sitting on one
instance, 22:15–23:27 UTC.

## Setup

| | |
|---|---|
| instance | `i-040ed957b67c2d514`, **g6.4xlarge** on-demand, us-east-1d (16 vCPUs, 60 GiB). The rigs' `tpu.env` names g6.2xlarge; it had no spot or on-demand capacity in any us-east-1 zone that day, nor g6.4xlarge spot |
| card | NVIDIA L4, 23,034 MiB, driver 595.112.07 (`nvidia-smi.txt`); 2,040 MHz during most arms, 1,920 (mistral.rs) and 1,875 (PyTorch) at their single sample |
| image | PyTorch 2.13 / Ubuntu 26.04 DLAMI `ami-0f5139ed9ed52b805`, system python 3.14.4 |
| llama.cpp | `7182ef71e6d5dd0c8ea76486fac134fcd25025d4` (the 3050 flight's), CUDA 13.1, `-t 4 -tb 8` |
| vLLM | 0.33.0 with its own `torch 2.14.0+cu130` from PyPI (`setup-logs/vllm.log`: CUDA 13.0, runs on this driver) |
| SGLang | 0.5.23 + the 3050 rig's two patches |
| mistral.rs | v0.9.5, built here with CUDA 13.1 for sm_89 |
| PyTorch | `gpu-pytorch-g6-2b-q4w4a16`'s int4 engine on the DLAMI's torch 2.13 |
| LiteRT-LM | 0.19.0, GPU through WebGPU on Vulkan (the DLAMI lists the L4 as a Vulkan device) |

Requests (`engines.py`): six prompts, greedy, thinking off, 256 tokens; one request at a time, 3 repeats each,
non-streaming (`e2e`) and streaming (`decode`, TTFT); then 4 requests at once, 3 repeats. Memory is the card's
after warm-up. `repeat` counts prompts whose greedy text was identical in all three repeats.

## Results (`SUMMARY.txt`, from `summarize.py .`)

```
arm              e2e tok/s vs llama  decode TTFT ms 4 at once  x one    memory ready s repeat
llama                135.7    1.000   136.9      29     437.6   3.23  2930 MiB     6.1   6/6
llama-d2             207.7    1.531   211.4      30     373.2   1.80  3004 MiB     4.1   6/6
mistralrs            144.6    1.066   144.4      11     500.0   3.46  4664 MiB    24.1   6/6
mistralrs-d1         137.6    1.015   138.0      11     299.2   2.17  4720 MiB     4.1   6/6
pytorch              134.7    0.993       -       -     134.5   1.00  3248 MiB   236.2   6/6
litert-q4             84.4    0.622   103.3     502      86.2   1.02  2012 MiB    14.5   6/6
litert-int248         98.4    0.725   125.2     512     100.5   1.02  1489 MiB     9.4   6/6
litert-int248-d      112.8    0.832   143.7     510     115.9   1.03  1495 MiB     5.1   6/6
vllm                 140.4    1.035   140.7      16     537.2   3.83 18228 MiB   167.3   6/6
vllm-d4              275.3    2.030   281.2      19     711.3   2.58 18918 MiB   132.5   6/6
sglang               136.5    1.006   139.7      50     475.4   3.48 18345 MiB   307.2   6/6
sglang-d4            189.6    1.398   198.1      55     508.3   2.68 15191 MiB    61.1   6/6
mistralrs-card       144.6    1.066   144.4      11     499.8   3.46  5286 MiB     4.1   6/6
```

#### Findings

- **For one request without a drafter, five engines are within 7% of each other**: mistral.rs 144.6, vLLM 140.4,
  SGLang 136.5, llama.cpp 135.7 and PyTorch 134.7 tok/s (0.993–1.066x llama.cpp). LiteRT-LM is behind at
  0.62x (exact Q4_0 build) and 0.73x (Google's int248 file), and takes about 0.5 s to its first token
  against 11–55 ms for the others.
- **vLLM with Google's drafter is the fastest single stream**: 275.3 tok/s at 4 drafted tokens, 1.96x vLLM
  without it and 1.33x llama.cpp with its best drafter. Drafter gains for one request, each against the same
  engine without it: vLLM 1.96x, llama.cpp 1.53x, SGLang 1.39x, LiteRT 1.15x, mistral.rs 0.95x.
- **At 4 requests at once the drafter pays only on vLLM, SGLang and LiteRT**: vLLM 1.32x (711.3 tok/s, the
  highest), SGLang 1.07x, LiteRT 1.15x; llama.cpp 0.85x and mistral.rs 0.60x of no drafter. Without drafters
  vLLM 537.2, mistral.rs 500.0, SGLang 475.4 and llama.cpp 437.6 tok/s; PyTorch and LiteRT serve one request
  at a time (1.00–1.03x).
- **Every arm repeated its greedy text exactly**, 6 of 6 prompts over three repeats each.
- **mistral.rs does not load at 64 requests on the 23 GB card.** It plans activation memory per running
  request, about 395 MiB each: at 64 it asked for 25,316 MiB plus 2,976 MiB fixed against 21,192 MiB usable;
  at 32 with a 65,536-token cache, 15,966 MiB of activations left too little for the cache
  (`mistralrs-card-nofit/`). 32 requests with a 16,384-token cache serves (`mistralrs-card`, 5,286 MiB),
  and at 4 requests it matches the 3050 limits (499.8 against 500.0 tok/s), as it should.
- vLLM and SGLang reserve most of the card by design (0.85 of it); memory is not a ranking here, since each
  engine had the card to itself.

## Against the 3050 (`COMPARE-3050.txt`, from `compare_3050.py`)

```
arm                L4 one  3050 one  L4/3050  L4 four  3050 four  L4/3050
llama               135.7     108.8    1.247    437.6      277.4    1.578
llama-d2            207.7     147.4    1.409    373.2      285.4    1.308
mistralrs           144.6     118.9    1.216    500.0      316.8    1.578
mistralrs-d1        137.6      85.8    1.604    299.2      141.9    2.109
pytorch             134.7     118.9    1.133    134.5      119.1    1.129
litert-q4            84.4      61.9    1.363     86.2       63.6    1.355
litert-int248        98.4      62.0    1.587    100.5       64.5    1.558
litert-int248-d     112.8      68.1    1.656    115.9       68.1    1.702
vllm                140.4     121.1    1.159    537.2      454.6    1.182
vllm-d4             275.3     230.5    1.194    711.3      470.7    1.511
sglang              136.5     119.9    1.138    475.4      446.7    1.064
sglang-d4           189.6     158.9    1.193    508.3      288.0    1.765
```

The 3050 values are from its three engine runs (vLLM's for vLLM, llama.cpp, PyTorch and LiteRT; mistral.rs's
and SGLang's own), summarized by the same `summarize.py`. One request without a drafter runs 1.13–1.25x faster
on the L4 for the CUDA engines and 1.36–1.59x for LiteRT. **The engine order is the same on both cards**: vLLM
with its drafter first, the batching engines level for one request, LiteRT last. Two things move: llama.cpp's
drafter pays more on the L4 (1.53x against 1.36x), as `gpu-llamacpp-g6-2b-q4_0/benchmarks/runs/2026-10-26-drafter-g6`
found, and mistral.rs's drafter loses less (0.95x against 0.72x). Two hosts on two days (different CPUs,
16 vCPUs here against a 6-core i7-5820K), so an L4 figure against a 3050 figure is a comparison of two
machines; ratios between engines on one card are the result.

## Files

- `results.jsonl` — every request (489 rows; PyTorch has no streaming rows); `server-<arm>.log` per arm;
  `config-litert-*.json` the LiteRT server configs.
- `engines.py`, `summarize.py` — the driver and summary as run (copies of `../../../flight/`);
  `compare_3050.py` — the table above.
- `flight-setup.out`, `setup-logs/` — installation of every engine on the instance; `flight-run*.out` — the
  three driver sittings (all arms; vLLM, SGLang and mistralrs-card after a PATH fix; mistralrs-card at 32 ×
  16,384); `nvidia-smi.txt`.
- `mistralrs-card-nofit/` — mistral.rs refusing 64 × 65,536 and 32 × 65,536.

Not run here: the concurrency sweeps (1–64 requests) and GSM8K/BFCL accuracy (`../../../flight/FLIGHT.md`, step 6).
