# 2026-11-02 — first light, LiteRT-LM on the RTX 3050 OEM

**Question:** does LiteRT-LM run Gemma 4 E2B on the 3050, how fast, and does its built-in drafter pay
on real prompts?

**Answer:** yes, on the GPU through WebGPU on Vulkan. 82.6 tok/s generation without the drafter and
1.17x with it on six chat prompts (0.87x to 1.80x), in about 1.4 GB of the card. The drafter stays
on (`SPECULATIVE_DECODING=on`).

## Setup

- Host `gpu`, RTX 3050 OEM (GA106, sm_86, 8192 MiB), driver 615.78.08, Vulkan loader 1.4.363.
- `litert-lm` 0.19.0 (`litert-lm-api` 0.19.0), protobuf 7.36.2, system python3 3.14.10.
- Model `litert-community/gemma-4-E2B-it-litert-lm` `gemma-4-E2B-it.litertlm`, revision `cbb97a3e`,
  SHA-256 `18193810…a63c`. Weight encoding in `encoding.txt` (int2/int4/int8, one scale per row).
- The engine picked the card itself: `Selected adapter: NVIDIA GeForce RTX 3050 OEM, arch=ampere,
  vendor=nvidia, backend=Vulkan` (`bench-gpufile.log`).

## Engine benchmark (`litert-lm benchmark --backend gpu -p 1024 -d 256 --runs 3`)

| | drafter off | drafter on |
| --- | ---: | ---: |
| prompt reading | 4,963 tok/s | 5,169 tok/s |
| generation | 85.81 tok/s | 164.21 tok/s |
| time to first token | 0.218 s | 0.205 s |

`bench-drafter-{off,on}.log`. The 164 tok/s with the drafter is the benchmark's own synthetic text
and does not carry to chat (next section).

`--gpu-decode-steps-per-sync` 1 / 2 / 4 / 8 (drafter off): 86.12 / 85.18 / 85.36 / 84.97 tok/s, no
gain (`bench-sync*.log`). While generating, the process used about 195% CPU and the GPU showed 66%
busy (one `nvidia-smi` sample).

The repo's `gemma-4-E2B-it-gpu.litertlm` (text only, no drafter, different encoding) measured 73.22
tok/s generation and 4,447 tok/s prompt reading at 256/256, one run (`bench-gpufile.log`).

## Real prompts, drafter off / on / adaptive (`spec_ab.py`)

The six prompts of `local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-13-spec-mtp-3050oem`,
greedy, 256 tokens maximum, 3 repeats, one server per arm, GPU at or below 40 °C before each arm.
Generation speed measured on the stream (the server reports no timings). Median of 3, ratio to off,
spread (max−min)/median:

| prompt | off | on | adaptive |
| --- | ---: | ---: | ---: |
| code | 83.6 | **150.4 (1.80x)** | 136.3 (1.63x) |
| chat | 78.1 | **94.4 (1.21x)** | 89.0 (1.14x) |
| story | 82.8 | 81.8 (0.99x) | 83.4 (1.01x) |
| explain | 78.1 | **97.9 (1.25x)** | 84.4 (1.08x) |
| facts | 82.5 | 89.3 (1.08x) | 87.0 (1.05x) |
| summary | 84.2 | 73.4 (0.87x) | 65.6 (0.78x) |
| **geometric mean** | 1.000 | **1.168** | 1.089 |

- Spread without the drafter is 6.7–11.9% per prompt, against 0.7% for llama.cpp on this card; with it,
  1.9–8.3%. The adaptive arm's summary cell spread 54%.
- With the drafter on, 0.468 stream chunks per generated token (accepted drafts arrive together); off,
  0.999.
- GPU memory after load: 1,316 MiB off, 1,381 MiB on; at most 1,407 MiB during the run.
- First token after the request: median 0.85 s for these short prompts, both arms, against 0.2 s in the
  engine benchmark at 1,024 tokens. Not investigated.
- **Greedy text:** each of the off and on arms repeats itself exactly (1 distinct reply per prompt over 3
  repeats); the drafter changes the reply on all 6 prompts; the adaptive arm does not repeat itself (up
  to 3 distinct replies per prompt). Quality was not scored. Texts in `texts.json`.

## Against llama.cpp on the same card (indicative)

`local-llamacpp-3050oem-2b-q4_0` on the same six prompts (2026-10-13, llama.cpp `77c4540ce`, exact Q4_0
GGUF): 108.2 tok/s without a drafter, geometric mean 1.386x with its 2-token drafter. LiteRT-LM here is
about 0.76x of that without a drafter (82.6 / 108.2, medians), with a smaller drafter gain. Different
weights (Google's int2/int4/int8 mix against Q4_0), engines and days, so this is not a controlled A/B.

## Server quirks found

- `max_tokens` is ignored; only `max_completion_tokens` limits a reply. A request with `max_tokens: 256`
  returned 1,082–1,137-token replies on these prompts.
- `litert-lm convert` is an empty command in 0.19.0.

## Files

`spec_ab.py` (A/B driver), `results.jsonl`, `summary.txt`, `texts.json`, `server-*.log`, `config-*.json`,
`bench-*.log`, `encoding.txt` with `tflite_encoding.py` (it needs the PyPI `tflite`
2.18.0 schema on `sys.path` and sections dumped by `litert-lm-peek --dump_files_dir`).
