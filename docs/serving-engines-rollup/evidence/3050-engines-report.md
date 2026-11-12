# 2026-11-03 — every engine on the 3050, one sitting

Gemma 4 E2B on the RTX 3050 OEM (8 GB), every engine measured back to back by `engines.py` on the same six chat prompts, greedy, thinking off, up to 256 tokens per reply, no prompt cache. The card cooled to 50 °C or less before each engine and was empty between them.

| engine (arm) | weights | tok/s, one request | vs llama.cpp | decode tok/s | first token | 4 at once (total tok/s) | 4 at once vs one | GPU memory | ready in |
|---|---|---|---|---|---|---|---|---|---|
| PyTorch int4 (`pytorch`) | W4A16 QAT, int4 tables | 118.9 | 1.093 | — | — | 119.1 | 1.00 | 3,121 MiB | 236 s |
| LiteRT-LM (`litert-q4`) | exact Q4_0 `.litertlm` | 61.9 | 0.569 | 79.5 | 828 ms | 63.6 | 1.03 | 1,846 MiB | 2 s |
| LiteRT-LM (`litert-int248`) | Google's `.litertlm` | 62.0 | 0.569 | 82.1 | 857 ms | 64.5 | 1.04 | 1,316 MiB | 2 s |
| LiteRT-LM + built-in drafter (`litert-int248-d`) | Google's `.litertlm` | 68.1 | 0.626 | 95.2 | 857 ms | 68.1 | 1.00 | 1,372 MiB | 2 s |
| llama.cpp `7182ef71e` (`llama`) | exact Q4_0 GGUF | 108.8 | 1.000 | 110.5 | 42 ms | 277.4 | 2.55 | 2,809 MiB | 4 s |
| llama.cpp + Q4_0 drafter, 2 tokens (`llama-d2`) | exact Q4_0 GGUF | 147.4 | 1.355 | 150.7 | 42 ms | 285.4 | 1.94 | 2,885 MiB | 4 s |
| vLLM 0.33.0 (`vllm`) | W4A16 QAT, int4 tables | 121.1 | 1.113 | 124.5 | 53 ms | 454.6 | 3.75 | 6,697 MiB | 48 s |
| vLLM + W4A16 drafter, 4 tokens (`vllm-d4`) | W4A16 QAT, int4 tables | 230.5 | 2.118 | 244.5 | 55 ms | 470.7 | 2.04 | 7,345 MiB | 66 s |

All weights are on Google's QAT 4-bit grid except LiteRT's `int248` file (int2/int4/int8, one scale per row).

## Findings

- **With drafters, vLLM is fastest at every load measured here**: 230.5 tok/s for one request, 2.12x llama.cpp without a drafter and 1.56x llama.cpp with its drafter; 470.7 tok/s with four requests at once against llama.cpp's 285.4.
- **Without drafters, one request runs at about the same speed on PyTorch, vLLM and llama.cpp** (118.9, 121.1 and 108.8 tok/s). They separate with several requests at once: four requests give vLLM 3.75x one, llama.cpp 2.55x, PyTorch 1.00x.
- **PyTorch and LiteRT serve one request at a time**: four at once give the same total as one (1.00–1.04x).
- **LiteRT is about half llama.cpp's speed** per request (0.57x), and its first token takes 828–857 ms against 42–55 ms on llama.cpp and vLLM. Its built-in drafter adds 1.10x per request on these prompts.
- **vLLM costs memory and start time**: 6.7–7.3 GB of the 8 GB card and 48–66 s to ready, against 2.8–2.9 GB and 4 s for llama.cpp. LiteRT uses least (1.3–1.8 GB).
- Within one server, every engine repeated its greedy text exactly on all six prompts.

## Method

- One request: three non-streaming requests per prompt; tok/s = completion tokens / wall time; geometric mean over prompts of each prompt's median. This is the one figure every engine supports, because the PyTorch server does not stream.
- Decode and first token: three streaming requests per prompt (not available on PyTorch).
- 4 at once: four requests sent together, three repeats, median total tok/s.
- Memory: `nvidia-smi` after warm-up. `ready in`: server start to first healthy response.
- llama.cpp: `-c 8192 --parallel 4`. vLLM: rig defaults (`serve.py`). PyTorch: `openai_server.py --max-ctx 8192`. LiteRT-LM 0.19.0 `serve`.

Files: `engines.py` (runner), `summarize.py` (table → `SUMMARY.txt`), `results.jsonl` (every request), `server-*.log`, `config-litert-*.json`, `run.out`.
