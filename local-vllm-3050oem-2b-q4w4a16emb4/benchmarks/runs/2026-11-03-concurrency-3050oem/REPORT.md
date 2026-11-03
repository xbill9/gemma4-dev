# 2026-11-03 — concurrency sweep: vLLM against llama.cpp on the RTX 3050 OEM

**Question:** with the same QAT Q4_0 E2B weights on the same card, how does vLLM's output scale with
requests at once, against llama.cpp?

**Answer:** vLLM gives more output at every level. One request: 1.11–1.12x llama.cpp. From 8 requests
up: 1.64–1.85x at 512 in / 128 out and 2.43–2.61x at 128 in / 512 out, peaking at 1,651 tok/s against
633. Time per token stays at about half llama.cpp's from 8 requests up. vLLM holds 7,082 MiB of the
8 GB card to llama.cpp's 2,612.

## Method

`sweep.py` concurrency mode with the arguments of llama.cpp's sweeps on this card: shapes 512 in / 128
out and 128 in / 512 out, levels 1–64, 5 repeats per level (median reported), unique prompts with a
leading nonce so vLLM's prefix cache cannot hit, decode timed on the stream, thinking on, greedy. Both
engines ran without a drafter, one after the other on the same day:

- **vLLM** 0.33.0 (torch 2.14.1+cu130, `versions.txt`), `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4`
  at `db715f83`, Marlin int4 kernels, Triton attention, `--max-num-seqs 64 --max-model-len 16384
  --gpu-memory-utilization 0.85` (`command.txt`). KV pool 259,018 tokens.
- **llama.cpp** `7182ef71e`, exact Q4_0 GGUF (`419db9a6…`), 64 slots × 1,024 tokens, flash attention:
  `local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-11-03-concurrency-3050oem`. Within 0.7–2.8% of
  the same sweep on `77c4540ce` (2026-10-13) at every level.

Both servers attested `device=gpu` from the live process. `compare.py` computes the table
(`COMPARE.txt`). vLLM's copy of `sweep.py` reads thinking tokens from `delta.reasoning`, the field vLLM
0.33 streams them in, as well as llama.cpp's `delta.reasoning_content`.

## 512 in / 128 out

| requests | vLLM tok/s | llama.cpp tok/s | ratio | TTFT ms vLLM / llama.cpp | TPOT ms vLLM / llama.cpp |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 105.4 | 95.3 | 1.11 | 201 / 202 | 9.28 / 9.19 |
| 2 | 183.3 | 160.9 | 1.14 | 357 / 343 | 9.49 / 10.06 |
| 4 | 294.0 | 201.2 | 1.46 | 647 / 636 | 10.02 / 15.37 |
| 8 | 422.2 | 228.1 | 1.85 | 1,228 / 1,333 | 10.90 / 25.34 |
| 16 | 531.0 | 321.5 | 1.65 | 2,390 / 2,615 | 13.21 / 29.94 |
| 32 | 587.3 | 354.7 | 1.66 | 4,415 / 5,086 | 23.65 / 51.22 |
| 64 | 603.1 | 367.6 | 1.64 | 6,748 / 8,154 | 59.64 / 111.37 |

## 128 in / 512 out

| requests | vLLM tok/s | llama.cpp tok/s | ratio | TTFT ms vLLM / llama.cpp | TPOT ms vLLM / llama.cpp |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 120.0 | 107.1 | 1.12 | 123 / 129 | 8.90 / 9.14 |
| 2 | 231.5 | 193.2 | 1.20 | 184 / 186 | 9.08 / 10.04 |
| 4 | 435.0 | 262.8 | 1.66 | 279 / 326 | 9.46 / 14.52 |
| 8 | 777.2 | 312.6 | 2.49 | 431 / 577 | 10.16 / 24.41 |
| 16 | 1,235.9 | 508.9 | 2.43 | 765 / 1,095 | 12.22 / 29.08 |
| 32 | 1,544.3 | 592.7 | 2.61 | 1,496 / 2,059 | 19.02 / 49.48 |
| 64 | 1,651.2 | 633.3 | 2.61 | 2,787 / 3,748 | 35.62 / 93.11 |

- **Time per token is where the engines part.** llama.cpp's rises from 9.2 to 15 ms at 4 requests and 25 ms
  at 8; vLLM's stays at 9–11 ms up to 8 and 12–13 ms at 16. Time to first token is level up to 4 requests.
- **vLLM is steadier.** Every vLLM cell repeated within 1.5%. llama.cpp's 512 in / 128 out cells at 8, 16
  and 32 requests spread 21.1%, 26.0% and 14.9% across repeats, as on 2026-10-13; all its other cells
  stayed within 1.6%.
- **Memory.** vLLM reserves 85% of the card at start (7,082 MiB, `vram.txt`); llama.cpp's 64 slots held
  2,612 MiB (2,621 peak). Only llama.cpp leaves room for a second model on the card.
- **Drafters are out of this run.** llama.cpp with its drafter serves one request at 151 tok/s
  (`local-llamacpp-3050oem-2b-q4_0`, 2026-10-14), ahead of vLLM's 120 without one; vLLM with Google's
  QAT drafter is the next measurement.
