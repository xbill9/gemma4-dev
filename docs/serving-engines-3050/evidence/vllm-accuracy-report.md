# vLLM 0.33.0 accuracy on the RTX 3050 OEM — 2026-11-03

Gemma 4 E2B, `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4` @ `db715f83`, served by `../../../serve.py`
from `tpu.env` (command in `command.txt`), no drafter. Scored with `jev-tpu-v5e1/gen_eval.py`, one request
at a time, thinking off, greedy, GSM8K at 2,048 tokens and BFCL simple at 512, the same settings as the
other engines' runs on this card. `run.sh` drives it; `compare.py` prints `COMPARE.txt`.

#### Result

| engine (same QAT int4 weights) | GSM8K (1,319) | BFCL simple (400) | GSM8K wall |
|---|---|---|---|
| vLLM 0.33.0 | **0.9014** (1,189) | **0.925** (370) | 4,360 s |
| llama.cpp exact Q4_0 | 0.9014 (1,189) | 0.9225 (369) | 4,993 s |
| PyTorch int4 engine | 0.8992 (1,186) | 0.9225 (369) | 4,377 s |
| LiteRT-LM exact Q4_0 | 0.8992 (1,186) | 0.9125 (365) | 7,805 s |

Paired by question against vLLM (95% paired range, 2,000 resamples, seed 0):

- GSM8K: llama.cpp +0.0000 [−0.0106, +0.0106], PyTorch +0.0023 [−0.0068, +0.0114], LiteRT +0.0023 [−0.0061, +0.0114]. Level with all three.
- BFCL: llama.cpp and PyTorch +0.0025 [+0.0000, +0.0075], one question apart, level. LiteRT (constrained decoding off) +0.0125 [+0.0025, +0.0250], 5 questions, all in vLLM's favour.

**Answers are the same on every engine that serves the exact QAT grid.** vLLM and llama.cpp get the same
1,189 GSM8K questions right in count, with 24 won and 24 lost each way, and 396 of 400 BFCL replies are
byte-identical to llama.cpp's.

#### Wall time

Wall time is one request at a time, so it reads as single-stream speed: vLLM finished GSM8K 1.15x sooner than
llama.cpp, level with PyTorch and 1.79x sooner than LiteRT. The other engines ran on other days (llama.cpp
2026-10-14 on `77c4540ce`, PyTorch 2026-10-31, LiteRT 2026-11-02), so the ratios are indicative.

#### Files

`gsm8k-vllm.{jsonl,log}`, `bfcl_simple-vllm.{jsonl,log}`, `driver.log` (GPU temperature, clock and memory at
each start), `server.log`.
