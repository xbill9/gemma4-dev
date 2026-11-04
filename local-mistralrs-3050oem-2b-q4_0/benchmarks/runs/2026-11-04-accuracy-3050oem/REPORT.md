# mistral.rs v0.9.5 accuracy on the RTX 3050 OEM — 2026-11-04

Gemma 4 E2B, the exact Q4_0 GGUF that llama.cpp serves, with its mmproj, served by mistral.rs v0.9.5 from this
rig's `tpu.env` (command in `command.txt`), no drafter. Scored with `jev-tpu-v5e1/gen_eval.py`, one request at
a time, greedy, GSM8K at 2,048 tokens and BFCL simple at 512, the same settings as the other engines' runs on
this card, with `JEV_ENABLE_THINKING=0`. `run.sh` drives it; `compare.py` prints `COMPARE.txt`.

#### Result

| engine (same QAT int4 weights) | GSM8K (1,319) | BFCL simple (400) | GSM8K wall |
|---|---|---|---|
| mistral.rs v0.9.5 | **0.9030** (1,191) | **0.915** (366) | 4,664 s |
| vLLM 0.33.0 | 0.9014 (1,189) | 0.925 (370) | 4,360 s |
| llama.cpp exact Q4_0 | 0.9014 (1,189) | 0.9225 (369) | 4,993 s |
| PyTorch int4 engine | 0.8992 (1,186) | 0.9225 (369) | 4,377 s |
| LiteRT-LM exact Q4_0 | 0.8992 (1,186) | 0.9125 (365) | 7,805 s |

Paired by question against mistral.rs (95% paired range):

- GSM8K: vLLM +0.0015 [−0.0091, +0.0121], llama.cpp +0.0015 [−0.0091, +0.0114], PyTorch and LiteRT +0.0038
  [−0.0068, +0.0144]. Level with all four.
- BFCL: vLLM −0.0100 [−0.0225, +0.0000], llama.cpp −0.0075 [−0.0200, +0.0050], PyTorch −0.0075
  [−0.0200, +0.0025], LiteRT +0.0025 [−0.0125, +0.0175]. Level; vLLM's lead is 5 questions to 1, at the edge.

389 of 400 BFCL replies are byte-identical to llama.cpp's on the same GGUF, and 180 of 1,319 GSM8K replies.

#### Thinking

mistral.rs v0.9.5 thinks by default on Gemma 4; the other four engines do not. With the server default
BFCL scored 0.860 (344), 19 replies cut off at 512 tokens, median reply 304 tokens against 31 with thinking
off, in 1,183.6 s, 9.94x the 119.1 s with thinking off. Thinking off is +0.055 [+0.025, +0.0875] on BFCL at
this token cap. The harness now sends `chat_template_kwargs.enable_thinking` when `JEV_ENABLE_THINKING` is
set; a client of this server that wants short replies has to send it too.

#### Wall time

One request at a time: mistral.rs finished GSM8K 1.07x sooner than llama.cpp and took 1.07x vLLM's time;
BFCL 1.28x sooner than llama.cpp. The other engines ran on other days (llama.cpp 2026-10-14 on `77c4540ce`,
PyTorch 2026-10-31, LiteRT 2026-11-02, vLLM 2026-11-03), so the ratios are indicative.

#### Files

`gsm8k-mistralrs.{jsonl,log}`, `bfcl_simple-mistralrs.{jsonl,log}`, `driver.log` (GPU temperature, clock and
memory at each start), `server.log`; the thinking-default run as `*-thinkdefault.*`.
