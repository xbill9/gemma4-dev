# SGLang 0.5.23 accuracy on the RTX 3050 OEM — 2026-11-04

Gemma 4 E2B, `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4` @ `db715f83` (the checkpoint and revision vLLM
served), served by `../../../serve.py` from `tpu.env` with the rig's two patches (command in `command.txt`), no
drafter, SGLang's gemma4 tool-call and reasoning parsers. Scored with `jev-tpu-v5e1/gen_eval.py`, one request
at a time, greedy, thinking off, GSM8K at 2,048 tokens and BFCL simple at 512, the same settings as the other
engines' runs on this card. `run.sh` drives it; `compare.py` prints `COMPARE.txt`.

#### Result

| engine | GSM8K (1,319) | BFCL simple (400) | GSM8K wall |
|---|---|---|---|
| SGLang 0.5.23 | **0.8992** (1,186) | **0.915** (366) | 4,491 s |
| vLLM 0.33.0 | 0.9014 (1,189) | 0.925 (370) | 4,360 s |
| llama.cpp exact Q4_0 | 0.9014 (1,189) | 0.9225 (369) | 4,993 s |
| mistral.rs v0.9.5 | 0.9030 (1,191) | 0.915 (366) | 4,664 s |
| PyTorch int4 engine | 0.8992 (1,186) | 0.9225 (369) | 4,377 s |
| LiteRT-LM exact Q4_0 | 0.8992 (1,186) | 0.9125 (365) | 7,805 s |

Paired by question against SGLang (95% paired range):

- GSM8K: level with all five (vLLM −0.0023 [−0.0114, +0.0068], llama.cpp −0.0023, mistral.rs −0.0038,
  PyTorch and LiteRT +0.0000).
- BFCL: level with llama.cpp, mistral.rs, PyTorch and LiteRT. **4 questions behind vLLM** (−0.0100
  [−0.0200, −0.0025], 0 won, 4 lost), with 393 of 400 replies identical to vLLM's.

Of those four, two are SGLang's tool-call parser: the model writes the arguments `1e-9` and `100e-06`, and
SGLang returns them as strings where vLLM returns numbers (`simple_38`, `simple_42`). The other two are
different replies from the model: an interest rate of 5 where vLLM's reply has 0.05 (`simple_145`), and a
question back to the user instead of a call (`simple_290`). With the parser difference set aside SGLang is two
questions behind vLLM, level.

#### Wall time

One request at a time: GSM8K 1.11x sooner than llama.cpp and 1.03x vLLM's time; BFCL 1.15x sooner than
llama.cpp. The other engines ran on other days, so the ratios are indicative.

#### Files

`gsm8k-sglang.{jsonl,log}`, `bfcl_simple-sglang.{jsonl,log}`, `driver.log` (GPU temperature, clock and memory
at each start), `server.log`, `command.txt`.
