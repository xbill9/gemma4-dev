# 2026-11-02 — accuracy: LiteRT-LM against llama.cpp on the RTX 3050 OEM

**Question:** on the same card, do LiteRT-LM's two E2B builds answer as well as llama.cpp's exact Q4_0
GGUF? One is Google's `gemma-4-E2B-it.litertlm` (int2/int4/int8, one scale per row); the other is this rig's
build holding the QAT Q4_0 grid exactly, the same int4 codes and scales as llama.cpp's file.

**Answer:** on GSM8K the exact build is level with llama.cpp (0.8992 against 0.9014) and Google's file is
0.022 behind. On BFCL both trail through the stock server, 0.8625 and 0.825 against 0.9225, because of
the server's constrained decoding. With it off, the exact build is level with llama.cpp on BFCL too
(0.9125; `local-litertlm-3050oem-2b-int248/benchmarks/runs/2026-11-03-tools-constraint-3050oem`).
LiteRT takes 1.46–1.56x llama.cpp's time on GSM8K.

## Method

GSM8K (1,319, 2,048-token limit) and BFCL simple (400, 512-token limit) through
`jev-tpu-v5e1/gen_eval.py`, greedy, one request at a time: the harness, sets and limits of the llama.cpp
and PyTorch accuracy runs on this card. Stock `litert-lm serve` 0.19.0, GPU backend, drafter off in both
arms so only the weights differ (`run.sh`, `config-*.json`). `proxy.py` moves `max_tokens` to
`max_completion_tokens`, the only limit the server reads. The q4exact GSM8K arm was run by `resume.sh` on
2026-11-03 with the same server config, proxy and harness; 11 minutes of it shared the card with the
constrained-decoding BFCL runs, which lengthens its wall time and leaves greedy answers unchanged.
llama.cpp is `local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-accuracy-fullctx-3050oem`
(llama.cpp `77c4540ce`, exact Q4_0 E2B, no drafter, `-c 131072`). `compare.py` computes every figure;
ranges are 95% paired bootstrap (2,000 resamples, seed 0).

## Results

| task | Google int248 | exact q4exact | llama.cpp exact Q4_0 |
| --- | ---: | ---: | ---: |
| GSM8K | 0.8795 (1,160) | **0.8992** (1,186) | 0.9014 (1,189) |
| BFCL simple | 0.8250 (330) | 0.8625 (345) | 0.9225 (369) |
| GSM8K wall time | 7,266 s | 7,805 s | 4,993 s |

| difference | GSM8K | BFCL simple |
| --- | --- | --- |
| q4exact − llama.cpp | −0.0023 [−0.0121, +0.0068] | −0.0600 [−0.0875, −0.0350] |
| int248 − llama.cpp | −0.0220 [−0.0379, −0.0061] | −0.0975 [−0.1325, −0.0625] |
| q4exact − int248 | +0.0197 [+0.0045, +0.0364] | +0.0375 [+0.0125, +0.0650] |

- **The exact QAT grid answers the same in either runtime on GSM8K.** The range includes zero; 234 of
  1,319 replies are word-for-word identical to llama.cpp's.
- **Google's mobile build costs answers.** It is behind the exact build on both tasks, ranges above zero.
  Its int2 layers and per-row scales are the difference between the two LiteRT arms, which share the
  runtime, server and settings.
- **The BFCL gap to llama.cpp is the server's.** `litert-lm serve` constrains every tool call to the
  schema's declared argument order and the model drops optional arguments under it; with the constraint
  off the exact build scores 0.9125, −0.0100 [−0.0225, +0.0025] against llama.cpp.
- No reply hit the 2,048-token limit in either LiteRT arm (llama.cpp: 3).
