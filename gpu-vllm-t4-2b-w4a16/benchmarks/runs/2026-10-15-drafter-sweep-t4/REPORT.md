# 2026-10-15 — Gemma 4 drafters on the Tesla T4 under vLLM, E2B and E4B

**Answer: both drafters speed both models up, the gain keeps rising to 4 drafted tokens, and the
QAT and non-QAT drafters are level.** One request at a time, greedy: E2B 1.29x at the serving 3
tokens and 1.31x at 4; E4B 1.57x at 3 and 1.62x at 4. Sampled at temperature 0.8, E2B peaks at 3
tokens (1.17x) and loses at 1 (0.93x); E4B gains at every length (1.15x at 1, 1.48x at 4).

## Setup

`run.sh` on the GCE `n1-standard-2` with the attached Tesla T4 (15,360 MiB, driver 615.71.09),
vLLM 0.29.0, torch 2.13.0+cu130, transformers 5.17.0 (`host.txt`).

- **Targets**, served with each rig's `tpu.env` flags as `~/bin/vllm-t4` passes them (fp16,
  `--max-model-len 16384 --max-num-seqs 8 --gpu-memory-utilization 0.90 --language-model-only`):
  E2B `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4` (`gpu-vllm-t4-2b-w4a16`, revision
  `db715f83`), E4B `xbill9/gemma-4-E4B-it-qat-q4_0-w4a16-ct-text-emb4`
  (`gpu-vllm-t4-4b-q4w4a16emb4`, `2c02a2c8`).
- **Drafters**, bf16 safetensors from the Hugging Face cache, offline: `qat` is
  `google/gemma-4-E?B-it-qat-q4_0-unquantized-assistant` (the rigs' `SPECULATIVE_MODEL`; E2B
  `ebc7e1a2`, E4B `27f8d204`); `nq` is `google/gemma-4-E?B-it-assistant` (E2B `2d874ef7`, E4B
  `8d0031ea`, the revisions the 3050 run converted). The exact Q4_0 GGUF drafters of the 3050 and
  1650 Ti runs cannot load in vLLM.
- **`spec_ab.py`**: the six prompts of the 3050 and 1650 Ti runs, greedy x3 plus one sampled pass
  (temperature 0.8, seed 1234), 256 tokens, one request at a time, after one warm-up request.
  Decode tok/s per request is (tokens − 1) over the time between the first and last streamed
  token; acceptance is the difference of the server's spec-decode counters across that request.
  One server per arm; card cooled to ≤50 °C before each. Arms: none; each drafter at 1/2/3/4
  tokens; qat at 3 again; none again. `summarize.py` is the 1650 Ti run's, unchanged.
- **E4B is missing its last arm** (no drafter, repeated): the run stopped after the qat-n3 repeat.
  Drift on E4B is read from that repeat instead.

## Results (`e2b/summary.txt`, `e4b/summary.txt`)

| greedy, geometric mean of 6 prompts | E2B tok/s | x | acceptance | E4B tok/s | x | acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no drafter | 110.7 | 1.000 | — | 60.9 | 1.000 | — |
| qat, 1 token | 113.8 | 1.028 | 0.624 | 74.6 | 1.224 | 0.628 |
| qat, 2 tokens | 135.3 | 1.222 | 0.498 | 88.0 | 1.444 | 0.479 |
| **qat, 3 tokens (serving)** | **142.6** | **1.288** | 0.385 | **95.9** | **1.573** | 0.390 |
| qat, 4 tokens | 145.0 | 1.309 | 0.313 | 98.9 | 1.622 | 0.321 |
| nq, 1 token | 112.8 | 1.018 | 0.575 | 74.6 | 1.224 | 0.601 |
| nq, 2 tokens | 131.6 | 1.188 | 0.449 | 88.7 | 1.455 | 0.464 |
| nq, 3 tokens | 139.0 | 1.255 | 0.348 | 96.6 | 1.585 | 0.380 |
| nq, 4 tokens | 143.0 | 1.291 | 0.290 | 99.5 | 1.632 | 0.309 |
| qat, 3 tokens, repeated | 145.0 | 1.310 | 0.385 | 98.2 | 1.612 | 0.390 |
| no drafter, repeated last | 111.6 | 1.008 | — | not run | | |

| sampled (temperature 0.8), x vs none | qat 1 | qat 2 | qat 3 | qat 4 | nq 1 | nq 2 | nq 3 | nq 4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E2B | 0.934 | 1.069 | **1.173** | 1.125 | 0.951 | 1.078 | 1.138 | 1.090 |
| E4B | 1.150 | 1.396 | 1.412 | **1.483** | 1.096 | 1.291 | 1.380 | 1.425 |

- **Code gains most, stories least.** E2B at 3 tokens: code 1.97x, story 1.02x; at 1 token story
  is 0.91x. E4B at 3 tokens: code 2.27x, story 1.19x.
- **QAT against non-QAT drafter**: E2B qat ahead by 1–3% at 2–4 tokens with higher acceptance at
  every length (0.385 against 0.348 at 3); E4B level (within 1%) greedy, qat ahead sampled. On TPU
  v5e the non-QAT drafter was far ahead (`QUANTIZATION.md`); on the T4 it is not.
- **Drift**: E2B's repeated no-drafter arm is 1.008x the first and its repeated qat-n3 arm 1.017x;
  E4B's repeated qat-n3 is 1.025x the first. Gains between neighbouring draft lengths smaller than
  that (E2B 3 against 4 tokens greedy) are not resolved by this run. Within-cell spread: E2B up to
  6.5% (qat-n4), E4B under 1.7%. Decode without a drafter slips about 3% within an arm as the card
  warms from 61 to 73 °C.
- **Memory** (engine log, `server-*.log`): weights 2.86 GiB for E2B alone, 3.01 GiB with either
  drafter; 4.54 / 4.69 GiB for E4B. The KV pool is smaller with the non-QAT drafter than with the
  QAT one (E2B 945,358 against 1,063,683 tokens; E4B 291,717 against 292,092), and nvidia-smi
  reported ~1.1 GB more in use on E2B's non-QAT arms. Cause not investigated.
- **Startup**: 220–290 s to healthy for the first servers, 90–135 s once the compile cache was warm.

## Greedy text

- **No drafter is deterministic on both models**: the three repeats of each prompt are identical,
  and E2B's repeated no-drafter arm matches the first on 18 of 18.
- **With a drafter, the same prompts change at every draft length, with either drafter.** E2B:
  story, facts and summary differ from no drafter (27 of 30 replies each; the 3 matches are the
  repeated no-drafter arm), code, chat and explain match. E4B: only explain differs. The drafted
  replies agree with each other (E2B story and summary have 2 distinct texts across all arms,
  facts 3), and each diverges partway in (E2B story at character 136, facts 216, summary 621),
  the pattern of a near-tie between two tokens decided differently when the target verifies
  several positions in one pass. Not investigated further. On the 3050 under llama.cpp, E2B at 1
  drafted token matched no drafter on 18 of 18.

## Against the 3050 and 1650 Ti (indicative only: different engine, card and host)

| best greedy setting, x vs no drafter | T4, vLLM | RTX 3050, llama.cpp | GTX 1650 Ti, llama.cpp |
| --- | ---: | ---: | ---: |
| E2B | 1.31x (4 tokens) | 1.40x (2) | 1.18x (1) |
| E4B | 1.62x (4 tokens) | 1.72x (2) | 1.43x (1 or 2) |

Acceptance at a given draft length is close across all three (2 tokens: E2B 0.498 / 0.475 /
0.495, E4B 0.479 / 0.481 / 0.504), so the best length differs by what a drafted step costs
against a target step on each engine and card, not by how well the drafter guesses. That reading
is inferred, not measured.

## W4A16 drafters (`e?b-q4ct/`, `q4ct-sha256.txt`)

`repack/repack_drafter.py` repacks the QAT drafters' 22 linears to compressed-tensors W4A16 (all
on the 4-bit grid); the embedding stays bf16 because vLLM reads its rows directly for the
drafter's logits, so the file shrinks only to 143,062,680 B (E2B) and 143,505,048 B (E4B) from
157,565,344 and 159,138,208. vLLM 0.29.0 loads them beside the W4A16 target and reports
`MarlinLinearKernel` for them. Same harness, arms none, bf16 QAT at 3, W4A16 at 3 and 4, none
again, run back to back **without** the cool-down:

| greedy, geomean of 6 prompts | E2B tok/s | acceptance | E4B tok/s | acceptance |
| --- | ---: | ---: | ---: | ---: |
| no drafter | 113.7 | — | 60.1 | — |
| bf16 QAT, 3 tokens | 143.3 | 0.385 | 94.2 | 0.390 |
| **W4A16 QAT, 3 tokens** | **141.5** | 0.387 | **94.4** | 0.390 |
| W4A16 QAT, 4 tokens | 142.2 | 0.315 | 97.8 | 0.321 |
| no drafter, repeated last | 105.3 | — | 60.1 | — |

- W4A16 against bf16 at 3 tokens, neighbouring servers: E2B 0.987x greedy and 0.968x sampled,
  E4B 1.002x and 0.995x, at the same acceptance. Greedy text is identical to the bf16 drafter's
  on both models. Weights in the engine log: 3.00 against 3.01 GiB (E2B).
- **E2B's numbers against no drafter are not usable from this run**: without the cool-down the
  repeated no-drafter server decoded 0.926x the first (105.3 against 113.7 tok/s, card 71–81 °C
  against 52–63 °C). E4B's repeat matched (60.1 and 60.1).
- No reason found to serve them: the rigs keep the bf16 QAT drafter.

## Not run

- `run_q4ct.sh`, the W4A16 drafters at 1–4 tokens with cool-down; the trimmed arms above replaced it.
- **Serving changes**: none. Both rigs keep the QAT drafter at 3 tokens. 4 tokens is ahead greedy
  by less than this run's drift on E2B and behind sampled.
