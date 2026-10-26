# 2026-10-26 — Google's QAT drafter on E4B, NVIDIA L4 (EC2 G6)

**Answer: on the L4 the exact Q4_0 QAT drafter lifts E4B decode 1.65x at 2 drafted tokens and 1.75x at 3, against
1.72x (2 tokens) on the RTX 3050 and 1.49x (1 token) on the GTX 1650 Ti.** Acceptance is the same on all three cards at
each draft length. GSM8K and BFCL are level with and without the drafter, and GSM8K finishes 1.58x sooner. Drafting
keeps paying at 120,000 tokens of context (1.57x at 2 tokens). With the drafter E4B generates at 0.97x E2B without one
on the same card.

## Setup

The instance, binary, flags and scripts of `gpu-llamacpp-g6-2b-q4_0/benchmarks/runs/2026-10-26-drafter-g6`, run right
after it on the same spot `g6.2xlarge` (`us-east-1d`): NVIDIA L4, driver 595.112.07, llama.cpp `77c4540ce` built with
CUDA 13.1 for sm_89. Target `gemma-4-E4B-it-q4_0-exact.gguf`, drafter `gemma-4-E4B-it-qat-assistant-q4_0-exact.gguf`;
SHA-256 in `models.sha256`, equal to `tpu.env`.

Serving flags: `-ngl 99 -fa 1`, f16 K/V, `per_layer_token_embd` on the card, `--reasoning off`, `--spec-type draft-mtp
-ngld all`, `-t 4 -tb 8`. One llama-server per arm, cooled to ≤45 °C or 90 s before each. `run.sh` drives everything;
`driver.log` has the timeline, `run.out` its output.

## Speed, `-c 8192` (`speed.py`, `speed/summary.txt`)

The six prompts of the 3050 and 1650 Ti runs, greedy ×3 plus one sampled pass (temperature 0.8, seed 1234), 256 tokens,
prompt cache off. Geometric mean over prompts of the median of repeats.

| greedy | tok/s | x vs none | acceptance | replies equal to none |
| --- | ---: | ---: | ---: | ---: |
| no drafter | 74.0 | 1.000 | — | 18/18 |
| 1 token | 106.8 | 1.443 | 0.620 | 3/18 |
| 2 tokens | 121.8 | 1.646 | 0.484 | 3/18 |
| **3 tokens** | **129.7** | **1.754** | 0.398 | 3/18 |
| 4 tokens | 129.7 | 1.754 | 0.345 | 0/18 |
| no drafter, run last | 73.1 | 0.988 | — | 18/18 |

3 and 4 tokens are level greedy (129.72 and 129.74 tok/s); sampled, 3 leads: 1.442 / 1.655 / **1.754** / 1.691x at 1 /
2 / 3 / 4 tokens. Per prompt at 3 tokens, code 2.55x to story 1.38x. VRAM 4,520 MiB without the drafter, 4,604 with it.
The last no-drafter arm is 1.2% under the first; the card ran 21 °C warmer by then (55 → 76 °C).

## Three cards, same drafter, same prompts

Greedy, x vs no drafter on the same card [acceptance]. 3050 and 1650 Ti: the `e4b/speed` halves of
`local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-drafter-gaps-3050oem` and
`local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-10-14-drafter-gaps-1650ti`; all three on `77c4540ce`.

| drafted tokens | GTX 1650 Ti | RTX 3050 OEM | L4 |
| ---: | ---: | ---: | ---: |
| none, tok/s | 38.0 | 58.3 | 74.0 |
| 1 | **1.488x** [0.635] | 1.542x [0.624] | 1.443x [0.620] |
| 2 | 1.436x [0.506] | **1.717x** [0.481] | 1.646x [0.484] |
| 3 | 1.342x [0.414] | 1.703x [0.398] | **1.754x** [0.398] |
| 4 | — | — | 1.754x [0.345] |
| best, tok/s | 56.5 | 100.0 | 129.7 |

Acceptance at each draft length agrees across the cards to within 0.03, as it did for E2B. The best draft length moves
the same way as E2B's: 1 token on the 1650 Ti, 2 on the 3050, 3 on the L4 (where 4 ties). Without the drafter the L4 is
1.27x the 3050 and 1.95x the 1650 Ti.

Greedy text with 1 drafted token matches no drafter on 3 of 18 replies here, against 18 of 18 for E4B on the 3050 and
the 1650 Ti, and 18 of 18 for E2B on this card.

## Long context, `-c 131072` (`longctx.py`, `longctx/summary.txt`)

The first D tokens of this instance's NVIDIA driver README (SHA-256 in `doc.sha256`, the same file as the E2B run), then
a story request, a code request and a request to summarize it; greedy, 256 tokens, prompt cache on.

| tokens ahead | none, tok/s (story) | 1 token | 2 tokens |
| ---: | ---: | ---: | ---: |
| 0 | 73.5 | 1.482x | **1.754x** |
| 8,192 | 67.2 | 1.475x | **1.739x** |
| 32,768 | 56.6 | 1.439x | **1.692x** |
| 65,536 | 46.7 | 1.424x | **1.661x** |
| 120,000 | 36.1 | 1.371x | **1.567x** |

Means over the questions at each depth. 2 tokens lead at every depth; code holds 1.93–2.13x throughout, the story
request 1.30–1.39x. On the 3050, E4B at 2 tokens fell from 1.864x to 1.552x over the same depths; E2B on this card from
1.699x to 1.489x. Peak VRAM 6,774 MiB with the drafter at 131,072 tokens (6,570 without).

## Answer quality (`compare.py`, `accuracy/COMPARE.md`)

`jev-tpu-v5e1/gen_eval.py`, the 3050 quality runs' flags (`-c 32768 --parallel 8`, 8 requests at once, GSM8K at 2,048
tokens), no drafter then the drafter at 2 tokens. Paired 95% ranges from `gen_compare.compare`.

| task | no drafter | drafter | difference (95% range) | wall time | speed-up |
| --- | ---: | ---: | --- | --- | ---: |
| GSM8K (1,319) | 0.9310 | 0.9287 | −0.0023 (−0.0099 to +0.0053) | 1,416 → 893 s | 1.58x |
| BFCL simple (400) | 0.9100 | 0.9150 | +0.0050 (+0.0000 to +0.0125) | 86 → 69 s | 1.25x |

Level on both. BFCL: 0 answers right only without the drafter, 2 only with it; 392 of 400 texts identical. The 3050's
E4B at the same flags scored GSM8K 0.9340 / 0.9325 and BFCL 0.9100 / 0.9150 without / with the drafter
(`local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-accuracy-fullctx-3050oem`).

## E4B against E2B on the same card

From this run and `gpu-llamacpp-g6-2b-q4_0/benchmarks/runs/2026-10-26-drafter-g6`, same instance, binary, prompts and
flags:

| | E2B | E4B | E4B / E2B |
| --- | ---: | ---: | ---: |
| no drafter, tok/s | 133.4 | 74.0 | 0.555 |
| best drafter, tok/s | 217.6 (3 tokens) | 129.7 (3 tokens) | 0.596 |
| drafter gain at best | 1.631x | 1.754x | |
| 2 tokens at 120,000 ahead | 1.489x | 1.567x | |
| GSM8K, no drafter / drafter | 0.8992 / 0.8984 | 0.9310 / 0.9287 | |
| BFCL simple, no drafter / drafter | 0.930 / 0.925 | 0.910 / 0.915 | |
| GSM8K wall, drafter at 2 tokens | 713 s | 893 s | |

The drafter gains more on E4B, so E4B with it decodes at 0.972x E2B without one, and finishes GSM8K in 893 s against
E2B's 1,014 s without the drafter. Paired by question (`accuracy/e4b-vs-e2b.txt`), E4B is ahead on GSM8K by +0.0318
(+0.0167 to +0.0470) without the drafter and +0.0303 (+0.0159 to +0.0455) with it; BFCL is level (−0.0200, −0.0475 to
+0.0075, and −0.0100, −0.0375 to +0.0175).

## Files

`speed/`, `longctx/`, `accuracy/` (results, logs, server logs, `/metrics` before and after, `e4b-vs-e2b.txt`), `driver.log`, `run.out`,
`models.sha256`, `doc.sha256`; scripts `common.py`, `speed.py`, `longctx.py`, `run.sh`, `summarize_speed.py`,
`summarize_longctx.py`, `compare.py`.
