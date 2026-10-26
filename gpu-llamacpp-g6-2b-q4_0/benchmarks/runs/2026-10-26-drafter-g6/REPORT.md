# 2026-10-26 — Google's QAT drafter on E2B, NVIDIA L4 (EC2 G6)

**Answer: on the L4 the exact Q4_0 QAT drafter lifts E2B decode 1.58x at 2 drafted tokens and 1.63x at 3, against
1.40x (2 tokens) on the RTX 3050 and 1.18x (1 token) on the GTX 1650 Ti.** Acceptance is the same on all three cards at
each draft length; the card decides what an accepted token is worth. GSM8K and BFCL are level with and without the
drafter, and GSM8K finishes 1.42x sooner. Drafting keeps paying at 120,000 tokens of context (1.49x at 2 tokens).

## Setup

Spot `g6.2xlarge` (`us-east-1d`, 8 vCPU, 32 GiB) launched by `gpu-pytorch-g6-2b-q4w4a16`'s `create_g6_instance`. NVIDIA
L4, driver 595.112.07 (CUDA 13.2). llama.cpp `77c4540ce`, the commit of every drafter run on the 3050 and 1650 Ti, built
with CUDA 13.1 for sm_89 (633 s). Target `gemma-4-E2B-it-q4_0-exact.gguf` (exact Q4_0 on Google's QAT grid), drafter
`gemma-4-E2B-it-qat-assistant-q4_0-exact.gguf`; SHA-256 in `models.sha256`, equal to the 3050 rig's `tpu.env`.

The 3050 runs' serving flags: `-ngl 99 -fa 1`, f16 K/V, `per_layer_token_embd` on the card, `--reasoning off`,
`--spec-type draft-mtp -ngld all`. Threads `-t 4 -tb 8` (the 3050 host ran `-t 6 -tb 12`). One llama-server per arm,
cooled to ≤45 °C or 90 s before each. `run.sh` drives everything; `driver.log` has the timeline.

## Speed, `-c 8192` (`speed.py`, `speed/summary.txt`)

The six prompts of the 3050 and 1650 Ti runs, greedy ×3 plus one sampled pass (temperature 0.8, seed 1234), 256 tokens,
prompt cache off. Geometric mean over prompts of the median of repeats.

| greedy | tok/s | x vs none | acceptance | replies equal to none |
| --- | ---: | ---: | ---: | ---: |
| no drafter | 133.4 | 1.000 | — | 18/18 |
| 1 token | 185.5 | 1.391 | 0.604 | **18/18** |
| 2 tokens | 211.0 | 1.582 | 0.497 | 3/18 |
| **3 tokens** | **217.6** | **1.631** | 0.403 | 3/18 |
| 4 tokens | 196.6 | 1.474 | 0.298 | 3/18 |
| no drafter, run last | 132.4 | 0.993 | — | 18/18 |

Sampled: 1.368 / 1.496 / 1.517 / 1.465x at 1 / 2 / 3 / 4 tokens. Per prompt at 3 tokens, code 2.39x to story 1.31x.
VRAM 2,900 MiB without the drafter, 2,980 with it. The last no-drafter arm is 0.7% under the first.

## Three cards, same drafter, same prompts

Greedy, x vs no drafter on the same card [acceptance]. 3050: `local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-drafter-gaps-3050oem`;
1650 Ti: `local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-10-14-drafter-gaps-1650ti`; both `77c4540ce`.

| drafted tokens | GTX 1650 Ti | RTX 3050 OEM | L4 |
| ---: | ---: | ---: | ---: |
| none, tok/s | 76.5 | 108.2 | 133.4 |
| 1 | **1.179x** [0.602] | 1.38x [0.609] | 1.391x [0.604] |
| 2 | 1.093x [0.469] | **1.40x** [0.475] | 1.582x [0.497] |
| 3 | 1.010x [0.378] | 1.32x [0.384] | **1.631x** [0.403] |
| 4 | — | — | 1.474x [0.298] |
| best, tok/s | 90.3 | 151.2 | 217.6 |

Acceptance at each draft length agrees across the cards to within 0.03. What differs is the cost of checking the
drafted tokens: each step the target reads its weights once and scores 1 + n tokens. The 1650 Ti has no tensor cores
and the gain falls from the first extra token; the 3050 peaks at 2 and the L4 at 3. The best setting is a property of
the card.

1 drafted token keeps greedy text identical to no drafter on the L4 (18/18), as on the 3050; on the 1650 Ti it did not
(0/18).

## Long context, `-c 131072` (`longctx.py`, `longctx/summary.txt`)

The first D tokens of this instance's NVIDIA driver README (SHA-256 in `doc.sha256`; a different driver version from
the 3050 host's, so the texts differ from that run), then a story request, a code request and a request to summarize
it; greedy, 256 tokens, prompt cache on.

| tokens ahead | none, tok/s (story) | 1 token | 2 tokens |
| ---: | ---: | ---: | ---: |
| 0 | 134.1 | 1.417x | **1.699x** |
| 8,192 | 122.7 | 1.425x | **1.560x** |
| 32,768 | 103.5 | 1.399x | **1.530x** |
| 65,536 | 86.0 | 1.371x | **1.443x** |
| 120,000 | 66.7 | 1.333x | **1.489x** |

Means over the questions at each depth. 2 tokens lead at every depth; code holds 1.82–2.06x throughout, the story
request 1.14–1.34x. On the 3050 E2B at 2 tokens fell from 1.51x to 1.21x over the same depths, and on the 1650 Ti
1 token fell to 0.89x at 120k. Peak VRAM 3,952 MiB with the drafter at 131,072 tokens.

## Answer quality (`compare.py`, `accuracy/COMPARE.md`)

`jev-tpu-v5e1/gen_eval.py`, the 3050 quality runs' flags (`-c 32768 --parallel 8`, 8 requests at once, GSM8K at 2,048
tokens), no drafter then the drafter at 2 tokens. Paired 95% ranges from `gen_compare.compare`.

| task | no drafter | drafter | difference (95% range) | wall time | speed-up |
| --- | ---: | ---: | --- | --- | ---: |
| GSM8K (1,319) | 0.8992 | 0.8984 | −0.0008 (−0.0099 to +0.0083) | 1,014 → 713 s | 1.42x |
| BFCL simple (400) | 0.9300 | 0.9250 | −0.0050 (−0.0125 to +0.0000) | 50 → 44 s | 1.15x |

Level on both. BFCL: 2 answers right only without the drafter, 0 only with it; 396 of 400 texts identical. The 3050's
E2B scored GSM8K 0.8976 / 0.9022 and BFCL 0.9250 / 0.9225 without / with the drafter.

## Against the PyTorch engine on the same card

`gpu-pytorch-g6-2b-q4w4a16/benchmarks/runs/2026-10-26-first-light-g6` decodes the same QAT int4 weights at 141.35 tok/s
(61-token prompt, 63 tokens) and is at the card's memory limit for one request. llama.cpp without the drafter decodes
these prompts at 133.4. Different prompts and lengths, so the two are level rather than ranked; the drafter takes
llama.cpp to 217.6, past that limit, because one read of the weights now yields more than one token.

## Files

`speed/`, `longctx/`, `accuracy/` (results, logs, server logs, `/metrics` before and after), `driver.log`, `run.out`,
`models.sha256`, `doc.sha256`; scripts `common.py`, `speed.py`, `longctx.py`, `run.sh`, `summarize_speed.py`,
`summarize_longctx.py`, `compare.py`.
