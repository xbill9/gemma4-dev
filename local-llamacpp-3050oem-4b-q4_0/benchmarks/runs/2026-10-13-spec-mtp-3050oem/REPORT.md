# 2026-10-13 — speculative decoding with Google's E4B drafter, RTX 3050 OEM

**Answer: 1.72x overall generation with 2 drafted tokens per step, and greedy output unchanged.**
Adopted as the serving default.

## The drafter

`google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant` (revision `27f8d204`), Google's drafter for
the QAT E4B checkpoint: `Gemma4AssistantForCausalLM`, 4 layers, hidden size 256, reading the
target's 2,560-wide hidden state and sharing its KV cache. Converted with llama.cpp's
`convert_hf_to_gguf.py --outtype q8_0` (llama.cpp `77c4540ce`): 98.7 MB, SHA-256 `92dcd636…05bf`.
Run as `--spec-type draft-mtp`. `make draft` repeats the fetch and conversion.

## Method

`spec_ab.py`, copied from the E2B sister rig with the drafter path and arms changed: one
llama-server per arm on port 8099 at `-c 8192`, GPU cooled to ≤40 °C per arm, six prompts,
greedy ×3 (max 256 tokens, prompt cache off) plus one sampled pass per prompt (temperature 0.8,
seed 1234). Arms: no draft, and draft-mtp at 2, 3 and 4 drafted tokens. Raw records
`results.jsonl`, tables `summary.txt`. Largest within-cell spread: 0.4%.

## Generation speed, greedy (median of 3; x vs no draft; fraction of drafted tokens accepted)

| prompt | no draft | 2 tokens | 3 tokens | 4 tokens |
| --- | ---: | ---: | ---: | ---: |
| code | 58.2 | **128.6 (2.21x)** 0.80 | 144.5 (2.48x) 0.79 | 147.2 (2.53x) 0.71 |
| chat | 58.3 | **95.2 (1.63x)** 0.43 | 95.8 (1.64x) 0.38 | 83.4 (1.43x) 0.26 |
| story | 58.2 | **89.5 (1.54x)** 0.37 | 82.1 (1.41x) 0.27 | 70.5 (1.21x) 0.18 |
| explain | 58.2 | **96.3 (1.66x)** 0.44 | 92.5 (1.59x) 0.35 | 91.5 (1.57x) 0.31 |
| facts | 58.3 | **107.8 (1.85x)** 0.56 | 108.8 (1.86x) 0.48 | 104.6 (1.79x) 0.40 |
| summary | 58.2 | **89.7 (1.54x)** 0.38 | 85.6 (1.47x) 0.30 | 73.8 (1.27x) 0.20 |
| **geometric mean** | 1.00 | **1.723** | 1.712 | 1.582 |

Sampled at temperature 0.8: 1.703 / 1.665 / 1.604 for 2 / 3 / 4 tokens; 2 tokens ranges from
1.37x to 2.15x.

E4B gains more than E2B (1.386x at 2 tokens) at similar acceptance rates. A likely reason,
untested: each plain E4B token costs about twice an E2B token (58.2 against 108.2 tok/s) while the
drafter is the same size for both, so drafting adds relatively less to an E4B step. Three or four tokens beat two on code only.

## Greedy output

At 2 and 3 drafted tokens, all 18 greedy replies are byte-identical to the no-draft replies (by
SHA-256 of the text). At 4 tokens all 18 differ. On E2B every drafting setting changed the greedy
text, so whether drafting changes greedy output depends on the model and the draft length here; the
cause was not investigated.

## Serving cost, at the rig's `-c 131072`

`prompt_check.sh`, ~1,068-token prompt, runs 2–4:

| | no draft | 2 tokens |
| --- | ---: | ---: |
| VRAM | 6430 MiB | **6662 MiB** |
| prompt phase | 545.3 ms mean | 563.4 ms mean (1.033x slower) |
| generation, 64 tokens | 56.1–56.7 tok/s | 105.3–108.8 tok/s |

## Against the E2B sister rig, same card and day

Geometric mean over the six greedy prompts: E2B 108.2 tok/s plain and 150.0 drafted; E4B 58.2
plain and **100.3 drafted**, which is 0.93x plain E2B and 0.67x drafted E2B. Answer quality was not
compared in this run.
