# 2026-10-13 — speculative decoding with Google's E2B drafter, RTX 3050 OEM

**Question:** for a single user, does speculative decoding with a draft model raise generation
speed?

**Answer: yes, 1.39x overall, with 2 drafted tokens per step.** Adopted as the serving default.

## The drafter

`google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant` (revision `ebc7e1a2`), Google's drafter
for the QAT E2B checkpoint this rig serves: `Gemma4AssistantForCausalLM`, 4 layers, hidden size
256, sharing the target's KV cache, same 262,144-token vocabulary. Converted here with
llama.cpp's `convert_hf_to_gguf.py --outtype q8_0` (llama.cpp `77c4540ce`): 49 tensors, 97.8 MB,
SHA-256 `1760d577…a261`. llama.cpp runs it as `--spec-type draft-mtp`
(`src/models/gemma4-assistant.cpp`). `make draft` repeats the fetch and conversion.

## Method

`spec_ab.py`: one llama-server per arm on port 8099, serving flags (`-c 8192` for the A/B),
GPU cooled to ≤40 °C before each arm. Six prompts (code, chat, story, explain, facts, summary),
greedy, max 256 tokens, prompt cache off, 3 repeats; then one sampled pass per prompt
(temperature 0.8, seed 1234). Arms: no draft, and draft-mtp at `--spec-draft-n-max` 2, 3, 4, 6.
Raw records in `results.jsonl`, tables in `summary.txt`. Largest within-cell spread across the
3 greedy repeats: 0.7%.

## Generation speed, greedy (median of 3; x vs no draft; fraction of drafted tokens accepted)

| prompt | no draft | 2 tokens | 3 tokens | 4 tokens | 6 tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| code | 108.2 | **199.9 (1.85x)** 0.85 | 209.0 (1.93x) 0.78 | 198.5 (1.84x) 0.64 | 170.5 (1.58x) 0.50 |
| chat | 108.1 | **140.1 (1.30x)** 0.41 | 127.1 (1.18x) 0.31 | 129.6 (1.20x) 0.30 | 106.0 (0.98x) 0.22 |
| story | 108.1 | **129.3 (1.20x)** 0.33 | 116.5 (1.08x) 0.25 | 103.8 (0.96x) 0.19 | 82.0 (0.76x) 0.13 |
| explain | 108.1 | **151.3 (1.40x)** 0.49 | 142.4 (1.32x) 0.39 | 131.6 (1.22x) 0.31 | 105.0 (0.97x) 0.22 |
| facts | 108.2 | **160.2 (1.48x)** 0.55 | 154.2 (1.43x) 0.46 | 145.6 (1.35x) 0.38 | 118.6 (1.10x) 0.27 |
| summary | 108.5 | **129.8 (1.20x)** 0.34 | 120.8 (1.11x) 0.27 | 106.3 (0.98x) 0.20 | 81.9 (0.76x) 0.13 |
| **geometric mean** | 1.00 | **1.386** | 1.313 | 1.226 | 0.990 |

Sampled at temperature 0.8: geometric mean 1.358 / 1.265 / 1.179 / 0.954 for 2 / 3 / 4 / 6
tokens; 2 tokens ranges from 1.16x (summary) to 1.74x (code).

Drafting more tokens per step only pays where the drafter is usually right. Three tokens beats two
on code (1.93x against 1.85x) and loses everywhere else, and six is a net loss overall.

## Greedy replies are not byte-identical

All 18 greedy replies from every drafting arm differ from the no-draft arm. `diverge.py` captured
the texts (`texts.json`, `diverge.txt`): each arm repeats itself exactly, and the two part at a
near-tie word partway in, at character 43 to 423. Examples: "but it's completely normal" against
"but it's **also** completely normal"; "sweep of the light" against "sweep of light"; one more
doctest in the code reply. The replies after the split are equally fluent; quality was not scored.
The 1650 Ti rig found the same with n-gram drafting and attributes it to batched verification
running different kernels. That cause is not tested here.

## Serving cost, at the rig's `-c 131072`

`prompt_check.sh`, ~1,068-token prompt, 4 requests per arm, runs 2–4:

| | no draft | 2 tokens |
| --- | ---: | ---: |
| VRAM (`nvidia-smi`) | 3610 MiB | **3840 MiB** |
| prompt phase | 310.8–310.9 ms | 322.1–322.4 ms (1.036x slower) |
| generation, 64 tokens | 105.4–105.5 tok/s | 163.6–169.7 tok/s |

## Live check

After the switch, through `http://192.168.0.156:8080`: 161.0 tok/s on a coding prompt (158 of 281
drafted tokens accepted) and 152.7 on a travel question (149 of 300).

## Not measured

Drafting with more than one slot (`--parallel` > 1); the concurrency run predates it. A draft
converted at bf16 or f16 rather than q8_0. Answer quality with and without drafting.
