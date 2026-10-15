# 2026-10-14 — speculative decoding with the exact Q4_0 drafters, GTX 1650 Ti Max-Q

**Answer: the drafter speeds both models up on this card, by less than on the RTX 3050, and the
best draft length is shorter.** E2B: 1.18x with 1 drafted token (79.0 → 93.2 tok/s), 1.11x with 2,
1.02x with 3. E4B: 1.43x with 1 or 2 (38.5 → 55.1 tok/s), 1.32x with 3.

## Setup

`run.sh` on the laptop `yogagpu` (i7-10750H, GTX 1650 Ti Max-Q, 4096 MiB, driver 615.71.09),
llama-server build 318 (`f95b0d9`), launched over SSH from the 3050 host.

- **Drafters** fetched by `fetch_drafters.sh` from
  `xbill9/gemma-4-E{2,4}B-it-qat-assistant-q4_0-exact-gguf` (revisions `6188294`, `a946809`),
  SHA-256 checked (`fetch.log`). Built and measured on the 3050 in
  `local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-drafter-q4exact-3050oem/`.
- **Targets and flags** from each rig's `tpu.env`: E2B `local-llamacpp-1650ti-2b-q4_0`, E4B
  `local-llamacpp-1650ti-4b-q4_0`, exact Q4_0 GGUFs, `-ngl 99 -ctk f16 -ctv f16 -fa 1 -t 6 -tb 12
  --reasoning off`, `-c 8192 --parallel 1`; `per_layer_token_embd` stays on the host (no `-ot`).
- **`spec_ab.py`**: the 3050 run's six prompts, greedy ×3 plus one sampled pass (temperature 0.8,
  seed 1234), 256 tokens, prompt cache off; arms no drafter, 1 / 2 / 3 drafted tokens, and no drafter
  again last. Card cooled to ≤50 °C before each arm.

## Results (`e2b/summary.txt`, `e4b/summary.txt`)

| greedy, geometric mean of 6 prompts | E2B tok/s | x | acceptance | E4B tok/s | x | acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no drafter | 79.0 | 1.00 | — | 38.5 | 1.00 | — |
| 1 drafted token | **93.2** | **1.180** | 0.619 | **55.1** | **1.432** | 0.594 |
| 2 drafted tokens | 87.6 | 1.109 | 0.495 | 55.1 | 1.430 | 0.504 |
| 3 drafted tokens | 80.5 | 1.019 | 0.394 | 50.9 | 1.321 | 0.404 |
| no drafter, repeated last | 79.0 | 1.000 | — | 38.8 | 1.009 | — |

Sampled (temperature 0.8): E2B 1.143 / 1.055 / 0.953x and E4B 1.449 / 1.406 / 1.271x for 1 / 2 / 3
tokens.

- **Per prompt, code gains most and stories least.** E2B at 1 token ranges 1.04x (story) to 1.38x
  (code); at 3 tokens code still gains 1.57x while story drops to 0.81x. E4B at 2 tokens ranges
  1.23x to 1.81x.
- **Acceptance matches the 3050's** at the same draft length (2 tokens: E2B 0.495 here, 0.475 there;
  E4B 0.504 and 0.481), so the smaller gain comes from the card: each drafted step costs a larger
  share of a target step on this GPU. That reading is inferred, not measured.
- **VRAM at `-c 8192`**: E2B 1,485 → 1,557 MiB with the drafter, E4B 2,853 → 2,927 MiB (+72–74 MiB).
- **Drift**: the repeated no-drafter arm is within 1% of the first. The card ran 43–63 °C, SM clock
  1,575–1,875 MHz. Within-cell spread is under 2% on E2B; E4B's drafted arms reach 5.9–12.1%.

## Greedy text

- **E2B** is deterministic here: each arm's three greedy repeats are identical, and the repeated
  no-drafter arm matches the first on all 18 replies. Every drafted arm differs from no drafter on
  all 18, as on the 3050.
- **E4B is not deterministic on this card, with or without the drafter.** At temperature 0 the three
  repeats of a prompt give 2–3 different texts in every arm, including both no-drafter arms. So E4B's
  greedy text cannot be compared across arms here. On the 3050 the same E4B file repeated exactly.
  The cause was not investigated.

## Against the 3050 (indicative: different card, host and llama.cpp build)

| | 1650 Ti best | 3050 at 2 tokens |
| --- | ---: | ---: |
| E2B | 1.18x (1 token) | 1.40x |
| E4B | 1.43x (1 or 2 tokens) | 1.72x |

Both 1650 Ti rigs serve with the exact Q4_0 drafter since `5680c97`: E2B at 1 drafted token, E4B at 2.
Q8_0 and bf16 drafters on this card: `2026-10-14-drafter-formats-1650ti`.
