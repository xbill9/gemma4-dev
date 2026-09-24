# Results: Jev-style label reads of Gemma 4 on one TPU v6e chip

Run `2026-09-24-v6e1`: one on-demand `ct6e-standard-1t` in europe-west4-a, `vllm/vllm-tpu@sha256:19a1a0526476f902eb83e1057f3d8938f35b457dd7716a30e9ab4f7bee90d507` (vLLM `0.29.1rc1.dev468+g0b7f11a1e`), the flags in `PREREGISTRATION.md`, client on the VM. Every figure below comes from `results/2026-09-24-v6e1-SWEEP.md` (`sweep_score.py`) and `results/2026-09-24-v6e1-SUITE.md` (`nimble_suite/suite_sweep.py`); logs, raw request probes, suite checksums and cost are in `results/2026-09-24-v6e1-evidence/`.

## What served

| Size | Checkpoint | Served on one v6e chip | Time to serve |
|---|---|---|---|
| E2B | `google/gemma-4-E2B-it`, bf16 | yes | 346 s |
| E4B | `google/gemma-4-E4B-it`, bf16 | yes | 421 s |
| 12B | `google/gemma-4-12B-it`, bf16 | yes | 512 s |
| 26B-A4B | `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic`, fp8 (26.67 GiB) | **yes** | 616 s |
| 31B | `google/gemma-4-31B-it-qat-w4a16-ct`, w4a16 | no: `NotImplementedError: compressed-tensors scheme for layer 'model.language_model.layers.0.self_attn.q_proj' is not yet supported in the JAX path` | failed at 120 s |
| 31B | `cyankiwi/gemma-4-31B-it-AWQ-4bit` (stored as compressed-tensors w4a16) | no: the same error | failed at 120 s |

The 26B AWQ fallback was not tried, because the fp8 checkpoint served. The 31B fp8 builds (30.98 GiB) exceed the chip's usable HBM and were not tried.

## Accuracy on the public suite, beside Jev 1.13.0

Bespoke Labs' 3,880 records, all 13 subsets matching Nimble's checksums on the VM. Jev figures are Bespoke Labs' published ones; ranges on Jev minus a Gemma arm are unpaired.

| Questions | Jev 1.13.0 | 26B-A4B fp8 | 12B | E4B | E2B |
|---|---|---|---|---|---|
| All 3,880 | 77.3% | 76.0% | 76.2% | 72.8% | 68.5% |
| Yes/no, 1,399 | 84.6% | 84.2% | 85.1% | 81.5% | 76.2% |
| Multiple choice, 1,848 | 82.8% | 79.7% | 78.1% | 74.5% | 70.8% |
| Five-level rating, 633 | 45.2% | 47.4% | 51.0% | 48.5% | 44.9% |
| Jev minus arm, all (95% range) | | +1.3 (−0.6 to +3.2) | +1.1 (−0.8 to +3.0) | +4.6 (+2.6 to +6.5) | +8.8 (+6.8 to +10.8) |

Calibration, median over the 13 subsets (10 bins; Jev as shipped 0.071):

| | 26B-A4B fp8 | 12B | E4B | E2B |
|---|---|---|---|---|
| ECE as shipped | 0.163 | 0.167 | 0.169 | 0.246 |
| ECE after 50 labels | 0.070 | 0.070 | 0.077 | 0.092 |
| Brier as shipped (Jev 0.267) | 0.324 | 0.331 | 0.358 | 0.487 |
| Subsets at or below Jev after 50 labels | 7 of 13 | 5 of 13 | 5 of 13 | 5 of 13 |

## Four tasks, 300 examples each

| | sst2 | AG News | DAIR Emotion | irony |
|---|---|---|---|---|
| E2B | 88.7% | 30.0% | 53.3% | 73.7% |
| E4B | 94.0% | 83.3% | 54.3% | 84.0% |
| 12B | 95.0% | 86.3% | 59.3% | 84.7% |
| 26B-A4B fp8 | 94.7% | 86.0% | 58.3% | 91.0% |

Median time per decision on the chip, one request at a time: E2B 9 ms, E4B 12 ms, 12B 26 to 27 ms, 26B-A4B fp8 27 to 33 ms. At concurrency 8 the 1,200 decisions took 15, 18, 23 and 26 seconds.

## Checks

- **Same read as the L4.** E2B and E4B were read on the L4 in `../jev` with the same checkpoints and prompts. The predicted label matched on 292 to 300 of 300 examples per task. The 26B fp8 here and the 26B AWQ 4-bit on the L4 are different quantizations of the same model and land within 1.3 points on every task and 0.7 points on the suite.
- **Top 32 instead of explicit label ids (deviation 2).** Every label made the top 32 on 78% to 100% of reads for E2B, E4B and 12B, and on 99% for the 26B on the two-label tasks but only 3.7% (AG News) and 5.0% (DAIR Emotion) on the others. Raising every missing label from the floor to the top-32 bound, the most probability it could have had, leaves raw ECE unchanged to three decimals and moves ECE after 50 labels by at most 0.005. That analysis was added after the run.
- **Request shapes.** Text prompts, token-id prompts and `logprobs` of 5 or 32 all answered; only a request carrying `logprob_token_ids` came back HTTP 500 `list index out of range` (`*.raw-probe.txt`).

## Cost

At most $3.25 of instance time: 0.81 h on demand for the measured run, 0.17 h for the first attempt that stopped at the E2B smoke read, and 0.12 h of a spot VM preempted before it served (`cost.txt`).
