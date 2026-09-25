# Pre-registration: does Gemma 4's QAT 4-bit lose accuracy on the label read?

Written 2026-09-25 at 16:45 UTC, before any scored 4-bit read: the run's VM was requested at 17:00 UTC and started reading at 17:34 UTC (`results/2026-09-25-w4a16-evidence/run.log`). It was not committed to git, uploaded, or included in the run's code bundle until after scoring, so the file's modification time is the only record of that order. The read, data, prompts, label tokens, probability code and scoring are `../jev-tpu`'s, byte-identical for that run (`run_eval.py`, `score.py`, `tasks.py`, `sweep_score.py`, `vendor/`, `data/`, `nimble_suite/`); `../jev-tpu/PREREGISTRATION.md` defines them, including its two deviations (on-demand capacity; the top 32 log-probabilities instead of explicit label ids).

## Question

Google's QAT W4A16 checkpoints (`google/gemma-4-{E2B,E4B,12B,31B}-it-qat-w4a16-ct`) serve on one TPU v6e chip once three patches are applied to `tpu_inference`. On the same records, with the same read and serving flags, how much accuracy do they give up against the bf16 checkpoints?

## Serving

- One TPU v6e chip (`ct6e-standard-1t`), provisioning by availability; zone recorded per run.
- `vllm/vllm-tpu@sha256:19a1a0526476f902eb83e1057f3d8938f35b457dd7716a30e9ab4f7bee90d507` (vLLM `0.29.1rc1.dev468+g0b7f11a1e`), the image `../jev-tpu` read the bf16 arms on, with its installed `tpu_inference` patched by three diffs applied in order and recorded with their hashes in the run bundle:
  - `kvshare.diff`: vllm-project/tpu-inference#3299 rebased onto `main` (KV-shared layers own no K/V parameters; needed by E2B and E4B QAT exports).
  - `wna16.diff`: a JAX-path compressed-tensors W4A16 linear method on the `gmm_v2` kernel.
  - `unified.diff`: `Gemma4UnifiedForConditionalGeneration` (12B) registered text-only on the JAX path.
- Serving flags as `../jev-tpu` for every arm: `--tensor-parallel-size 1 --max-model-len 2048 --max-num-seqs 16 --max-logprobs 32 --generation-config vllm --limit-mm-per-prompt '{"image":0,"audio":0}' --enable-prefix-caching`.
- Boot budget 60 minutes per arm instead of 25: the kernel path compiles about 2.7 times longer than the XLA path at startup, and 31B did not finish compiling in 25. The JAX compile cache may be pre-seeded from an earlier boot of the same image and chip type; that changes boot time only.

## Arms

| Arm | Checkpoint | Paired reference |
|---|---|---|
| e2b-w4a16 | `google/gemma-4-E2B-it-qat-w4a16-ct` | `../jev-tpu` `2026-09-24-v6e1-e2b` (bf16) |
| e4b-w4a16 | `google/gemma-4-E4B-it-qat-w4a16-ct` | `../jev-tpu` `2026-09-24-v6e1-e4b` (bf16) |
| 12b-bf16 | `google/gemma-4-12B-it` on the patched image | stack check against `../jev-tpu` `2026-09-24-v6e1-12b`, which ran on the PyTorch path |
| 12b-w4a16 | `google/gemma-4-12B-it-qat-w4a16-ct` | this run's `12b-bf16` (same stack) |
| 31b-w4a16 | `google/gemma-4-31B-it-qat-w4a16-ct` | none: bf16 31B does not fit one chip |

Run in that order. An arm that does not serve is recorded with its failure, and the run continues.

## Outcomes

- **Primary:** per arm with a reference, the paired accuracy difference (4-bit minus bf16) on the 3,880-record public suite, with a 95% bootstrap range over records, and the counts of records that flip each way (`quant_compare.py`).
- The same per task (300 examples each) and on the reversed-option variants of the three choice tasks.
- Everything `../jev-tpu` reports per arm (`sweep_score.py`, `nimble_suite/suite_sweep.py`): accuracy, ECE, Brier, ECE after 50 labels, reversed-order changes, share of reads with every label in the top 32, latency.
- The 12B stack check reports the same paired difference; a nonzero one is a property of the serving stack, and 12B's 4-bit arm is compared against this run's bf16 for that reason.

## Commitments

- Every arm is published, including arms that fail to serve or lose accuracy.
- Cost cap: 4 hours of v6e-1 time. The VM deletes itself when the run ends and carries `--max-run-duration` as a backstop.
- Deviations are recorded here with a date before the affected results are scored.

## Deviations

All recorded 2026-09-25, after the run `2026-09-25-w4a16` was scored; none changes a number already reported for it.

- **Label coverage.** The read scores a label outside the top 32 at a floor, and arms differ in how often that happens (for example E2B emotion: every label returned on 82.7% of bf16 reads, 100% of 4-bit reads). `quant_compare.py` now reports, per group, that share for each arm and the paired difference on only the records where both arms returned every label. For the four tasks and the reversed-option rows those differences move by at most 0.015 from the full-record ones (E2B, AG News, against the patched-image bf16 reference). The suite reads did not record the count; `nimble_suite/run_suite.py` now does (one added field, `labels_returned`, as `run_eval.py` records for the tasks), so it differs from `../jev-tpu`'s copy from here on.
- **Serving flag.** `--limit-mm-per-prompt` becomes `'{"image":0,"audio":0,"video":0}'` for later runs. The 12B registration now refuses to start unless every modality the checkpoint declares is listed at 0, because vLLM treats an unlisted modality as unlimited. For the text read this changes nothing: the runner already treated every arm as text-only.
- **References for E2B and E4B.** Their bf16 references were read by `../jev-tpu` on the unpatched image, while the 4-bit arms ran on the patched one, whose KV-sharing change touches the E2B and E4B model code. Its equivalence rests on identical greedy output on 8 prompts. A later run re-reads bf16 E2B and E4B on the patched image, as 12B was, and reports the same paired comparison against those.
- **Primary outcome.** Only the suite difference is primary. The per-task and reversed-option rows are secondary, and their 95% ranges are not corrected for multiple comparisons.
