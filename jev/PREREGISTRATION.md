# Pre-registration: Gemma 4 26B read by label logits vs DiffusionGemma one-step reads

Committed before any model call. Everything below is fixed for run `2026-09-23-l4-awq`; a change after the first call is reported as a deviation, with the reason.

## Question

How accurate and how well calibrated are the label probabilities of Gemma 4 26B-A4B read by next-token label logits, and of DiffusionGemma 26B-A4B read by one denoise step (vLLM PR #57250), on human-labelled public data, and how many labels does a single fitted temperature need to calibrate them?

## Models and serving

- Plain arm: `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`
- Diffusion arm: `cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4`
- Both: compressed-tensors, 4-bit weights, group size 32, symmetric, same quantizer, same layers kept in bf16 (about 17.2 GB each)
- Server: `vllm/vllm-openai:nightly-e9757321527ca1ecd514c07c1418dd2c53da3d19` (22 commits after the PR #57250 merge commit `1b3b88ec`)
- One flag set for both arms: `--attention-backend TRITON_ATTN --gpu-memory-utilization 0.92 --max-model-len 2048 --max-num-seqs 16 --max-logprobs 32 --enforce-eager --enable-prefix-caching`; diffusion adds `--diffusion-config '{"canvas_length": 64}'`
- Hardware: one AWS EC2 G6 instance, one NVIDIA L4, us-east-1; configuration in `aws/user-data.sh`

## Data

`data/*.jsonl` as committed: 300 examples each of sst2 (validation), ag_news (test), dair-ai/emotion (test) and tweet_eval irony (test), stratified with seed 20260923 by `build_eval_set.py`. Labels are the datasets' own.

## Reads

- Prompt, answer template, label tokens and `slot_distribution` from the PR's `structured_server.py`, vendored at `1b3b88ec`, identical for both arms; `run_eval.py` checks the prefix before any call
- Plain arm: one completion token, label logprobs requested with `logprob_token_ids`
- Diffusion arm: one read-only denoise step over the seeded canvas, 4 noise draws per example with the proxy's seed schedule
- Option-order check: the three choice tasks re-run with options listed in reverse order (`--variant reversed`), both arms

## Outcomes, computed by `score.py`

Primary:
1. Accuracy per task and arm
2. Expected calibration error (15 equal-width bins, top-answer confidence) per task and arm, at temperature 1
3. Calibration after fitting one temperature (log-loss grid 0.25–7.9) on N = 0, 25, 50, 100, 150 labels from a fixed half, scored on the other half (seed 20260923)

Secondary: Brier score, log loss, AUROC of confidence for correctness, share of probability on the allowed labels, the escalation rule at the proxy's 0.1 entropy threshold, spread across diffusion noise draws, share of answers that change under reversed option order, latency.

Readouts: `ar` (plain arm), `dg1` (diffusion, one draw), `dg4` (mean of four draws), `dgauto` (the proxy's automatic rule).

## Comparisons, stated in advance

- Plain against diffusion on the same items, paired, with 95% ranges from 2,000 resamples
- No Jev call is made. Published Jev figures on other items are context only and are labelled as such
- All results are published, including any where either arm does worse, with per-item outputs committed under `results/`

## Addendum, 2026-09-23, before run `2026-09-23-l4-latency`

Committed after the first run and before any call in this one. It adds to the design above and changes nothing in it.

- **Latency pass.** Both 26B arms at client concurrency 1, first 100 examples of each task (400 per arm), same image, flags and checkpoints. Plain arm: one read. Diffusion arm: four reads with the proxy's schedule, so `first_ms` is one read and `first_ms + extra_ms` is the automatic rule's path when it re-reads. The client runs on the EC2 instance itself, against `localhost:8000`, so no internet round trip is in the timing.
- **Off-label tokens.** The diffusion arm stores the five highest returned tokens per read (`--keep-top 5`), to show where probability outside the allowed labels goes.
- **Small models.** Plain arms for `google/gemma-4-E4B-it` and `google/gemma-4-E2B-it` in bf16 on the same L4 and flags, all 1,200 examples plus the reversed variant, scored with the same `score.py`. Exploratory: no hypothesis was stated for them in advance.
- **Analysis added after the first run:** the label-count calibration fit repeated over 20 random splits, reported as mean and range beside the pre-registered single split.
- **Deviation, recorded before the E4B/E2B arms ran:** the plain arm's prompt is built from the served model's own chat template followed by the answer lead. For Gemma 4 26B this is token-for-token the prompt used in both runs (`run_eval.py` checks it). Gemma 4 E4B and E2B chat templates end at the model turn with no empty thought block, so their prompt has none; the first attempt, which required the 26B prefix, refused to run them.

## Addendum, 2026-09-23, before run `2026-09-24-l4-suite`

Committed before any call in this run.

- **Data.** Bespoke Labs' 13-subset public suite, 3,880 human-labelled records, rebuilt on the instance from the public sources with Nimble's converters at commit `0e67403` and the ids in its committed manifests (`nimble_suite/build.sh`). All 13 `dataset_sha256` values must match Nimble's manifests; a mismatch is reported and that subset is excluded.
- **Arms.** Plain Gemma 4 26B (`cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`), DiffusionGemma (`cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4`, 4 reads) and Gemma 4 E4B bf16, with the image, flags and read method of the earlier runs; each record's Jev request is parsed by the PR proxy's own `jev_schema`.
- **Outcomes** (`nimble_suite/score_suite.py`), with Bespoke Labs' definitions so they sit beside its published Jev 1.13.0 and Nimble-9B figures on the same records: accuracy, ECE over 10 equal-width bins, multiclass Brier, per subset and pooled by question type; plus ECE after one temperature fitted on 50 labels, mean of 20 splits.
- **Comparison with Jev** uses Bespoke Labs' published per-subset aggregates, from one Jev run through its API. No Jev call is made here.
