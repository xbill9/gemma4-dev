---
title: "Gemma 4 on One TPU v6e as a Jev-Style Decision Model: 12B and 26B Within 1.3 Points of Jev Overall, Behind on Multiple Choice"
published: false
description: "Gemma 4 E2B, E4B, 12B and a 26B-A4B fp8 build read by their label probabilities with vLLM on one TPU v6e chip, on 1,200 labelled examples and the 3,880-record public suite where Jev 1.13.0 has published results. Pre-registered, with accuracy, calibration after 50 labels, speed, cost, and why no 31B fits one chip today."
tags: gemma, googlecloud, machinelearning, llm
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/jev-tpu/devto-jev-tpu-v6e1-banner.afe3a2e3.jpg
---

This article provides a step by step guide to reading Gemma 4 as a Jev-style decision model with vLLM on one Google Cloud TPU v6e chip, across four sizes from E2B to 26B-A4B, and sets the results beside Jev's published figures. The measurement was pre-registered, and every per-item output is committed.

On Bespoke Labs' 3,880-record public suite, Gemma 4 12B and a 26B-A4B fp8 build land 1.1 and 1.3 points behind Jev 1.13.0 overall, with 95% ranges that include a tie; both are level on yes/no questions and 3 to 5 points behind on multiple choice. After one temperature fitted on 50 labels, both reach a median calibration error of 0.070, against Jev's 0.071 as shipped. No 31B checkpoint both fits one v6e chip and loads in vLLM on TPU today.

https://github.com/xbill9/gemma4-dev/tree/main/jev-tpu

---

#### Why Measure This?

A Jev-style decision model answers a typed question with a probability for each allowed option: end the prompt where the answer starts, read the scores of the allowed label tokens, and apply a softmax over those. TypeSafe's Jev does this as a hosted service, and any open model can be read the same way.

A [companion article](https://dev.to/gde/plain-gemma-4-26b-vs-jev-on-one-ec2-l4-21-points-behind-overall-level-on-yesno-45-behind-on-15k6) measured this read on one NVIDIA L4 GPU, where the largest Gemma 4 that fits is a 4-bit 26B-A4B. This run asks the same questions on one TPU v6e chip, with the same prompts, data, label tokens and scoring code: how each Gemma 4 size compares with Jev, and which quantized 26B-A4B and 31B checkpoints serve on one chip at all.

---

#### At This Point You Should Have…

- A Google Cloud project with v6e quota in a region that offers `ct6e-standard-1t`, and the `gcloud` CLI logged in
- A Hugging Face token stored as the Secret Manager secret `hf-token`, readable by the Compute Engine default service account
- A Cloud Storage bucket for results
- The repository cloned: `git clone https://github.com/xbill9/gemma4-dev` and `cd gemma4-dev/jev-tpu`

---

#### Step 1 — Pre-Register the Measurement

The models, image, serving flags, data, metrics and the order of the quantized attempts are written down and committed before any model call. The file is `PREREGISTRATION.md`, and each departure from it is recorded there with a date before the affected results are scored.

```shell
git show -s --oneline e0cce69 4937fb8
```

```plaintext
e0cce69 jev-tpu: sibling of jev for one TPU v6e chip — copied read, data, suite and scoring code (proxy byte-identical); pre-registration for E2B/E4B/12B bf16 and exploratory quantized 26B-A4B/31B probes
4937fb8 jev-tpu: VM boot, serve and run drivers; pre-registration amended before any model call — suite built locally at 0e67403 (13/13 checksums) and re-checked on the VM
```

---

#### Step 2 — Pick Checkpoints That Fit One Chip

One v6e chip has about 28.7 GiB of usable memory. E2B, E4B and 12B fit at bf16. The 26B-A4B and 31B do not, so the pre-registration lists quantized builds in the order to try, using only formats vLLM's TPU backend can load or that were worth one boot to check.

| Size | Checkpoint | Format | Size on disk |
|---|---|---|---|
| E2B | `google/gemma-4-E2B-it` | bf16 | |
| E4B | `google/gemma-4-E4B-it` | bf16 | |
| 12B | `google/gemma-4-12B-it` | bf16 | |
| 26B-A4B | `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic` | fp8 | 26.67 GiB |
| 31B | `google/gemma-4-31B-it-qat-w4a16-ct` | w4a16 | 21.67 GiB |
| 31B | `cyankiwi/gemma-4-31B-it-AWQ-4bit` | 4-bit | 19.47 GiB |

The 31B fp8 builds are 30.98 GiB, over the chip's usable memory, and were not tried.

---

#### Step 3 — Launch One v6e Chip

The VM runs the whole measurement from its startup script, with the code and suite fetched from Cloud Storage, and deletes itself when the run ends.

```shell
gcloud compute instances create jev-tpu-v6e1 --zone europe-west4-a \
  --machine-type ct6e-standard-1t \
  --image-family ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e --image-project ubuntu-os-accelerator-images \
  --boot-disk-size 200GB --scopes cloud-platform --maintenance-policy TERMINATE \
  --provisioning-model STANDARD --max-run-duration 6h --instance-termination-action DELETE \
  --metadata jev-code=<code-tarball>,jev-run=2026-09-24-v6e1 --metadata-from-file startup-script=tpu/startup.sh
```

```plaintext
NAME          ZONE            MACHINE_TYPE      PREEMPTIBLE  INTERNAL_IP    EXTERNAL_IP    STATUS
jev-tpu-v6e1  europe-west4-a  ct6e-standard-1t               10.164.15.207  34.32.155.166  RUNNING
```

`--scopes cloud-platform` lets the VM read the Hugging Face token from Secret Manager at boot, so the token never enters instance metadata. The run used on-demand capacity.

---

#### Step 4 — Serve Each Model

`tpu/serve.sh` starts one model at a time with the same flags for every arm and waits for `/v1/models`. Abridged, with the token and volume options left out:

```shell
docker run -d --name vllm --privileged --net=host --shm-size 10gb vllm/vllm-tpu:nightly \
  vllm serve google/gemma-4-12B-it --tensor-parallel-size 1 --max-model-len 2048 --max-num-seqs 16 \
  --max-logprobs 32 --generation-config vllm --limit-mm-per-prompt '{"image":0,"audio":0}' --enable-prefix-caching
```

The run log records each model's time to serve:

```plaintext
[jev-run 2026-09-24T14:42:48Z] READY google/gemma-4-E2B-it after 346s
[jev-run 2026-09-24T14:51:34Z] READY google/gemma-4-E4B-it after 421s
[jev-run 2026-09-24T15:02:01Z] READY google/gemma-4-12B-it after 512s
[jev-run 2026-09-24T15:14:59Z] READY RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic after 616s
```

The image was `vllm/vllm-tpu@sha256:19a1a0526476f902eb83e1057f3d8938f35b457dd7716a30e9ab4f7bee90d507`, vLLM `0.29.1rc1.dev468+g0b7f11a1e`. On v6e it stores the KV cache in fp8 by default for every model, as each boot log records:

```plaintext
INFO 09-24 14:53:37 [tpu_platform.py:232] Automatically using fp8_e5m2 for FP8 KV cache on TPU v6e.
```

So "bf16" below describes the weights. The L4 runs kept the default 16-bit KV cache.

---

#### Step 5 — Read the Labels Out of the Top 32

vLLM's TPU backend returns only the top-k log-probabilities, so a request for the log-probabilities of specific label tokens fails. The run sends four request shapes before each model's reads and keeps the answers:

```plaintext
== probe 3: {"prompt":[2,106,1645,108],"max_tokens":1,"temperature":0,"logprobs":32,"return_tokens_as_token_ids":true}
{"id":"cmpl-9ee6dc5f0461280b","object":"text_completion", ...
== probe 4: {"prompt":[2,106,1645,108],"max_tokens":1,"temperature":0,"logprobs":5,"logprob_token_ids":[236776,236799,236780],"return_tokens_as_token_ids":true}
{"error":{"message":"list index out of range","type":"InternalServerError","param":null,"code":500}}
```

So the read asks for the top 32 log-probabilities (`JEV_TOPK=32`) and takes the label tokens from them. A label inside the 32 gets its exact log-probability, the number an explicit request would return from this server. A label outside gets the proxy's fallback value, the lowest returned log-probability minus 5, and every record keeps how many labels came back.

---

#### Step 6 — Run and Score

Each model gets a five-example smoke read, the four tasks (300 examples each from sst2, AG News, DAIR Emotion and tweet_eval irony), the three choice tasks with their options reversed, a latency pass at concurrency 1, and Bespoke Labs' 3,880-record public suite, rebuilt with Nimble's converters and checked against all 13 published checksums on the VM.

```plaintext
[jev-run 2026-09-24T14:36:51Z] suite: 13 of 13 subsets match
[jev-run 2026-09-24T15:02:42Z] 12b four tasks: 23s for 1200 decisions at concurrency 8
[jev-run 2026-09-24T15:04:25Z] 12b suite: 60s for 3880 records
```

```shell
python3 sweep_score.py --prefix 2026-09-24-v6e1 --arms e2b e4b 12b 26b-fp8
python3 nimble_suite/suite_sweep.py --prefix 2026-09-24-v6e1 --arms e2b e4b 12b 26b-fp8
```

Both write their tables into `results/`, and every figure below comes from those files.

---

#### How Does Each Size Compare With Jev?

On all 3,880 records, 12B and 26B-A4B are level with Jev; E4B and E2B are behind. Level means the 95% range of the difference includes zero.

| Questions | Jev 1.13.0, published | 26B-A4B fp8 | 12B | E4B | E2B |
|---|---|---|---|---|---|
| All, 3,880 | 77.3% | 76.0% | 76.2% | 72.8% | 68.5% |
| Yes/no, 1,399 | 84.6% | 84.2% | 85.1% | 81.5% | 76.2% |
| Multiple choice, 1,848 | 82.8% | 79.7% | 78.1% | 74.5% | 70.8% |
| Five-level rating, 633 | 45.2% | 47.4% | 51.0% | 48.5% | 44.9% |

Jev leads 12B by 1.1 points overall (95% range −0.8 to +3.0) and 26B-A4B by 1.3 (−0.6 to +3.2). On multiple choice it leads 12B by 4.7 (+2.2 to +7.3) and 26B-A4B by 3.2 (+0.7 to +5.7). On ratings 12B leads Jev by 5.8 (0.4 to 11.3), on exact-level accuracy. E4B trails Jev by 4.6 overall and E2B by 8.8. Jev's per-record answers are unpublished, so these ranges compare two independent proportions, and several subsets draw many records from one passage or article, which widens the true ranges, most of all for ratings.

The L4's 4-bit 26B-A4B scored 75.3% on the same records, 2.1 points behind Jev with a range from 0.2 to 4.0. The fp8 build here scores 0.7 points more, so read both as one to two points behind Jev overall.

Each record is a Jev request converted by the PR proxy's own parser, so Gemma sees yes/no questions answered with the word yes or no, multiple-choice options lettered A, B, C, and rating levels numbered from 1. Gemma was read in that format with no prompt tuning; a different prompt could move the multiple-choice gap in either direction.

---

#### Are the Probabilities Calibrated?

As shipped, no Gemma size matches Jev. After one temperature fitted on 50 labels from each subset, 12B and 26B-A4B do.

| Median over 13 subsets | 26B-A4B fp8 | 12B | E4B | E2B |
|---|---|---|---|---|
| ECE as shipped (Jev 0.071) | 0.163 | 0.167 | 0.169 | 0.246 |
| ECE after 50 labels | 0.070 | 0.070 | 0.077 | 0.092 |
| Brier as shipped (Jev 0.267) | 0.324 | 0.331 | 0.358 | 0.487 |
| Subsets at or below Jev after 50 labels | 7 of 13 | 5 of 13 | 5 of 13 | 5 of 13 |

The fitted figures are scored on the held-out half of each subset, and Jev's are as shipped; a temperature fitted on Jev's output could lower its numbers as well.

---

#### How Does Size Show Up on the Four Tasks?

| Task | Majority | E2B | E4B | 12B | 26B-A4B fp8 |
|---|---|---|---|---|---|
| sst2 | 51.0% | 88.7% | 94.0% | 95.0% | 94.7% |
| AG News | 25.0% | 30.0% | 83.3% | 86.3% | 86.0% |
| DAIR Emotion | 35.0% | 53.3% | 54.3% | 59.3% | 58.3% |
| tweet_eval irony | 60.3% | 73.7% | 84.0% | 84.7% | 91.0% |

E2B's AG News score sits near the 25% majority rate, as it did on the L4. From E4B up, sst2 and AG News change little with size; irony moves most, 84.7% for 12B against 91.0% for 26B-A4B.

---

#### How Fast Is Each Size?

Median time per decision on the chip, one request at a time over the first 100 examples of each task:

| | E2B | E4B | 12B | 26B-A4B fp8 |
|---|---|---|---|---|
| Median | 9 ms | 12 ms | 26 to 27 ms | 27 to 33 ms |
| Decisions a second, eight in flight | about 274 | about 206 | about 101 | about 80 |

The client ran on the same VM against `localhost`, so network time is excluded. Decisions a second are eight divided by the mean request time at eight requests in flight.

---

#### 🔎 Tip: Check the Read Against a Second Platform

Serving the same checkpoint on two platforms with the same prompts is a cheap check on the whole read, from chat template to label tokens. E2B and E4B were also read on the L4, and the predicted label matched on 1,186 and 1,193 of 1,200 examples; accuracy per task differed by at most 1.0 and 0.4 points. The 26B-A4B fp8 build here and the 4-bit AWQ build on the L4 are different quantizations of the same model and differ by at most 1.3 points per task and 0.7 points on the suite.

---

#### Did the Top-32 Read Lose Anything?

Not on accuracy. At least one label came back on every four-task read, so the predicted label is exact; only the probabilities of labels outside the 32 are approximate.

Every label made the top 32 on 78.3% to 100% of four-task reads for E2B, E4B and 12B. For 26B-A4B it did on 99% of reads on the two-label tasks but on only 3.7% (AG News) and 5.0% (DAIR Emotion) of the others, because most of its probability goes to one label and to tokens that are not labels at all, which push the other labels out of the 32.

On the four tasks, moving every missing label from the fallback value to the top-32 bound, the most probability it could have had, changes raw ECE by less than 0.001 and ECE after 50 labels by at most 0.005. That check was added after the run. The suite records do not keep how many labels came back, so the suite calibration figures have no such check; by a lower-bound count, at least 10.2% of 12B's suite records and 17.8% of 26B-A4B's used the fallback value, most of them on MASSIVE's 18-option questions and, for 26B-A4B, 245 of 599 VitaminC records.

The share of probability on the allowed labels matters here as well. On irony, 12B's most likely token was a label on 5 of 300 reads, with a median of 3.1% of its probability on the two labels. 26B-A4B's was a label on 161 of 300 AG News reads, with a median of 55.2%. The read rescales the labels to sum to one, so the answer reads as confident either way; `label_mass` in every record shows it, and it is worth logging in any deployment.

---

#### What About 31B?

Both 31B builds failed 120 seconds into the boot with the same error:

```plaintext
NotImplementedError: compressed-tensors scheme for layer 'model.language_model.layers.0.self_attn.q_proj' is not yet supported in the JAX path.
```

Gemma 4 runs only on the JAX path of vLLM's TPU backend, and that path loads fp8 but not 4-bit weights. The one 31B format it loads, fp8 at 30.98 GiB, is larger than one chip's usable memory; every 31B format small enough (w4a16, AWQ, GPTQ, AutoRound, NVFP4) is either unimplemented there or implemented only on the path Gemma 4 does not use. The L4 comparison also stops at 26B-A4B, so the two runs cover the same sizes.

---

#### Compare and Contrast

| | E2B | E4B | 12B | 26B-A4B fp8 |
|---|---|---|---|---|
| Suite, all records | 68.5% | 72.8% | 🥇 76.2% | 76.0% |
| Gap to Jev, all records | 8.8 behind | 4.6 behind | level | level |
| Multiple choice | 70.8% | 74.5% | 78.1% | 🥇 79.7% |
| Five-level rating | 44.9% | 48.5% | 🥇 51.0% | 47.4% |
| Median ECE after 50 labels | 0.092 | 0.077 | 🥇 0.070 | 🥇 0.070 |
| Time per decision | 🥇 9 ms | 12 ms | 26 to 27 ms | 27 to 33 ms |
| Checkpoint | bf16 | bf16 | bf16 | fp8, 26.67 GiB |

In this table 🥇 marks the best value in each row.

---

#### So, Which One?

For a Jev-style decision service on one v6e chip, Gemma 4 12B at bf16 plus one temperature fitted on about 50 labels. It is within 1.1 points of Jev overall and level on yes/no questions, reaches a median ECE of 0.070 after fitting against Jev's 0.071 as shipped, and needs no quantized checkpoint. Check `label_mass` on your own task first: on irony 12B put a median of 3.1% of its probability on the labels.

The 26B-A4B fp8 build is level with 12B on the suite and scores higher on irony and multiple choice, but at 26.67 GiB it leaves little room for longer inputs, and it adds a third-party quantization to the stack.

Where speed matters more than the last few points, E4B at 12 ms trails Jev by 4.6 points. E2B is too weak on multi-way topic questions with this prompt.

---

#### What Does It Cost?

At about 101 decisions a second for 12B with eight requests in flight, one v6e chip costs $8.19 per million decisions at the europe-west4 on-demand price of $2.97 an hour, or $3.72 at the flex-start price of $1.35, a mode that had no capacity during this run. E2B is $3.01 and $1.37; 26B-A4B is $10.37 and $4.71.

TypeSafe prices Jev at $5.54 per million decisions at the L4 run's median prompt of 132 tokens. So one v6e chip at on-demand prices costs more per decision than Jev for 12B and 26B-A4B and less for E2B and E4B; at flex-start prices every size costs less. The chip is charged by the hour whether busy or idle, and eight requests in flight is below the 16 the server allows, so these figures hold only while the chip is kept at least this busy.

The measured run took 48 minutes of on-demand time, $2.39, and all instance time for the measurement came to at most $3.25.

---

#### Teardown

The VM deletes itself when `tpu/run.sh` finishes, and `--max-run-duration 6h` with `--instance-termination-action DELETE` is the backstop. Confirm that nothing is left:

```shell
gcloud compute instances describe jev-tpu-v6e1 --zone europe-west4-a
```

```plaintext
ERROR: (gcloud.compute.instances.describe) Could not fetch resource:
 - The resource 'projects/aisprint-491218/zones/europe-west4-a/instances/jev-tpu-v6e1' was not found
```

---

#### Summary

The goal of this article was to read four sizes of Gemma 4 as Jev-style decision models with vLLM on one TPU v6e chip and set them beside Jev's published results. The key to the solution was the same prompts, labels and scoring as the L4 run, a pre-registered design, and a label read taken from the top 32 log-probabilities. The results were:

- 🟢 Gemma 4 12B and a 26B-A4B fp8 build are level with Jev 1.13.0 over the 3,880-record suite: 76.2% and 76.0% against 77.3%
- 🟢 Both are level on yes/no questions
- ⚠️ 12B scores 51.0% on five-level ratings against Jev's 45.2%, a range that barely excludes a tie before accounting for records that share a passage
- ⚠️ Jev leads on multiple choice by 4.7 points over 12B and 3.2 over 26B-A4B
- ⚠️ As shipped, every Gemma size is less calibrated than Jev: median ECE 0.163 to 0.246 against 0.071
- 🟢 One temperature fitted on 50 labels brings 12B and 26B-A4B to 0.070
- 🟢 9 ms per decision for E2B and 26 to 27 ms for 12B on the VM, one request at a time
- 🟢 The same checkpoints on the TPU and the L4 give the same label on 1,186 and 1,193 of 1,200 examples
- ⚠️ vLLM on TPU returns only the top-k log-probabilities; on the four tasks, reading the labels out of the top 32 changes calibration by at most 0.005, and the suite calibration figures have no such check
- ⚠️ At about 101 decisions a second, 12B costs $8.19 per million decisions on demand against Jev's $5.54, and $3.72 at flex-start prices
- ❌ No 31B checkpoint both fits one v6e chip and loads in vLLM on TPU
- 🟢 The whole measurement cost at most $3.25 of instance time

Scope: one TPU v6e chip (`ct6e-standard-1t`, on demand) in europe-west4-a, vLLM `0.29.1rc1.dev468+g0b7f11a1e` at the image digest above, E2B, E4B and 12B with bf16 weights and 26B-A4B through RedHatAI's fp8 build, every model with vLLM's default fp8 KV cache on v6e, `--max-model-len 2048`, one run per model, 300 examples per task plus the 3,880-record public suite, client on the VM. The read takes labels from the top 32 log-probabilities, a dated deviation from the pre-registration; the top-32 bound check, added after the run, covers the four tasks only. On-demand capacity is also a dated deviation from the pre-registered flex-start. Every source dataset was published before Gemma 4 and may be in its training data. No Jev call was made: the Jev figures are Bespoke Labs' published results on the same records, from one run of Jev 1.13.0 by a company that sells a competing model, counting an invalid Jev response as wrong, with Jev's probabilities rounded to two decimals by its API. Parts of the analysis and writing were done with AI assistance (Claude); every figure comes from the committed output files.

The strategy for using label probabilities to run Gemma 4 on TPU as a decision model was validated with an incremental step by step approach.

---

#### References

- Code, pre-registration and per-item results: https://github.com/xbill9/gemma4-dev/tree/main/jev-tpu
- Companion L4 measurement: https://dev.to/gde/plain-gemma-4-26b-vs-jev-on-one-ec2-l4-21-points-behind-overall-level-on-yesno-45-behind-on-15k6
- Companion review of the independent evidence on Jev: https://dev.to/gde/jev-after-eight-days-of-independent-tests-level-with-mid-price-llms-behind-the-frontier-1kln
- vLLM TPU documentation: https://docs.vllm.ai/projects/tpu/en/latest/
- tpu-inference: https://github.com/vllm-project/tpu-inference
- 26B-A4B fp8 checkpoint: https://huggingface.co/RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic
- Gemma 4 12B: https://huggingface.co/google/gemma-4-12B-it
- Bespoke Labs, public-suite results for Jev 1.13.0: https://github.com/bespokelabsai/nimble/blob/0e67403/docs/PUBLIC_BENCHMARKS.md
- Cloud TPU v6e: https://cloud.google.com/tpu/docs/v6e
- Guo et al., On Calibration of Modern Neural Networks: https://arxiv.org/abs/1706.04599
