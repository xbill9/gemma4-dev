---
title: "Running a Jev-Style Decision Model on One TPU v6e: What Fits, What It Costs, and What Changes From a GPU"
published: false
description: "Gemma 4 E2B, E4B, 12B and a 26B-A4B fp8 build read by their label probabilities with vLLM on one TPU v6e chip, checked against the same read on an NVIDIA L4 and against Jev 1.13.0's published results. What fits one chip, how to read labels when vLLM on TPU returns only the top 32 log-probabilities, speed, cost, and why no 31B loads today."
tags: gemma, googlecloud, machinelearning, llm
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/jev-tpu/devto-jev-tpu-v6e1-banner.afe3a2e3.jpg
---

This article provides a step by step guide to running a Jev-style decision model on one Google Cloud TPU v6e chip with Gemma 4 and vLLM, and compares it with the same read on one NVIDIA L4 GPU. The measurement was pre-registered, and every per-item output is committed.

One v6e chip serves Gemma 4 E2B, E4B and 12B at bf16 and a 26B-A4B fp8 build; no 31B checkpoint loads. The same checkpoints give the same answers on the TPU and the L4, and the 26B decides in 27 to 33 ms against 61 ms on the L4. On demand, the TPU costs more per decision than the L4 or Jev; 12B is the size to pick.

https://github.com/xbill9/gemma4-dev/tree/main/jev-tpu

---

#### Why Measure This?

A Jev-style decision model answers a typed question with a probability for each allowed option: end the prompt where the answer starts, read the scores of the allowed label tokens, and apply a softmax over those. TypeSafe's Jev does this as a hosted service, and any open model can be read the same way.

A [companion article](https://dev.to/gde/plain-gemma-4-26b-vs-jev-on-one-ec2-l4-21-points-behind-overall-level-on-yesno-45-behind-on-15k6) measured this read on one NVIDIA L4 and set it beside Jev's published results. This one asks what changes on a TPU: which Gemma 4 sizes fit one v6e chip, whether the read works the same way, and what the chip buys in speed and cost.

---

#### At This Point You Should Have…

- A Google Cloud project with v6e quota in a region that offers `ct6e-standard-1t`, and the `gcloud` CLI logged in
- A Hugging Face token stored as the Secret Manager secret `hf-token`, readable by the Compute Engine default service account
- A Cloud Storage bucket for results, set as `BUCKET` in `tpu/run.sh` and `tpu/startup.sh`
- The repository cloned: `git clone https://github.com/xbill9/gemma4-dev` and `cd gemma4-dev/jev-tpu`

---

#### Step 1 — Pre-Register the Measurement

The models, image, serving flags, data, metrics and the order of the quantized attempts are written down and committed before any model call, in `PREREGISTRATION.md`; each departure from it is recorded there with a date before the affected results are scored.

```shell
git show -s --oneline e0cce69 4937fb8
```

```plaintext
e0cce69 jev-tpu: sibling of jev for one TPU v6e chip — copied read, data, suite and scoring code (proxy byte-identical); pre-registration for E2B/E4B/12B bf16 and exploratory quantized 26B-A4B/31B probes
4937fb8 jev-tpu: VM boot, serve and run drivers; pre-registration amended before any model call — suite built locally at 0e67403 (13/13 checksums) and re-checked on the VM
```

---

#### Step 2 — Pick Checkpoints That Fit One Chip

One v6e chip has 31.24 GiB of memory, of which vLLM uses up to 28.74 GiB:

```plaintext
Memory statistics | total_hbm_limit_gb=31.24GiB | total_hbm_limit_cap_gb=28.74GiB | total_hbm_used_gb=24.56GiB | total_hbm_avail_gb=4.19GiB
```

E2B, E4B and 12B fit at bf16. The 26B-A4B and 31B do not, so the pre-registration lists quantized builds to try in order:

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

The VM runs the whole measurement from its startup script, fetching the code and the suite from Cloud Storage, and deletes itself when the run ends.

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

`--scopes cloud-platform` lets the VM read the Hugging Face token from Secret Manager at boot, so the token never enters instance metadata.

---

#### Step 4 — Serve Each Model

`tpu/serve.sh` starts one model at a time with the same flags for every arm and waits for `/v1/models`. Abridged, with the token and volume options left out:

```shell
docker run -d --name vllm --privileged --net=host --shm-size 10gb vllm/vllm-tpu:nightly \
  vllm serve google/gemma-4-12B-it --tensor-parallel-size 1 --max-model-len 2048 --max-num-seqs 16 \
  --max-logprobs 32 --generation-config vllm --limit-mm-per-prompt '{"image":0,"audio":0}' --enable-prefix-caching
```

```plaintext
[jev-run 2026-09-24T14:42:48Z] READY google/gemma-4-E2B-it after 346s
[jev-run 2026-09-24T14:51:34Z] READY google/gemma-4-E4B-it after 421s
[jev-run 2026-09-24T15:02:01Z] READY google/gemma-4-12B-it after 512s
[jev-run 2026-09-24T15:14:59Z] READY RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic after 616s
```

The image was `vllm/vllm-tpu@sha256:19a1a0526476f902eb83e1057f3d8938f35b457dd7716a30e9ab4f7bee90d507`, vLLM `0.29.1rc1.dev468+g0b7f11a1e`. On v6e it stores the KV cache in fp8 by default for every model:

```plaintext
INFO 09-24 14:53:37 [tpu_platform.py:232] Automatically using fp8_e5m2 for FP8 KV cache on TPU v6e.
```

So "bf16" below describes the weights. The L4 run kept the default 16-bit KV cache.

---

#### Step 5 — Read the Labels Out of the Top 32

On a GPU, the read asks vLLM for the log-probabilities of the label tokens by id. vLLM's TPU backend returns only the top-k log-probabilities, and a request for specific token ids fails. Four request shapes are sent before each model's reads, and the answers are kept:

```plaintext
== probe 3: {"prompt":[2,106,1645,108],"max_tokens":1,"temperature":0,"logprobs":32,"return_tokens_as_token_ids":true}
{"id":"cmpl-9ee6dc5f0461280b","object":"text_completion", ...
== probe 4: {"prompt":[2,106,1645,108],"max_tokens":1,"temperature":0,"logprobs":5,"logprob_token_ids":[236776,236799,236780],"return_tokens_as_token_ids":true}
{"error":{"message":"list index out of range","type":"InternalServerError","param":null,"code":500}}
```

So the read asks for the top 32 log-probabilities (`JEV_TOPK=32`) and takes the label tokens from them. A label inside the 32 gets its exact log-probability; a label outside gets the proxy's fallback value, the lowest returned log-probability minus 5, and every four-task record keeps how many labels came back.

---

#### Step 6 — Run and Score

Each model gets a five-example smoke read, the four tasks (300 examples each from sst2, AG News, DAIR Emotion and tweet_eval irony), the three choice tasks with their options reversed, a latency pass at one request at a time, and Bespoke Labs' 3,880-record public suite, rebuilt with Nimble's converters and checked against all 13 published checksums on the VM.

```plaintext
[jev-run 2026-09-24T14:36:51Z] suite: 13 of 13 subsets match
[jev-run 2026-09-24T15:02:42Z] 12b four tasks: 23s for 1200 decisions at concurrency 8
[jev-run 2026-09-24T15:04:25Z] 12b suite: 60s for 3880 records
```

```shell
python3 sweep_score.py --prefix 2026-09-24-v6e1 --arms e2b e4b 12b 26b-fp8
python3 nimble_suite/suite_sweep.py --prefix 2026-09-24-v6e1 --arms e2b e4b 12b 26b-fp8
```

Both write their tables into `results/`, and every figure below comes from those files and `article_figures.py`.

---

#### What Fits and Loads on One v6e Chip?

| Size | Checkpoint | Serves | Time to serve |
|---|---|---|---|
| E2B | bf16 | yes | 346 s |
| E4B | bf16 | yes | 421 s |
| 12B | bf16 | yes | 512 s |
| 26B-A4B | fp8, 26.67 GiB | yes | 616 s |
| 31B | w4a16, 21.67 GiB | no | fails at 120 s |
| 31B | 4-bit, 19.47 GiB | no | fails at 120 s |

Both 31B builds stop with the same error:

```plaintext
NotImplementedError: compressed-tensors scheme for layer 'model.language_model.layers.0.self_attn.q_proj' is not yet supported in the JAX path.
```

Gemma 4 runs only on the JAX path of vLLM's TPU backend, which loads fp8 but no 4-bit format for a dense model. The one 31B format it loads is 30.98 GiB, over one chip's usable memory.

---

#### Does the TPU Change the Answers?

No. E2B and E4B were also read on the L4, with the same checkpoints and prompts. The predicted label matched on 1,186 and 1,193 of 1,200 examples, and accuracy per task differed by at most 1.0 and 0.4 points. The 26B-A4B fp8 build here and the 4-bit AWQ build on the L4 are different quantizations of the same model and differ by at most 1.3 points per task and 0.7 points on the suite.

| Task | Majority | E2B | E4B | 12B | 26B-A4B fp8 |
|---|---|---|---|---|---|
| sst2 | 51.0% | 88.7% | 94.0% | 95.0% | 94.7% |
| AG News | 25.0% | 30.0% | 83.3% | 86.3% | 86.0% |
| DAIR Emotion | 35.0% | 53.3% | 54.3% | 59.3% | 58.3% |
| tweet_eval irony | 60.3% | 73.7% | 84.0% | 84.7% | 91.0% |

Per-task calibration, option-order changes and 90th-percentile times are in `results/2026-09-24-v6e1-SWEEP.md`.

---

#### How Do the Sizes Compare With Jev?

The Jev comparison is the [L4 article's](https://dev.to/gde/plain-gemma-4-26b-vs-jev-on-one-ec2-l4-21-points-behind-overall-level-on-yesno-45-behind-on-15k6) subject, with its caveats in full. On the same 3,880 public records:

| Questions | Jev 1.13.0, published | 26B-A4B fp8 | 12B | E4B | E2B |
|---|---|---|---|---|---|
| All, 3,880 | 77.3% | 76.0% | 76.2% | 72.8% | 68.5% |
| Yes/no, 1,399 | 84.6% | 84.2% | 85.1% | 81.5% | 76.2% |
| Multiple choice, 1,848 | 82.8% | 79.7% | 78.1% | 74.5% | 70.8% |
| Five-level rating, 633 | 45.2% | 47.4% | 51.0% | 48.5% | 44.9% |
| Median ECE after 50 labels (Jev 0.071 as shipped) | | 0.070 | 0.070 | 0.077 | 0.092 |

Jev leads 12B by 1.1 points overall (95% range −0.8 to +3.0) and 26B-A4B by 1.3 (−0.6 to +3.2), and on multiple choice by 4.7 and 3.2. The L4's 4-bit 26B scored 75.3%, 2.1 points behind with a range from 0.2 to 4.0, so read the 26B as one to two points behind Jev on either platform. Jev's per-record answers are unpublished, so these ranges compare two independent proportions, and records that share a passage widen them. Gemma was read in the PR proxy's prompt format with no tuning; a different prompt could move the multiple-choice gap in either direction.

---

#### Did the Top-32 Read Lose Anything?

Not on accuracy. At least one label came back on every four-task read, so the predicted label is exact; only the probabilities of labels outside the 32 are approximate.

On the four tasks, moving every missing label from the fallback value to the top-32 bound, the most probability it could have had, changes raw ECE by less than 0.001 and ECE after 50 labels by at most 0.005. That check was added after the run. The suite records do not keep how many labels came back, so the suite calibration figures have no such check; by a lower-bound count, at least 10.2% of 12B's suite records and 17.8% of 26B-A4B's used the fallback value.

The share of probability on the allowed labels varies too. On irony, 12B's most likely token was a label on 5 of 300 reads, with a median of 3.1% of its probability on the two labels; 26B-A4B's was a label on 161 of 300 AG News reads, with a median of 55.2%. The read rescales the labels to sum to one, so the answer reads as confident either way, and `label_mass` in every record shows it.

---

#### TPU v6e Against the L4

| | One TPU v6e chip | One NVIDIA L4 (`g6.xlarge`) |
|---|---|---|
| Memory for the model | 🥇 28.74 GiB | 23034 MiB |
| Largest Gemma 4 read here | 12B at bf16, 26B-A4B at fp8 | 26B-A4B at 4-bit |
| Accuracy, same checkpoint | same answers on 1,186 to 1,193 of 1,200 | same |
| 26B time per decision, one at a time | 🥇 27 to 33 ms | 61 ms |
| Price per hour, on demand | $2.97 | 🥇 $0.8048 |
| 26B cost per million decisions, on demand | $10.37 | 🥇 at most $5.43 |
| Label read | top 32 log-probabilities | 🥇 exact label token ids |
| KV cache | fp8 by default | 🥇 16-bit |

In this table 🥇 marks the better value. The 26B decides 1.8 to 2.2 times faster on the TPU. The chip costs 3.7 times as much per hour on demand, or 1.7 times at the flex-start price of $1.35, a mode that had no capacity during this run. The L4's cost comes from its first run's throughput with the client on a home connection, so it is an upper bound; the TPU's comes from the request times at eight in flight.

---

#### 🔎 Tip: Serve the Same Checkpoint on Both Platforms First

Before comparing hardware, serve one checkpoint on both with the same prompts and compare predicted labels record by record. It checks the whole read, from chat template to label tokens; here the TPU's top-32 read and the L4's exact read agreed on 1,186 and 1,193 of 1,200 answers.

---

#### So, Which One?

For a Jev-style decision service on a TPU, Gemma 4 12B at bf16 plus one temperature fitted on about 50 labels. It matches the 26B on the suite, 76.2% against 76.0%, though the 26B scores 6.3 points higher on irony; it reaches a median ECE of 0.070 after fitting, decides in 26 to 27 ms, and needs no third-party quantized checkpoint. Check `label_mass` on your own task first: on irony, 12B put a median of 3.1% of its probability on the labels.

Choose the TPU when time per decision matters or the rest of the stack is already on Google Cloud: it halves the 26B's time per decision against the L4. Choose the L4, or Jev itself, when cost per decision matters: on demand, 12B on one v6e chip costs $8.19 per million decisions, against Jev's $5.54 and at most $5.43 for the 26B on an L4.

---

#### What Does It Cost?

Measured from the request times at eight in flight, one v6e chip answers about 274 decisions a second with E2B, 206 with E4B, 101 with 12B and 80 with 26B-A4B. At the europe-west4 on-demand price of $2.97 an hour that is $3.01, $4.00, $8.19 and $10.37 per million decisions; at the flex-start price of $1.35, $1.37, $1.82, $3.72 and $4.71. TypeSafe prices Jev at $5.54 per million decisions at the L4 run's median prompt of 132 tokens. The chip is charged by the hour whether busy or idle, and eight requests in flight is below the 16 the server allows, so these figures hold only while the chip is kept at least this busy.

The measured run took 48 minutes of on-demand time, $2.39, and all instance time for the measurement came to at most $3.25.

---

#### Teardown

The VM deletes itself when `tpu/run.sh` finishes, and `--max-run-duration 6h` with `--instance-termination-action DELETE` is the backstop. Confirm it is gone:

```shell
gcloud compute instances describe jev-tpu-v6e1 --zone europe-west4-a
```

```plaintext
ERROR: (gcloud.compute.instances.describe) Could not fetch resource:
 - The resource 'projects/aisprint-491218/zones/europe-west4-a/instances/jev-tpu-v6e1' was not found
```

---

#### Summary

The goal of this article was to run a Jev-style decision model on one TPU v6e chip with Gemma 4 and find what changes from a GPU. The key to the solution was the same prompts, labels and scoring as the L4 run, a pre-registered design, and a label read taken from the top 32 log-probabilities. The results were:

- 🟢 One v6e chip serves Gemma 4 E2B, E4B and 12B at bf16 and a 26B-A4B fp8 build
- ❌ No 31B checkpoint loads: the w4a16 and 4-bit builds fail on the JAX path, and the fp8 build is 30.98 GiB
- 🟢 The same checkpoints give the same answers on the TPU and the L4: 1,186 and 1,193 of 1,200
- 🟢 The 26B decides in 27 to 33 ms on the TPU against 61 ms on the L4
- ⚠️ On demand, 26B costs $10.37 per million decisions on the TPU against at most $5.43 on the L4 and $5.54 for Jev
- ⚠️ vLLM on TPU returns only the top-k log-probabilities; reading labels from the top 32 changes no answer, and changes four-task calibration by at most 0.005
- 🟢 12B matches the 26B: 76.2% and 76.0% on the 3,880-record suite, against Jev's 77.3%
- 🟢 After 50 labels, 12B and 26B-A4B reach a median ECE of 0.070, against Jev's 0.071 as shipped
- 🟢 The whole measurement cost at most $3.25 of instance time

Scope: one TPU v6e chip (`ct6e-standard-1t`, on demand) in europe-west4-a, vLLM `0.29.1rc1.dev468+g0b7f11a1e` at the image digest above, E2B, E4B and 12B with bf16 weights and 26B-A4B through RedHatAI's fp8 build, every model with vLLM's default fp8 KV cache on v6e, `--max-model-len 2048`, one run per model, 300 examples per task plus the 3,880-record public suite, client on the VM. The top-32 read and on-demand capacity are dated deviations from the pre-registration, and the top-32 bound check, added after the run, covers the four tasks only. L4 figures come from the companion run on a `g6.xlarge` and `g6.4xlarge` in us-east-1, and the two platforms were timed with different clients. Every source dataset was published before Gemma 4 and may be in its training data. No Jev call was made: the Jev figures are Bespoke Labs' published results on the same records, from one run of Jev 1.13.0 by a company that sells a competing model, counting an invalid Jev response as wrong, with Jev's probabilities rounded to two decimals by its API. Parts of the analysis and writing were done with AI assistance (Claude); every figure comes from the committed output files.

The strategy for running a Jev-style decision model on TPU with Gemma 4 was validated with an incremental step by step approach.

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
