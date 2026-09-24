---
title: "Gemma 4 as a Jev-Style Decision Model on One EC2 L4: Level With Jev on Yes/No, 4.5 Points Behind on Multiple Choice"
published: false
description: "Plain Gemma 4 26B read by its label probabilities against DiffusionGemma's one-step read, both as community 4-bit (AWQ) builds on one EC2 L4, on 1,200 human-labelled examples and on the 3,880-record public suite where Jev 1.13.0 has published results. Pre-registered, with accuracy, calibration, calibration after 0 to 150 labels, latency and cost."
tags: gemma, aws, machinelearning, llm
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/jev/devto-gemma4-l4-banner.cb16905f.jpg
---

This article provides a step by step guide to measuring Gemma 4 26B as a Jev-style decision model on an AWS EC2 L4 GPU, and compares a plain read of its label probabilities against DiffusionGemma's one-step read. The measurement was pre-registered, and every per-item output is committed.

https://github.com/xbill9/gemma4-dev/tree/main/jev

---

#### Why Measure This?

A Jev-style decision model answers a typed question with a probability for each allowed option, in one forward pass, with no generated text. TypeSafe's Jev does this as a hosted service. Any open model can do it: end the prompt where the answer starts, read the scores of the allowed label tokens, and apply a softmax over those.

A companion review of the independent evidence on Jev found two open questions for Gemma. Plain Gemma read this way had no published accuracy or calibration result. DiffusionGemma, read through vLLM PR #57250, had been described as well calibrated with no measurement behind it.

This run answers both on the same GPU, the same prompts, the same label tokens and the same scoring code, using community 4-bit (AWQ) builds of both models. A third run puts both, and Gemma 4 E4B, on the public suite where Bespoke Labs has published results for Jev, so the Gemma numbers sit beside Jev's on identical records. Google's reference checkpoints are bf16, and results at bf16 may differ.

---

#### At This Point You Should Have…

- An AWS account with quota for one G-family instance, and the AWS CLI logged in
- A security group, subnet and instance profile with `AmazonSSMManagedInstanceCore` for remote commands
- Python 3 with `transformers` and `pybase64` for the client side
- The repository cloned: `git clone https://github.com/xbill9/gemma4-dev` and `cd gemma4-dev/jev`

---

#### Step 1 — Pre-Register the Measurement

The models, image, serving flags, data, metrics and comparisons are written down and committed before any model call. The file is `PREREGISTRATION.md`.

```shell
git show -s --oneline 3c67f62 d94471d adf0c16
```

```plaintext
3c67f62 jev: pre-registration for the L4 AWQ run; option-order variant, label-count calibration curve, EC2 user-data
d94471d jev: run 2026-09-23-l4-awq — Gemma 4 26B label logits vs DiffusionGemma one-step reads, both AWQ 4-bit on one EC2 L4; per-item outputs, summary, host evidence
adf0c16 jev: Bespoke Labs public-suite run — rebuild with checksum checks, runner via the proxy's Jev parser, scorer matched to Nimble's definitions, instance driver, generic launcher; pre-registration addendum
```

The public-suite run has its own dated addendum in the same file, committed before its first call. The pre-registration commits to publishing every result, including any where either arm does worse.

---

#### Step 2 — Pick a Matched Pair of Checkpoints

An NVIDIA L4 has 24 GB of memory, so both 26B models run at 4 bits. Both checkpoints come from the same quantizer with the same settings, which keeps the model the only difference between the two arms.

| | Plain arm | Diffusion arm |
|---|---|---|
| Checkpoint | `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` | `cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4` |
| Format | compressed-tensors, 4-bit, group 32, symmetric | same |
| Kept in bf16 | dense MLP and router | same |
| Size | 17.19 GB | 17.22 GB |

The diffusion checkpoint is the one used in the published DiffusionGemma-on-L4 recipe, and both arms share its serving flags.

---

#### Step 3 — Launch One L4 on EC2

The security group opens port 8000 to one address only.

```shell
aws ec2 create-security-group --region us-east-1 --group-name jev-eval-sg \
  --description "jev eval: vLLM 8000 from one IP" --vpc-id <vpc-id> --query GroupId --output text
aws ec2 authorize-security-group-ingress --region us-east-1 --group-id <sg-id> \
  --protocol tcp --port 8000 --cidr <your-ip>/32
```

```plaintext
sg-011776af18b8d0265
```

The instance boots the regional GPU Deep Learning AMI and runs `aws/user-data.sh`, which pulls a vLLM nightly that contains PR #57250 and serves one model at a time.

```shell
aws ec2 run-instances --region us-east-1 --image-id <gpu-dlami> --instance-type g6.xlarge \
  --subnet-id <subnet-id> --security-group-ids <sg-id> \
  --iam-instance-profile Name=<profile> --user-data file://aws/user-data.sh \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3,DeleteOnTermination=true}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=jev-eval},{Key=ManagedBy,Value=jev}]'
```

```plaintext
i-0c5b14e913b4c1019	g6.xlarge	us-east-1a
```

A `g6.xlarge` has the same L4 as the larger sizes with 16 GB of host memory, so `user-data.sh` adds a 16 GB swapfile on hosts under 30 GB; both 17 GB checkpoints loaded with it in place.

---

#### Step 4 — Serve the Plain Arm

One flag set serves both arms: Triton attention, eager mode, prefix caching, 2048-token context, 16 sequences and 32 logprobs. The diffusion arm adds a 64-token canvas.

```shell
/opt/jev/serve.sh ar
```

The client waits on `/v1/models` until the model answers.

```plaintext
READY after ~480s
['cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit']
```

That covers the image pull, the 17 GB download and the load. The host record from the instance:

```plaintext
NVIDIA L4, 595.91.07, 23034 MiB
image: vllm/vllm-openai:nightly-e9757321527ca1ecd514c07c1418dd2c53da3d19 digest: sha256:f75fec992c293dd41042745f8aa71bfdf12b188b6a01e0baf075e4bca5d297e5
vllm 0.29.1rc1.dev573+ge97573215 torch 2.13.0+cu130
```

---

#### Step 5 — Check the Read Before the Full Run

Both arms use the prompt, answer template, label tokens and probability code from the PR's own `structured_server.py`. `run_eval.py` confirms both chat templates render the identical prefix before it sends anything. A five-example run shows what one record holds (truncated below).

```shell
python3 run_eval.py --arm autoregressive --upstream http://<host>:8000 \
  --model cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit --run smoke --limit 5 --concurrency 2
```

```plaintext
autoregressive sst2: 5/5 (5s)
autoregressive ag_news: 5/5 (1s)
autoregressive emotion: 5/5 (0s)
autoregressive irony: 5/5 (0s)
```

```plaintext
{"id": "ag_news-5932", "task": "ag_news", "arm": "autoregressive", "variant": "none", "model": "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit", "gold": "business", "names": ["world", "sports", "business", "scitech"], "labels": ["A", "B", "C", "D"], "reads": [{"probs": [2.0848258713712555e-11, 1.3946453462126672e-09, 0.9999999985082466, 7.625976171822172e-11], "label_mass": 0.9995792505931735, "entropy": 0.00042069308461752564, "argmax_is_label": true, "labels_returned": 4}], ...
```

`labels_returned: 4` means every label came back with a real probability. `label_mass` is the share of the model's probability that landed on the allowed labels.

---

#### Step 6 — Run Both Arms

The data is 300 human-labelled examples each from sst2, AG News, DAIR Emotion and tweet_eval irony, stratified with a fixed seed. Each arm runs once as written and once with the choice options listed in reverse order.

```shell
python3 run_eval.py --arm autoregressive --upstream $U --model $PLAIN --run 2026-09-23-l4-awq --concurrency 8
python3 run_eval.py --arm autoregressive --upstream $U --model $PLAIN --run 2026-09-23-l4-awq --concurrency 8 --variant reversed
/opt/jev/serve.sh diffusion
python3 run_eval.py --arm diffusion --upstream $U --model $DIFF --run 2026-09-23-l4-awq --concurrency 4
python3 run_eval.py --arm diffusion --upstream $U --model $DIFF --run 2026-09-23-l4-awq --concurrency 4 --variant reversed
wc -l results/2026-09-23-l4-awq/diffusion-*.jsonl
```

```plaintext
    300 results/2026-09-23-l4-awq/diffusion-ag_news--reversed.jsonl
    300 results/2026-09-23-l4-awq/diffusion-emotion.jsonl
    300 results/2026-09-23-l4-awq/diffusion-emotion--reversed.jsonl
    300 results/2026-09-23-l4-awq/diffusion-irony.jsonl
    300 results/2026-09-23-l4-awq/diffusion-sst2.jsonl
    300 results/2026-09-23-l4-awq/diffusion-sst2--reversed.jsonl
   2100 total
```

The diffusion arm reads each example four times with different noise in the answer slots, as the PR's proxy does. The DiffusionGemma server came up in about 300 seconds with the image already pulled.

---

#### Step 7 — Score It

```shell
python3 score.py --run 2026-09-23-l4-awq
```

`score.py` writes `SUMMARY.md` and `summary.json` beside the records. Every figure below comes from those two files. Readouts: `plain` is one read of plain Gemma, `diffusion` is the mean of four DiffusionGemma reads, which is what the proxy's automatic rule returned on 96% to 100% of examples.

---

#### How Accurate Is Each One?

Level on three tasks, plain ahead on AG News.

| Task | Always-majority | Plain Gemma | DiffusionGemma | Difference (95% range) |
|---|---|---|---|---|
| sst2 | 51.0% | 🥇 95.0% | 93.3% | −1.7 (−4.0 to +0.7) |
| AG News | 25.0% | 🥇 86.7% | 83.0% | −3.7 (−6.7 to −0.7) |
| DAIR Emotion | 35.0% | 58.3% | 🥇 60.3% | +2.0 (−1.0 to +5.0) |
| tweet_eval irony | 60.3% | 🥇 89.7% | 87.3% | −2.3 (−5.7 to +1.0) |

Only the AG News difference excludes zero. All four test sets are public and may have been in either model's training data; the irony scores are high enough for that test set to make it likely there.

---

#### Are the Raw Probabilities Calibrated?

DiffusionGemma's are closer. Plain Gemma's are strongly overconfident.

| Task | Plain Gemma ECE | DiffusionGemma ECE | Difference (95% range) |
|---|---|---|---|
| sst2 | 0.051 | 0.045 | −0.005 (−0.021 to +0.020) |
| AG News | 0.129 | 0.119 | −0.010 (−0.030 to +0.020) |
| DAIR Emotion | 0.386 | 0.261 | −0.125 (−0.154 to −0.089) |
| tweet_eval irony | 0.098 | 0.043 | −0.055 (−0.077 to −0.014) |

ECE is expected calibration error over 15 bins: the gap between how confident a model says it is and how often it is right. DiffusionGemma is better calibrated on emotion and irony, and level on the other two.

The fitted temperatures show the size of the gap. One temperature that best fits the labels was 3.48 to 7.24 for plain Gemma once 25 or more labels were used, and 0.73 to 2.36 for DiffusionGemma. A temperature above 1 means the model's probabilities are too extreme.

---

#### What Do 50 Labels Buy?

Most of the correction, for both models.

Each task's examples split once into a fitting half and a held-out half. One temperature is fitted on the first N labels of the fitting half and scored on the held-out half.

| Task | Plain, 0 labels | Plain, 50 labels | Diffusion, 0 labels | Diffusion, 50 labels |
|---|---|---|---|---|
| sst2 | 0.040 | 0.029 | 0.047 | 0.025 |
| AG News | 0.139 | 0.064 | 0.132 | 0.061 |
| DAIR Emotion | 0.440 | 0.102 | 0.311 | 0.115 |
| tweet_eval irony | 0.088 | 0.042 | 0.055 | 0.054 |

That is the pre-registered single split. One split of 150 held-out examples moves ECE by several hundredths, so the same fit was repeated over 20 random splits, an analysis added after the run:

| Task | Plain, 0 labels | Plain, 50 labels | Diffusion, 0 labels | Diffusion, 50 labels |
|---|---|---|---|---|
| sst2 | 0.048 | 0.039 (0.021–0.061) | 0.050 | 0.042 (0.025–0.081) |
| AG News | 0.130 | 0.066 (0.037–0.132) | 0.121 | 0.085 (0.060–0.138) |
| DAIR Emotion | 0.392 | 0.089 (0.047–0.161) | 0.266 | 0.111 (0.062–0.155) |
| tweet_eval irony | 0.099 | 0.060 (0.035–0.130) | 0.054 | 0.066 (0.031–0.116) |

The pre-registered analysis is the single split above; these means over 20 splits, with the range in brackets, were added to show how much one split moves it. With no labels, DiffusionGemma is better calibrated on emotion and irony. With 50, plain Gemma's mean is at or below DiffusionGemma's on all four tasks, and the ranges overlap throughout.

---

#### Where Does DiffusionGemma's Probability Go?

Half of it lands outside the allowed labels. The median share on the allowed labels per task:

| Task | Plain Gemma | DiffusionGemma |
|---|---|---|
| sst2 | 1.000 | 0.503 |
| AG News | 0.999 | 0.370 |
| DAIR Emotion | 1.000 | 0.392 |
| tweet_eval irony | 1.000 | 0.593 |

The latency run kept the five highest tokens returned at the answer slot. Where a token outside the allowed labels made those five, it was most often `<eos>`, the end-of-sequence token: in 40% of reads on sst2, 53% on AG News and 64% on DAIR Emotion. On irony it was most often ` the`, in 24% of reads. At the answer slot, DiffusionGemma gives much of its probability to the answer ending there.

The proxy rescales the label probabilities to sum to one, so the answer reads as confident either way. `label_mass` is the field that shows it, and it is worth logging in any deployment.

---

#### Does the Automatic Re-Read Rule Work?

It fires on nearly every example. The proxy re-reads when the answer slot's entropy is above 0.1, and it measures that entropy over the full vocabulary, where DiffusionGemma's off-label probability lives. It re-read 100% of sst2 and AG News examples, 99.0% of DAIR Emotion and 96.3% of irony.

The re-reads also add little information. On three tasks, 60.8% to 65.0% of wrong answers had a spread across the four reads under 0.02. On irony the spread did separate right from wrong, with an AUROC of 0.873.

---

#### Does Option Order Change the Answer?

For both models, by similar amounts. Listing the choice options in reverse order changed:

| Task | Plain Gemma | DiffusionGemma |
|---|---|---|
| sst2 | 6 of 300 (2.0%) | 7 of 300 (2.3%) |
| AG News | 14 of 300 (4.7%) | 21 of 300 (7.0%) |
| DAIR Emotion | 27 of 300 (9.0%) | 26 of 300 (8.7%) |

---

#### How Fast Is Each One?

A second run timed both arms with the client on the instance itself, against `localhost`, one request at a time, over the first 100 examples of each task. Same image, flags and checkpoints, on a `g6.4xlarge`: the same L4 with a larger host, launched because no `g6.xlarge` was free.

```shell
python3 run_eval.py --arm autoregressive --upstream http://localhost:8000 --model $PLAIN --run 2026-09-23-l4-latency --limit 100 --concurrency 1
python3 run_eval.py --arm diffusion --upstream http://localhost:8000 --model $DIFF --run 2026-09-23-l4-latency --limit 100 --concurrency 1 --reads 4 --keep-top 5
```

```plaintext
| Task | Plain, median | Plain, 90th pct | Diffusion one read, median | Diffusion one read, 90th pct | Diffusion 4 reads (first + 3 parallel), median |
|---|---|---|---|---|---|
| sst2 | 61 | 62 | 118 | 119 | 306 |
| ag_news | 61 | 62 | 119 | 120 | 307 |
| emotion | 61 | 62 | 119 | 120 | 308 |
| irony | 61 | 61 | 121 | 122 | 312 |
```

Times are in milliseconds. A plain Gemma 26B decision takes 61 ms on one L4 for these prompts, which run 105 to 299 tokens with a median of 132; longer inputs such as full tickets or logs take longer. One DiffusionGemma read takes about twice that, and the automatic rule's four reads about five times, which is the path it takes on nearly every call.

---

#### 🔎 Tip: Fit One Temperature Before You Trust a Threshold

Both models give usable probabilities after one fitted temperature. `score.py` does it with a grid search on log loss:

```python
def fit_temperature(probs, gold_idx):
    return min(T_GRID, key=lambda t: nll([temper(p, t) for p in probs], gold_idx))
```

Label 50 real decisions from the task, fit on them, and set any act-or-escalate threshold on the rescaled probabilities. Plain Gemma needs the larger correction, and both reach similar calibration once it is applied.

---

#### What About the Smaller Gemma 4 Models?

The label read works with any Gemma 4. DiffusionGemma ships only at 26B-A4B, so the smaller models get the plain arm alone, run in bf16 on the same L4 with the same flags. Their chat templates end at the model turn without the empty thought block the 26B template adds, so their prompt follows their own template. These two arms were added after the first run and are exploratory. The 26B column is 4-bit and the others bf16, so the gaps mix model size with precision. All three read letter labels (A to F) with the meanings listed in the prompt, the format of the PR's proxy; smaller models may do better with the label words as the answers themselves.

| Task | 26B, 4-bit | E4B, bf16 | E2B, bf16 |
|---|---|---|---|
| sst2 | 🥇 95.0% | 94.3% | 88.7% |
| AG News | 🥇 86.7% | 83.7% | 30.3% |
| DAIR Emotion | 🥇 58.3% | 54.0% | 53.3% |
| tweet_eval irony | 🥇 89.7% | 84.3% | 72.7% |

E4B is within 1 to 5 points of the 26B, needs a smaller correction (fitted temperature 1.89 to 3.77 at 50 labels, against 4.65 to 7.05), and reaches similar calibration once corrected: 0.042, 0.071, 0.089 and 0.081 against the 26B's 0.039, 0.066, 0.089 and 0.060. It changed more answers when the options were reversed on AG News, 35 of 300 against 14.

E2B answered "world" on 277 of 300 AG News examples, and on 263 when the options were listed in reverse order, so the same option wins whatever its position, and its topic accuracy sits near the 25% majority rate. Its fitted temperatures reach 7.99, the top of the pre-registered search range, so its corrected calibration figures understate what a wider search would reach.

---

#### How Does It Compare With Jev?

Bespoke Labs, which makes the open Nimble-9B decision model, ran Jev 1.13.0 through its API on a 13-subset public suite and published the results per subset, with the converters and record ids. The suite holds 3,880 human-labelled records: yes/no questions from BoolQ, PAWS, SQuAD 2.0, Civil Comments and Aegis 2.0; multiple choice from MultiNLI, PubMedQA, VitaminC and MASSIVE intents in English and German; and five-level ratings from HelpSteer2 and SummEval.

A third run, on a `g6.xlarge` with the same image and flags, rebuilt the suite from the public sources and read it with all three Gemma arms.

```shell
bash nimble_suite/build.sh <workdir>
```

```plaintext
MATCH aegis2 250
MATCH boolq 300
MATCH civil_comments 300
```

All 13 subsets matched the published checksums, so the Gemma figures and Bespoke Labs' Jev figures come from the same records. Each record is a Jev request, which the PR's proxy converts with its own Jev parser, so the Gemma arms see the same prompt format as the first run. Scoring uses Bespoke Labs' definitions, and `tests/test_suite_scoring.py` checks it against Bespoke Labs' own code.

```shell
python3 nimble_suite/run_suite.py --arm autoregressive --upstream http://localhost:8000 \
  --model $PLAIN --records <workdir>/public --run 2026-09-24-l4-suite
python3 nimble_suite/suite_stats.py --run 2026-09-24-l4-suite --run 2026-09-24-l4-suite-e4b
```

Accuracy pooled over records, with the 95% range in brackets:

| Questions | Records | Jev 1.13.0, published | Plain Gemma 26B | DiffusionGemma 26B | Gemma 4 E4B |
|---|---|---|---|---|---|
| All | 3,880 | 🥇 77.3% (76.0–78.6) | 75.3% (73.9–76.6) | 75.9% (74.5–77.2) | 73.3% (71.9–74.6) |
| Yes/no | 1,399 | 84.6% (82.6–86.4) | 🥇 84.8% (82.8–86.6) | 84.5% (82.5–86.3) | 81.9% (79.8–83.8) |
| Multiple choice | 1,848 | 🥇 82.8% (81.1–84.5) | 78.3% (76.4–80.1) | 77.3% (75.4–79.2) | 74.8% (72.8–76.8) |
| Five-level rating | 633 | 45.2% (41.3–49.1) | 45.5% (41.7–49.4) | 🥇 52.8% (48.9–56.6) | 49.6% (45.7–53.5) |

On yes/no questions the 26B models match Jev. On multiple choice Jev leads plain Gemma by 4.5 points, and PubMedQA carries the largest share: 77.2% against 64.0%. On ratings DiffusionGemma leads. Gemma 4 E4B trails the 26B by 2.0 points over the whole suite.

Plain Gemma and DiffusionGemma are level over all 3,880 records: 187 right only for plain, 211 right only for DiffusionGemma, exact McNemar p = 0.25. On ratings DiffusionGemma was right alone on 97 records against 51, p = 0.0002, and the whole gap comes from SummEval, whose 384 records come from 24 news articles. Records from one article move together, so that p-value overstates the evidence.

Median calibration error over the 13 subsets, with Bespoke Labs' 10 bins:

| | As shipped | After 50 labels |
|---|---|---|
| Jev 1.13.0, published | 🥇 0.071 | |
| Nimble-9B, published | 0.109 | |
| Plain Gemma 26B | 0.180 | 0.080 |
| DiffusionGemma 26B | 0.114 | 0.074 |
| Gemma 4 E4B | 0.173 | 0.077 |

As shipped, Jev has a lower calibration error than plain Gemma on all 13 subsets, than DiffusionGemma on 11 and than E4B on 12. After one temperature fitted on 50 labels, the Gemma medians sit 0.003 to 0.009 above Jev's, and at or below Jev's on 4 to 6 of the 13 subsets. The fitted figures are scored on the held-out half of each subset, 72 to 300 records, and ECE reads higher on fewer records, which works against the Gemma columns. Jev's figures are as shipped, and a temperature fitted on Jev's own output could lower them as well.

DiffusionGemma's median share on the allowed labels was 66.5% per read on this suite, against 100% for plain Gemma.

Bespoke Labs publishes a competing model, and its Jev figures come from one run of Jev 1.13.0; it reports that Jev's API rounds each probability to two decimals. Most of the subsets predate Gemma 4 and may be in its training data, and whether they are in Jev's is unknown.

---

#### Compare and Contrast

| | Plain Gemma 4 26B, label read | DiffusionGemma 26B, one-step read |
|---|---|---|
| Accuracy, four tasks | 🥇 level or ahead on all four | level on three, 3.7 points behind on AG News |
| Accuracy, public suite | 75.3%, 🥇 78.3% on multiple choice | 75.9%, 🥇 52.8% on ratings |
| Raw calibration | overconfident, fitted temperature 3.5 to 7.2 | 🥇 closer, fitted temperature 0.7 to 2.4 |
| Median ECE, public suite, after 50 labels | 0.080 | 0.074 |
| After 50 labels | level | level |
| Probability on the allowed labels | 🥇 about 100% | 37% to 59%, 66.5% on the public suite |
| Time per decision on the instance | 🥇 61 ms | 118 to 121 ms for one read, 306 to 312 ms with the automatic rule |
| Option-order changes | 2.0% to 9.0% | 2.3% to 8.7% |
| Serving | any vLLM | vLLM with PR #57250 |

---

#### So, Which One?

For a Jev-style decision service on one L4, plain Gemma 4 26B read by its label probabilities, plus one temperature fitted on about 50 labels. It is as accurate or more, faster, and places all of its probability on the answers you allowed.

DiffusionGemma's lower raw calibration error is real on two of four tasks, and it matters when no labels exist at all. With 50 labels, the difference is gone. On five-level ratings it scored higher on the public suite, on few source articles; a rating task is the one place to try both.

Against Jev, on the same public records: a Gemma 4 26B read matches Jev on yes/no questions, trails it by 4.5 points on multiple choice, and matches its calibration within 0.01 once fitted on 50 labels. Jev is the better calibrated with no labels at all.

Where a smaller model has to do, Gemma 4 E4B in bf16 gives up 1 to 5 points against the 26B. E2B falls apart on four-way topic classification with this prompt.

---

#### What Does a Decision Cost?

The run logs give a throughput. Plain Gemma answered 2,100 decisions in 51 seconds, 41.2 a second, at client concurrency 8. DiffusionGemma, reading each example four times, answered 2,100 in 298 seconds, 7.0 a second, at concurrency 4. The client was on a home connection, so these are lower bounds on what the L4 serves.

At the `g6.xlarge` on-demand price of $0.8048 an hour, that is at most $5.43 per million decisions for plain Gemma and $31.72 for DiffusionGemma with four reads. TypeSafe prices Jev at $0.042 per million input tokens, which is $6.30 to $12.60 per million decisions at 150 to 300 input tokens each. The L4 is charged by the hour whether busy or idle, so its per-decision cost holds only at full load.

At 61 ms per decision one request at a time, one L4 answers about 16 plain decisions a second before any batching; the 41.2 a second above came from eight requests in flight.

The first run, from launch to termination in under 1.2 hours, cost at most $0.97 of instance time. The second, on a `g6.4xlarge` at $1.3232 an hour for 1.19 hours, cost $1.57. The public-suite run, on a `g6.xlarge` for 0.82 hours, cost $0.66, including rebuilding the suite and reading 11,640 records across three models. All three exclude the prorated 100 GB volume. All ran on demand; G-family spot capacity in us-east-1 was unavailable at launch time.

---

#### Teardown

Every command on the instance went through AWS Systems Manager, so no SSH port was ever open, and port 8000 was open to one address.

```shell
aws ec2 terminate-instances --region us-east-1 --instance-ids i-0c5b14e913b4c1019
```

```plaintext
shutting-down
```

Then delete the security group, and confirm that nothing tagged `ManagedBy=jev` remains.

---

#### Summary

The goal of this article was to measure Gemma 4 26B as a Jev-style decision model, read two ways, for accuracy and calibration on human-labelled data, and to set it beside Jev's published results on a public suite. The key to the solution was a matched pair of 4-bit checkpoints on one EC2 L4, identical prompts and label tokens, and a measurement pre-registered before any call. The results were:

- 🟢 Plain Gemma 4 26B read by label probabilities: 95.0%, 86.7%, 58.3% and 89.7% on sst2, AG News, DAIR Emotion and irony
- 🟢 DiffusionGemma level on three tasks, 3.7 points behind on AG News
- 🟢 DiffusionGemma's raw calibration error lower on emotion (0.261 against 0.386) and irony (0.043 against 0.098)
- 🟢 One temperature fitted on 50 labels brings both to similar calibration
- 🟢 On the instance, one request at a time: 61 ms per plain decision, against 118 to 121 ms for one DiffusionGemma read and 306 to 312 ms for four
- 🟢 At most $5.43 per million plain decisions on a `g6.xlarge` at full load
- 🟢 Gemma 4 E4B in bf16 within 1 to 5 points of the 26B, with similar calibration after 50 labels
- 🟢 On Bespoke Labs' 3,880-record public suite, plain Gemma 4 26B matched Jev 1.13.0 on yes/no questions, 84.8% against 84.6%
- 🟢 Plain Gemma and DiffusionGemma level over the whole suite, 75.3% and 75.9%, p = 0.25
- 🟢 After 50 labels, median calibration error 0.074 to 0.080 for the Gemma arms against Jev's 0.071
- 🟢 All three runs cost $3.20 of instance time
- ⚠️ Jev leads plain Gemma by 4.5 points on multiple choice, most of it on PubMedQA
- ⚠️ Jev is the best calibrated with no labels: median 0.071 against 0.114 to 0.180
- ⚠️ DiffusionGemma places 37% to 59% of its probability outside the allowed labels, and the proxy's rescaling hides it
- ⚠️ The proxy's automatic re-read rule fired on 96% to 100% of examples, doubling DiffusionGemma's time per decision
- ⚠️ Irony scores are high enough to suggest the public test set was in training data
- ⚠️ Spread across DiffusionGemma's four reads missed 60.8% to 65.0% of wrong answers on three tasks
- ❌ Gemma 4 E2B answered "world" on 277 of 300 AG News examples

Scope: one NVIDIA L4 in us-east-1 on three instances, a `g6.xlarge` for the accuracy run, a `g6.4xlarge` for the latency and small-model run and a `g6.xlarge` for the public suite, vLLM `0.29.1rc1.dev573+ge97573215`, the 26B models 4-bit from the same quantizer and E4B and E2B in bf16, 300 examples per task plus the 3,880-record public suite, one run per arm. Latency comes from 100 examples per task with the client on the instance; throughput comes from the first run's client on a home connection. The E4B and E2B arms and the 20-split calibration were added after the pre-registration and are labelled as such. bf16 weights and other GPUs were outside this run. No Jev call was made; the Jev and Nimble-9B figures are Bespoke Labs' published results on the same records. Code, pre-registration and every per-item output are in the repository. Parts of the analysis and writing were done with AI assistance (Claude); every figure comes from the committed output files.

The strategy for using label probabilities to run Gemma 4 as a decision model was validated with an incremental step by step approach.

---

#### References

- Code, pre-registration and per-item results: https://github.com/xbill9/gemma4-dev/tree/main/jev
- vLLM PR #57250, DiffusionGemma structured reads: https://github.com/vllm-project/vllm/pull/57250
- Plain checkpoint: https://huggingface.co/cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit
- Diffusion checkpoint: https://huggingface.co/cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4
- DiffusionGemma model card: https://huggingface.co/google/diffusiongemma-26B-A4B-it
- Guo et al., On Calibration of Modern Neural Networks: https://arxiv.org/abs/1706.04599
- Amazon EC2 G6 instances: https://aws.amazon.com/ec2/instance-types/g6/
- TypeSafe Jev: https://docs.typesafe.ai/concepts/system-one
- Bespoke Labs, Nimble public-suite results for Jev 1.13.0 and Nimble-9B: https://github.com/bespokelabsai/nimble/blob/0e67403/docs/PUBLIC_BENCHMARKS.md
