---
title: "Jev After Eight Days of Independent Tests: Level With Mid-Price LLMs, Behind the Frontier"
published: false
description: "What the independent measurements of TypeSafe's Jev found in its first eight days: arXiv preprints, GitHub evaluations and blog benchmarks, each traced to its primary source. Accuracy, calibration, speed, cost, failure modes, the prior art, the open alternatives, and what is still unmeasured."
tags: ai, machinelearning, llm, gemma
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/jev/devto-jev-review-banner.2360df33.jpg
---

This article provides a review of the independent evidence on TypeSafe's Jev, the open models built to replace it, and the prior art behind both, as of September 23, 2026. Every figure below is traced to a primary source, and re-scored from committed per-item outputs wherever the author published them. This is a snapshot eight days after launch, and the arXiv preprints it cites are days old and unrefereed.

On accuracy, Jev sits level with mid-price LLMs and 6.5 to 11.5 points behind the frontier in the cleanest comparison. Out of the box its probabilities are the best calibrated of the models measured on familiar English tasks and are wrong in both directions off them, and one temperature fitted on 50 to a few hundred labels fixes most of the error.

https://github.com/xbill9/gemma4-dev/tree/main/jev

---

#### What Is Jev?

Jev is a hosted model from TypeSafe AI, launched on September 15, 2026. It answers typed questions about a piece of text: `choice` picks one option from a list, `score` places the text on an ordered scale, and `noul` returns the probability that a statement is true. It writes no text.

The API is `POST /v1/systemone` at `api.typesafe.ai`. The price is $0.042 per million input tokens, and output is free. Access is through a waitlist, OpenRouter and Vercel's AI Gateway.

TypeSafe calls this category a System One model, after Kahneman's fast and intuitive system. The company raised a $40M seed round led by DCVC, and its co-founders are Diogo Almeida (CEO), Sasha Sheng (COO) and Erik Gafni (CTO).

The read itself is an old technique: end the prompt where the answer would start, take the model's scores for only the allowed label tokens, and apply a softmax over them. Any open model can be read this way; vLLM exposes it through `logprob_token_ids`, SGLang through `/v1/score`, and Featherless AI's SimpleJev wraps it as a Jev-compatible server.

---

#### What Counts as Evidence Here?

The review covers 14 arXiv preprints read in full, 104 GitHub evaluation repositories, 33 dev.to and Medium posts, TypeSafe's own pages and documentation, and press and forum coverage. Each figure carries one of three levels of support.

- **Re-scored:** recomputed from committed per-item outputs, and it matched
- **Checked:** the source's arithmetic recomputed from its published tables, with no per-item data available
- **Reported:** the author's statement, which cannot be checked from outside

Studies are graded A (human or public labels, at least 200 items, stated method, committed outputs), B (real but small, synthetic, LLM-labelled or unreproducible) or C (a demo or a restatement).

By that rubric, the posts split sharply. Of 33 dev.to and Medium posts read in full, 5 are careful measurements on human or public labels, 11 are small, synthetic or scored by agreement between models, and 17 restate vendor claims. Of the 104 GitHub repositories, 33 grade A, 57 grade B, 5 grade C, and 9 are tools. Of the 14 arXiv preprints, 10 call TypeSafe's hosted Jev.

Who ran a study matters too. The studies with committed per-item outputs and no product in the race are Ibrahim and Zaki, Janardhan, sanand0, SamuelSacco and OmarMujahid, the last the most favourable to Jev; Rafe and Das also has none, and withholds its per-item outputs. Every's review was written by a launch partner. LangChain, which ships the `langchain-typesafe` integration, reported 100% on 500 judgments that are 5 test cases repeated 100 times (not re-checked). And the "15.9% faster pipeline" headline came from a TypeSafe employee's three-case demo. Bespoke Labs sells Nimble, and the replica authors test their own models. Matt Mastracci wrote the DiffusionGemma PR he compares with Jev, and Google's Gemma account shared his thread. heiko-hotz, whose SNIPS comparison includes Gemma 4 31B, works at Google. synthorai, which found Jev behind flash LLMs, runs a gateway that sells access to the models it compared.

---

#### Where Does Jev Land on Accuracy?

Level with mid-price LLMs and behind the frontier, at a fraction of the frontier's price. That trade is the product's design, and the numbers below show its size.

The largest pre-registered study, Ibrahim and Zaki's replication of a social-science annotation suite with 7,977 human-labelled items (arXiv 2609.24574), has Jev behind the best of 19 LLMs on 14 of 15 tasks, by a median 11.6 macro-F1 points. That comparator is the best model per task, picked after the results; within Jev's price band, Gemma 4 31B and Qwen3 235B score 3 and 2 F1 points above it. Its one per-item file checked, the 266-item dialect task, re-scores exactly to the paper's table.

Manjunath Janardhan's 200-item, six-model comparison on BANKING77, BoolQ, Yelp and ChaosNLI re-scores exactly from its committed logs.

| Model | Accuracy | Calibration error (ECE) |
|---|---|---|
| 🥇 Claude Fable 5.1 | 84.0% | 0.064 |
| 🥈 GPT-6 Astra | 79.0% | 0.119 |
| 🥉 DeepSeek V4.1 Flash | 76.0% | 0.138 |
| MiniMax M3 | 75.5% | 0.112 |
| Kimi K3 | 74.5% | 0.119 |
| Jev | 72.5% | 0.161 |

The two frontier leads (+11.5 and +6.5 points) exclude zero in their 95% intervals. Kimi K3, MiniMax M3 and DeepSeek V4.1 Flash sit within noise of Jev.

Banking77 is the one task many authors ran independently. Eight runs, one of them JevBench's own first-party run, put Jev at 0.753 to 0.840, with a median of 0.809. In every study that ran LLMs beside it, the best of them was ahead.

---

#### What Does Jev Do Well?

Binary and few-class decisions, fast, with answers that always fit the schema.

- **Spam:** 98.33% on 18,514 emails, level with TF-IDF logistic regression at 98.39%, and it held up on newer mail where the trained baseline fell to 72.5%; the question wording was tuned on labelled errors from the same sets
- **Reranking:** level with a dedicated reranker on eight retrieval sets, 0.692 against 0.691
- **Monitoring code for backdoors:** AUROC of 0.976 and 0.970, catching about 90% of backdoors at 2% false positives; a best-of-5 attack cut detection to 60%
- **Schema compliance:** zero invalid answers across 23,703 decision-model calls, Jev's included, in the largest study
- **Repeatability:** 1.33% of answers changed between identical passes in one study, 2.2% in the phishing benchmark, and 0 of 96 repeats in another
- **Language:** German cost 0.5 points on MASSIVE
- **Speed and price:** about 0.1 s of server time, at $0.042 per million input tokens

---

#### Are Its Probabilities Calibrated?

They rank answers usefully; out of the box they beat most alternatives on familiar English tasks and miss badly off them.

TypeSafe publishes no calibration error, reliability plot, Brier score or log loss for Jev on any dataset. Its only confidence-to-accuracy figure is one cookbook of 60 SEC filings: 90% right at confidence 0.9 or above, 40% below.

The independent picture depends on the comparison.

- On Bespoke Labs' 13-subset public suite, Jev's median ECE is 0.071, lower than Nimble-9B's on 11 of 13 subsets, and lower than every Gemma 4 read in the companion article (0.114 to 0.180) before any fitting.
- Against LLMs that write their confidence as a number, Jev's own probabilities usually win. Its median ECE of 0.157 on the social-science tasks beats 16 of 19 LLMs.
- Once every model gets one fitted temperature, 15 of those LLMs beat it.
- Against LLMs that return full probability distributions, Jev's calibration error is the highest in Janardhan's run (0.161) and in sanand0's nine-model Banking77 pilot (0.138).

The direction of the error changes with the data. On public multi-class sets Jev is overconfident: on GoEmotions, labels it scored between 0.80 and 0.95 matched the human label 15% of the time. On crash narratives and synthetic items it is under-confident. Within one judge study, refit temperatures ran from 0.65 to 4.45, and its authors conclude that no single temperature fits.

The API also rounds every probability to 0.01. In one committed sample, 70.4% of `choice` probabilities came back as exactly 0, and `noul` is clamped between 0.01 and 0.98.

---

#### 🔎 Tip: Fit a Temperature on 50 to a Few Hundred of Your Own Labels

A small refit fixes most of the calibration error, for Jev and for any open model read the same way.

- A two-parameter refit on civil_comments took Jev's ECE from 0.16–0.21 to under 0.025 on a held-out half
- Transferring a fitted slope and refitting only the intercept on 50 labels cut mean ECE by 74%
- Out-of-fold Platt scaling cut Jev's ECE on crash narratives by 3.35x
- In the companion article, one temperature fitted on 50 labels brought plain Gemma 4 26B's median ECE from 0.180 to 0.080 on the public suite, against Jev's 0.071 as shipped

The open replicas report the same order of correction: Nimble fitted 2.179, Luce 2.5 to 3.3 on unseen tasks. Luce's rule is to ship with 100 to 300 real labelled items and fit the temperature on them.

---

#### How Much Faster and Cheaper Is It?

Speed runs from 0.5x, slower than a local Gemma, to 12.1x faster; cost runs from 0.6x, dearer, to 478x cheaper, depending on the comparison model, the network path and where the clock runs.

| Comparison | Source | Faster by | Cheaper by |
|---|---|---|---|
| GPT-5.6 Luna, reasoning off, client clock | OmarMujahid | 1.04x | 4.3x |
| DeepSeek, Banking77 | sanand0 | — | 1.18x |
| Haiku 4.5, phishing verdict | jev-phishing-bench | 2.9x | 12x |
| Mid-size open model, 47 account reviews | devopsdaily | 12.1x | 7.4x |
| Claude Fable 5.1, via OpenRouter | Janardhan | 9.9x | 478x |
| 19 LLMs, social-science tasks (median) | arXiv 2609.24574 | — | 18.6x |
| DeepSeek, 6G edge orchestration | arXiv 2609.23136 | 1.3x | 0.6x (Jev dearer) |
| DeepSeek, edge orchestration, fees per correct completion | arXiv 2609.22753, same lab | — | 3.2–3.4x |
| Local Gemma 4 26B-A4B Q4 on a mini-PC | ikkun1222 | 0.5–0.65x (Jev slower) | — |

The server itself is fast: about 105 ms in one run, and about 76 ms once the network round trip is subtracted in another. Most of the cost gap comes from output being free. In devopsdaily's workload 83% of the old bill was output, and per input token the two prices differed by 1.31x. In the companion article, plain Gemma 4 26B on one EC2 L4 at full load cost at most $5.43 per million decisions, against Jev's $5.54 at the same 132-token prompts.

---

#### Where Does It Break?

On wording, language, and questions whose answer is absent from the input.

Option names move answers. With the question, input, rubric and option set held fixed, swapping which rubric sits behind "no" and "yes" changed 32.5% of Jev's answers, against about 2% with neutral names, and AUROC fell from 0.81 to 0.58 (arXiv 2609.26758, n = 1,200; reported). The paper is graded C as evidence about Jev: it names no dataset or Jev version, its gold labels likely come from a teacher LLM, its code is unreleased, and no generative LLM was given the same swap.

Bare labels mislead it. A router given option names with no descriptions sent all 40 hard tasks to the cheap model, at a median confidence of 0.96. One-line option descriptions fixed 37 of 40.

Language shift costs accuracy. Russian XNLI dropped from 88.3% to 77.3% with ECE tripling, and Spanish cost 3 to 6 points.

Jev is most confidently wrong where the answer does not follow from the input. With no "unknown" option, every answer on KoBBQ bias questions is wrong by design, and Jev gave them 0.79 confidence. On a fair die its `choice` probabilities averaged 83% against a true 17%, while `noul` gave 19%. On heart-risk data its probabilities ran about 3x too high, and a chat LLM on the same 5,000 people ranked risk slightly better (AUC 0.79 against 0.77).

Questions batched in one request cannot see each other's answers. One reviewer's request returned "suspend" for an account the same response classified as a developer sending tests.

TypeSafe's own list of known weaknesses names literal reading, arithmetic and counting, dates, long irrelevant input, adversarial content and phrasing: "refund" scored 0.72 and "not a refund" 0.47 on the same ticket.

---

#### What Does TypeSafe Claim, Checked?

**193.6x faster and 444.6x cheaper.** The source is TypeSafe's four in-house workflow evals, which the launch post calls "on the higher end of real world gains". TypeSafe discloses that the workflows "were made by individuals on our model capabilities team, so some bias could exist", and that its LLM comparators ran through its own probability adapter, which it says is "slower and more expensive". The comparator is never named. Checked against the published summary points, 444.6x fits only Jev against Opus 5 in the workflow setup. Averaged over all eight workflow setups, the figures are 97.8x faster and 149.2x cheaper. Timings came from laptops on the US West Coast, with no case counts, run counts or variance.

**Eval accuracy.** Reference labels are "an average of the responses of GPT-6 Astra and Claude Fable 5.1, both at high thinking", so accuracy there means agreement with two LLMs. Jev scores 67.8%, equal to Sonnet 5, with GPT-5.6 Sol leading at 74.1%. On invoice processing Jev scores 61.8% against Sol's 79.1%. TypeSafe states that it chose to publish no public-benchmark results.

**"0% hallucination".** TypeSafe's launch post: "Our number is not empirical. Schema matching is guaranteed." Every answer is a valid option; the wrong-but-valid rate is the accuracy figures above.

**RLCD.** Reinforcement learning for calibrated decisions has no paper, patent or method description. The CEO said on Hacker News that the architecture is "close to the chest for now, but we have talked about writing a paper." On Latent Space he agreed that all of Jev's training data is synthetic. Model size and base are undisclosed.

**Pricing.** The launch post says "We can't prove it isn't subsidized", and the home-page FAQ says "We can serve Jev profitably at our current prices." How long the launch price holds is an open question for anyone building on it.

**The frontier comparison.** On Hacker News a commenter described Jev as "basically a zero-shot classifier" that classifies "as accurately (they claim) as a frontier-level LLM", and the CEO replied "exactly right!" The independent accuracy results above are the test of the second half.

**The founder.** TypeSafe's pages say Diogo Almeida "co-invented RLHF". He is not an author of Christiano et al. (2017), which introduced it; he is a primary author of InstructGPT (arXiv 2203.02155), which applied it to instruction following, and on Latent Space he separates the two.

How the claims travelled:

| Claim as circulated | Primary source |
|---|---|
| 194x faster and 445x cheaper against GPT-6 Astra (Tom's Hardware) | Astra is a reference labeller with no scored setup; 444.6x fits Opus 5 |
| Coverage of arXiv 2609.26758: Jev flips 70.4 of 100 answers | The paper gives 70.4 for an open ModernBERT head; Jev's is 32.5% |
| "Its own employee measured 15.9% on a real pipeline" (Cherry Creek News) | 3 cases of a demo support ticket; costs modelled from an estimated price |

Several self-published evaluations also state more than their own data supports, in both directions. The full report lists the corrections for Luce, Laya and NanoJev and uses the corrected figures from every study.

---

#### What Predates Jev?

Every piece of the mechanism.

| Piece | Prior art |
|---|---|
| Score options by label likelihood | GPT-3 §2.4 (2020); MMLU §4.1 (2021); lm-evaluation-harness `multiple_choice` |
| Map each label to one token | Verbalizers, Schick & Schütze (arXiv 2001.07676) |
| Label probabilities carry prior bias | Calibrate Before Use, Zhao et al. (2021) |
| One fitted temperature fixes overconfidence | Guo et al. (2017); Kadavath et al. (2022), a temperature of 2.5 for an RLHF policy, beside Jev's 2.66, Nimble's 2.179 and Luce's 2.5 to 3.3 |
| Post-training worsens calibration | GPT-4 report, Fig. 8: MMLU ECE 0.007 to 0.074 after PPO |
| Cross-entropy rewards reporting true probabilities | Gneiting & Raftery (2007): the log score is strictly proper |
| Cheap model first, defer when unsure | Selective classification (2017); FrugalGPT |

The visible contribution is the packaging: a typed API, one hosted endpoint, output priced at zero, and many questions evaluated against one input in a single request. Whether RLCD adds more can be judged once it is published; Jev's native probabilities beating most LLMs' self-reported confidence suggests the training does something.

---

#### What About the Open Alternatives?

Each trained replica is built on Qwen or an encoder, and each publishes its own evaluation.

| Project | Base | Own headline | Independent evidence |
|---|---|---|---|
| Nimble-9B (Bespoke Labs) | Qwen3.5-9B + LoRA | 75.9% vs Jev 77.3% on 13 public sets | none; run by the seller, with raw probabilities (fitted temperature 2.179); its own model trails Jev |
| Kev (Jared Palmer) | Qwen3.5 + LoRA + pointer head | Kev-9B 0.852 test vs Jev 0.857 dev | none |
| Luce (scienthoon) | Qwen3-4B + LoRA | 86.9% vs Jev 84.7% on 500 kubernetes issues | none |
| Laya (Convai) | ModernBERT-large 421M | 0.766 vs 0.727 after training on the benchmark | ikkun1222: 29.2 / 60.0 / 40.0 on three Japanese tasks; the card's own zero-shot 0.362 sits below its 0.461 majority baseline |
| SimpleJev (Featherless) | any HF model; demo serves Gemma 4 26B | no accuracy published | none |
| DiffusionGemma via vLLM PR #57250 | one denoise step of `diffusiongemma-26B-A4B-it` | 198/201 vs Jev 191/201, "roughly tied" | ywchiu: 32.2% vs Jev 61.4% on synthetic routing; companion article: 75.9% vs Jev 77.3% on the public suite |

SimpleJev's README states that its probabilities "are not calibrated probabilities of correctness". The DiffusionGemma headline comes from 201 items built by the PR's author; publishing the items would let others reproduce it.

Labels beat zero-shot wherever they were tried. A 310M Japanese encoder trained on 200 rows beat Jev on news topics by 12 points, 88.8% against 76.8%, and tied it on two sentiment tasks, and fine-tuned models led by 2 to 15 points on five public splits. On phishing, Luce trained on 1,000 labels reached 97.4% against Jev's 62.6% on the same benchmark, though on different items. On that benchmark, a two-line regex scored 91.6%, five narrow Jev questions combined by logistic regression reached 95.0% on a held-out half, and Haiku 4.5 asked the same five questions reached 93.2%, a difference too small to be significant (p = 0.063).

`thejevai.com`, registered on September 20 and promoted on Hugging Face as the endpoint for API keys, has no stated connection to TypeSafe. TypeSafe's API is `api.typesafe.ai`.

---

#### Compare and Contrast

| | Jev | Trained open replica | Plain open model, label scores read |
|---|---|---|---|
| Labels needed | none to run; about 50 to fit a temperature | hundreds to thousands, plus a training run | none to run; about 50 to fit a temperature |
| Accuracy on human labels | level with Kimi K3, MiniMax M3 and DeepSeek V4.1 Flash; 6.5 to 11.5 behind the frontier | trails or matches Jev generically; beats it when trained on the task's own labels | Gemma 4 26B, 4-bit: 75.3% vs Jev 77.3% on the 3,880-record suite; level on yes/no, 4.5 behind on multiple choice |
| Calibration | 🥇 median ECE 0.071 as shipped on the public suite; none published by the vendor | own evaluations | median ECE 0.180 as shipped, 0.080 after 50 labels |
| Where it runs | TypeSafe, OpenRouter, Vercel | your hardware | your hardware |
| Cost | $0.042 per million input tokens; $5.54 per million decisions at 132 tokens | your hardware | at most $5.43 per million decisions on an L4 at full load |

---

#### So, Which One?

For binary and few-class decisions in English, at high volume, with no infrastructure to run and no labels yet, Jev is fast and cheap, its accuracy sits with mid-price LLMs, and its probabilities are the best calibrated out of the box on familiar English tasks. Plan on labelling 50 to a few hundred decisions to set thresholds and fit its probabilities.

For multi-class work with many options, decisions where the rule lives outside the input, or non-English text, where Jev loses accuracy, a model trained on your own labels won wherever one was tried; for non-English that evidence is one Japanese study.

For data that cannot leave your network, any open model read by label scores gives the same typed interface. In the companion article plain Gemma 4 26B trails Jev by 2.1 points overall and 4.5 on multiple choice, and needs about 50 labels to come within 0.01 of Jev's calibration.

---

#### What Is Still Unmeasured?

The published Gemma data points so far:

- Gemma 4 31B beats Jev's median F1 in the pre-registered social-science study, 0.611 against 0.581
- Gemma 4 31B scores 98.86% on SNIPS against Jev's 97.14% after its output's code fences were stripped; 80.57% before
- Gemma 4 26B-A4B trails Jev by 11.6 points on Japanese news topics and ties it on two sentiment tasks
- Gemma 4 E4B, stating its confidence as text, scores 81.0% against Jev's 95.5% on synthetic emails
- Plain Gemma 4 26B read by its label probabilities, in the companion article: on four public tasks (sst2, AG News, DAIR Emotion, tweet_eval irony) and on Bespoke Labs' 3,880-record, 13-subset suite, 2.1 points behind Jev overall

Still unpublished: Jev's behaviour across model versions, calibration across difficulty on real data, and whether a generative LLM shows the same option-name sensitivity.

---

#### Summary

The goal of this article was to establish what independent evidence shows about TypeSafe's Jev and its open alternatives. The key to the solution was tracing every figure to its primary source and re-scoring from published per-item outputs wherever they existed. The results were:

- 🟢 Jev returns a valid option on every call, and its server time is about 0.1 s
- ⚠️ Accuracy sits with mid-price LLMs: 72.5% against 74.5–76.0% for Kimi K3, MiniMax M3 and DeepSeek V4.1 Flash in one six-model study, and a median 11.6 F1 behind the best LLM per task in the largest pre-registered one
- 🟢 Strong on binary and few-class work: 98.33% on 18,514 emails, and level with a dedicated reranker
- 🟢 Best calibrated out of the box on familiar English tasks: median ECE 0.071 on the 13-subset public suite
- 🟢 50 to a few hundred of your own labels and one fitted temperature fix most of its calibration error, and most of an open model's
- ⚠️ Measured speed gains run from 0.5x to 12.1x and cost gains from 0.6x to 478x; TypeSafe's 193.6x and 444.6x have no named comparator
- ⚠️ No calibration metric is published by TypeSafe, and the direction of its calibration error changes by domain
- ⚠️ Swapping option names changed about a third of its answers in one unreplicated study
- ⚠️ Plain Gemma 4 26B read by label probabilities (companion article) trails Jev by 2.1 points on the 3,880-record suite, is level on yes/no, 4.5 behind on multiple choice, and needs 50 labels to come within 0.01 of Jev's calibration
- ❌ RLCD has no published method, and the model's size, base and training data are undisclosed

Scope: sources read on September 23, 2026, eight days after launch: 14 arXiv preprints, 104 GitHub repositories, 33 dev.to and Medium posts, and TypeSafe's own pages; the companion article's public-suite run finished on September 24 UTC. Every accuracy, calibration and timing figure here is a third party's measurement, marked in the full report as re-scored, checked or reported, except the Gemma figures in Compare and Contrast and the Summary, which are the author's own measurement in the companion article; that article's Jev figures come from Bespoke Labs, which sells a competing model. Every run that states a version used `jev-1.13.0`. Sources were gathered and audited with AI assistance (Claude), and every figure was checked against its source, cited in the full report, except LangChain's, marked not re-checked there. The author is a Google Developer Expert and an AWS Community Builder; neither Google nor AWS funded, reviewed or saw this work. The author has no relationship with TypeSafe, Bespoke Labs or any replica reviewed.

The strategy for using primary sources to evaluate Jev was validated with an incremental step by step approach.

---

#### References

- Full evidence review with every source: https://github.com/xbill9/gemma4-dev/blob/main/jev/reports/Jev%20independent%20evidence%20review.md
- Companion measurement article: https://dev.to/gde/plain-gemma-4-26b-vs-jev-on-one-ec2-l4-21-points-behind-overall-level-on-yesno-45-behind-on-15k6
- TypeSafe launch post: https://typesafe.ai/blog/introducing-system-one-models-and-jev
- TypeSafe models and pricing: https://docs.typesafe.ai/models
- TypeSafe evals: https://evals.typesafe.ai/
- Jev 1.13 known weaknesses: https://docs.typesafe.ai/model-jaggedness/jev-1.13
- Ibrahim & Zaki, arXiv 2609.24574: https://arxiv.org/abs/2609.24574
- Rafe & Das, arXiv 2609.24052: https://arxiv.org/abs/2609.24052
- Li et al., arXiv 2609.26550: https://arxiv.org/abs/2609.26550
- Option naming, arXiv 2609.26758: https://arxiv.org/abs/2609.26758
- Janardhan, jev-frontier-bench: https://github.com/manjunathshiva/jev-frontier-bench
- Nimble public benchmarks: https://github.com/bespokelabsai/nimble/blob/main/docs/PUBLIC_BENCHMARKS.md
- jev-phishing-bench: https://github.com/anisselbd/jev-phishing-bench
- Luce: https://github.com/scienthoon/luce
- OmarMujahid, jev-decision-bench: https://github.com/OmarMujahid/jev-decision-bench
- bitnovus, jev-spam-eval: https://github.com/bitnovus/jev-spam-eval
- SamuelSacco, jev-exploration: https://github.com/SamuelSacco/jev-exploration
- sanand0, BANKING77 pilot: https://sanand0.github.io/llmevals/jev/
- heiko-hotz, jev-evaluation: https://github.com/heiko-hotz/jev-evaluation
- ywchiu, jev_benchmark: https://github.com/ywchiu/jev_benchmark
- ikkun1222, Jev against a 310M encoder: https://dev.to/ikkun1222/jev-vs-a-310m-encoder-i-trained-myself-750-rows-three-tasks-two-different-winners-242e
- SimpleJev: https://github.com/featherless-ai/simple-jev
- vLLM PR #57250: https://github.com/vllm-project/vllm/pull/57250
- Matt Mastracci, Jev against DiffusionGemma: https://x.com/mmastrac/status/2100626193943052784
- Guo et al., On Calibration of Modern Neural Networks: https://arxiv.org/abs/1706.04599
- Kadavath et al., Language Models (Mostly) Know What They Know: https://arxiv.org/abs/2207.05221
- Brown et al., GPT-3: https://arxiv.org/abs/2005.14165
- Hendrycks et al., MMLU: https://arxiv.org/abs/2009.03300
- Schick & Schütze, verbalizers: https://arxiv.org/abs/2001.07676
- Zhao et al., Calibrate Before Use: https://arxiv.org/abs/2102.09690
- Gneiting & Raftery, Strictly Proper Scoring Rules, Prediction, and Estimation, Journal of the American Statistical Association 102 (2007)
- Geifman & El-Yaniv, Selective Classification: https://arxiv.org/abs/1705.08500
- Chen et al., FrugalGPT: https://arxiv.org/abs/2305.05176
- GPT-4 technical report: https://arxiv.org/abs/2303.08774
- InstructGPT: https://arxiv.org/abs/2203.02155
- Christiano et al. 2017: https://arxiv.org/abs/1706.03741
