---
title: "What the Independent Evidence Says About Jev, TypeSafe's System One Model"
published: false
description: "A review of every independent measurement of TypeSafe's Jev found in its first eight days: arXiv preprints, GitHub evaluations and blog benchmarks, each traced to its primary source. Accuracy, calibration, speed, cost, failure modes, the prior art, and what is still unmeasured for Gemma."
tags: ai, machinelearning, llm, gemma
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/jev/devto-jev-review-cover.446b3967.jpg
---

This article provides a review of the independent evidence on TypeSafe's Jev, the open models built to replace it, and the prior art behind both, as of September 23, 2026. Every figure below is traced to a primary source, and re-scored from committed per-item outputs wherever the author published them.

https://github.com/xbill9/gemma4-dev/tree/main/jev

---

#### What Is Jev?

Jev is a hosted model from TypeSafe AI, launched on September 15, 2026. It answers typed questions about a piece of text: `choice` picks one option from a list, `score` places the text on an ordered scale, and `noul` returns the probability that a statement is true. It writes no text.

The API is `POST /v1/systemone` at `api.typesafe.ai`. The price is $0.042 per million input tokens, and output is free. Access is through a waitlist, OpenRouter and Vercel's AI Gateway.

TypeSafe calls this category a System One model, after Kahneman's fast and intuitive system. The company raised a $40M seed round led by DCVC, and its co-founders are Diogo Almeida (CEO), Sasha Sheng (COO) and Erik Gafni (CTO).

---

#### How Does a Jev-Style Read Work?

End the prompt where the answer would start. Take the model's scores for only the allowed label tokens. Apply a softmax over those few numbers. One forward pass returns a probability for every option, and the answer is always one of them.

A worked example from Fareed Khan's explainer on Level Up Coding: label scores of 25.28, 24.50 and 21.19 for A, B and C become billing 0.678, technical 0.311 and account 0.011.

Any open model can be read this way. vLLM exposes it through `logprob_token_ids`, SGLang through `/v1/score`, and Featherless AI's SimpleJev wraps it as a Jev-compatible server.

---

#### What Counts as Evidence Here?

The review covers 14 arXiv preprints read in full, 104 GitHub evaluation repositories, 33 dev.to and Medium posts, TypeSafe's own pages and documentation, and press and forum coverage. Each figure carries one of three levels of support.

- **Re-scored:** recomputed from committed per-item outputs, and it matched
- **Checked:** the source's arithmetic recomputed from its published tables, with no per-item data available
- **Reported:** the author's statement, which cannot be checked from outside

Studies are graded A (human or public labels, at least 200 items, stated method, committed outputs), B (real but small, synthetic, LLM-labelled or unreproducible) or C (a demo or a restatement).

The posts split sharply. Of 33 dev.to and Medium posts read in full, 5 are careful measurements on human or public labels, 11 are small, synthetic or scored by agreement between models, and 17 restate vendor claims. Of the 104 GitHub repositories, 33 grade A, 57 grade B, 5 grade C, and 9 are tools. Of the 14 arXiv preprints, 10 call TypeSafe's hosted Jev.

---

#### Where Does Jev Land on Accuracy?

Level with mid-size LLMs, and behind the frontier.

The largest pre-registered study, Ibrahim and Zaki's replication of a social-science annotation suite with 7,977 human-labelled items (arXiv 2609.24574), has Jev behind the best of 19 LLMs on 14 of 15 tasks, by a median 11.6 macro-F1 points. Its per-item release reproduces the paper's figures.

Manjunath Janardhan's 200-item, six-model comparison on BANKING77, BoolQ, Yelp and ChaosNLI re-scores exactly from its committed logs.

| Model | Accuracy | Calibration error (ECE) |
|---|---|---|
| 🥇 Claude Fable 5.1 | 84.0% | 0.064 |
| 🥈 GPT-6 Astra | 79.0% | 0.119 |
| 🥉 DeepSeek | 76.0% | 0.138 |
| MiniMax | 75.5% | 0.112 |
| Kimi | 74.5% | 0.119 |
| Jev | 72.5% | 0.161 |

The two frontier leads (+11.5 and +6.5 points) exclude zero in their 95% intervals. The three mid-size models sit within noise of Jev.

Banking77 is the one task many authors ran independently. Eight runs put Jev at 0.753 to 0.840, with a median of 0.809. In every study that ran LLMs beside it, the best of them was ahead.

Trained baselines beat zero-shot Jev wherever one was run. A 310M Japanese encoder trained on 200 rows scored 88.8% against Jev's 76.8% on news topics, and tied it on two sentiment tasks.

Jev does well on binary and few-class work. On 18,514 emails it scored 98.33% against TF-IDF logistic regression at 98.39%, and held up on newer mail where the trained baseline fell to 72.5%. It tied a dedicated reranker on eight retrieval sets, 0.692 against 0.691.

---

#### Are Its Probabilities Calibrated?

They rank answers usefully, and they need fitting on each task.

TypeSafe publishes no calibration error, reliability plot, Brier score or log loss for Jev on any dataset. Its only confidence-to-accuracy figure is one cookbook of 60 SEC filings: 90% right at confidence 0.9 or above, 40% below.

The independent picture depends on the comparison.

- Against LLMs that write their confidence as a number, Jev's own probabilities usually win. Its median ECE of 0.157 on the social-science tasks beats 16 of 19 LLMs.
- Once every model gets one fitted temperature, 15 of those LLMs beat it.
- Against LLMs that return full probability distributions, Jev's calibration error is the highest in Janardhan's run (0.161) and in sanand0's nine-model Banking77 pilot (0.138).

The direction of the error changes with the data. On public multi-class sets Jev is overconfident: on GoEmotions, labels it scored between 0.80 and 0.95 matched the human label 15% of the time. On crash narratives and synthetic items it is under-confident. Refit temperatures across studies run from 0.65 to 4.45, and the judge study's authors conclude that no single temperature fits.

The API also rounds every probability to 0.01. In one committed sample, 70.4% of `choice` probabilities came back as exactly 0, and `noul` is clamped between 0.01 and 0.98.

---

#### 🔎 Tip: Fit a Temperature on a Few Hundred of Your Own Labels

A small refit fixes most of the calibration error, for Jev and for any open model read the same way.

- A two-parameter refit on civil_comments took Jev's ECE from 0.16–0.21 to under 0.025 on a held-out half
- Transferring a fitted slope and refitting only the intercept on 50 labels cut mean ECE by 74%
- Out-of-fold Platt scaling cut Jev's ECE on crash narratives by 3.35x

The open replicas report the same order of correction: Nimble fitted 2.179, Luce 2.5 to 3.3 on unseen tasks. Luce's rule is to ship with 100 to 300 real labelled items and fit the temperature on them.

---

#### How Much Faster and Cheaper Is It?

Anywhere from 0.5x, slower than a local Gemma, to 478x cheaper, depending on the comparison model, the network path and where the clock runs.

| Comparison | Source | Faster by | Cheaper by |
|---|---|---|---|
| GPT-5.6 Luna, reasoning off, client clock | OmarMujahid | 1.04x | 4.3x |
| Haiku 4.5, phishing verdict | jev-phishing-bench | 2.9x | 12x |
| Mid-size open model, 47 account reviews | devopsdaily | 12.1x | 7.4x |
| Claude Fable 5.1, via OpenRouter | Janardhan | 9.9x | 478x |
| 19 LLMs, social-science tasks (median) | arXiv 2609.24574 | — | 18.6x |
| DeepSeek, 6G edge orchestration | arXiv 2609.23136 | 1.3x | 0.6x (Jev dearer) |
| Local Gemma 4 26B-A4B Q4 on a mini-PC | ikkun1222 | 0.5–0.65x (Jev slower) | — |

The server itself is fast: about 105 ms in one run, and about 76 ms once the network round trip is subtracted in another. Most of the cost gap comes from output being free. In devopsdaily's workload 83% of the old bill was output, and per input token the two prices differed by 1.31x.

---

#### Where Does It Break?

On wording, language, and questions whose answer is absent from the input.

Option names move answers. With the question, input, rubric and option set held fixed, swapping which rubric sits behind "no" and "yes" changed 32.5% of Jev's answers, against about 2% with neutral names, and AUROC fell from 0.81 to 0.58 (arXiv 2609.26758, n = 1,200). No generative LLM was given the same swap.

Bare labels mislead it. A router given option names with no descriptions sent all 40 hard tasks to the cheap model, at a median confidence of 0.96. One-line option descriptions fixed 37 of 40.

Language shift costs accuracy. Russian XNLI dropped from 88.3% to 77.3% with ECE tripling, and Spanish cost 3 to 6 points. German cost 0.5 points on MASSIVE.

Jev is most confidently wrong where the answer does not follow from the input. On bias questions with no "unknown" option it scored 0.000 accuracy at 0.79 confidence. On random outcomes it reported 83% where the true rate was 17%. On heart-risk prediction its probabilities ran about 3x too high.

Questions batched in one request cannot see each other's answers. One reviewer's request returned "suspend" for an account the same response classified as a developer sending tests.

TypeSafe's own list of known weaknesses names literal reading, arithmetic and counting, dates, long irrelevant input, adversarial content and phrasing: "refund" scored 0.72 and "not a refund" 0.47 on the same ticket.

---

#### What Does TypeSafe Claim, Checked?

**193.6x faster and 444.6x cheaper.** The source is TypeSafe's four in-house workflow evals, which the launch post calls "on the higher end of real world gains". The comparator is never named. Checked against the published summary points, 444.6x fits only Jev against Opus 5 in the workflow setup. Averaged over all eight workflow setups, the figures are 97.8x faster and 149.2x cheaper. Timings came from laptops on the US West Coast, with no case counts, run counts or variance.

**Eval accuracy.** Reference labels are "an average of the responses of GPT-6 Astra and Claude Fable 5.1, both at high thinking", so accuracy there means agreement with two LLMs. Jev scores 67.8%, equal to Sonnet 5, with GPT-5.6 Sol leading at 74.1%. On invoice processing Jev scores 61.8% against Sol's 79.1%. TypeSafe states that it chose to publish no public-benchmark results.

**"0% hallucination".** TypeSafe's launch post: "Our number is not empirical. Schema matching is guaranteed." Every answer is a valid option; the wrong-but-valid rate is the accuracy figures above.

**RLCD.** Reinforcement learning for calibrated decisions has no paper, patent or method description. The CEO said on Hacker News that the architecture is "close to the chest for now, but we have talked about writing a paper." On Latent Space he agreed that all of Jev's training data is synthetic. Model size and base are undisclosed.

**Pricing.** The launch post says "We can't prove it isn't subsidized." The home-page FAQ says "We can serve Jev profitably at our current prices."

**The founder credential.** TypeSafe's pages say Almeida co-invented RLHF. He is fourth of 20 authors on InstructGPT (arXiv 2203.02155), marked as a primary author, and an author of the GPT-4 report. He is not an author of Christiano et al. 2017 (arXiv 1706.03741), the paper that introduced deep reinforcement learning from human preferences, and on Latent Space he separated that line of work from the instruction-following work he means.

**The CEO's own summary.** To a Hacker News commenter's "basically a zero-shot classifier", he replied "exactly right!"

---

#### What Predates Jev?

Every piece of the mechanism.

| Piece | Prior art |
|---|---|
| Score options by label likelihood | GPT-3 §2.4 (2020); MMLU §4.1 (2021); lm-evaluation-harness `multiple_choice` |
| Map each label to one token | Verbalizers, Schick & Schütze (arXiv 2001.07676) |
| Label probabilities carry prior bias | Calibrate Before Use, Zhao et al. (2021) |
| One fitted temperature fixes overconfidence | Guo et al. (2017), arXiv 1706.04599 |
| Post-training worsens calibration | GPT-4 report, Fig. 8: MMLU ECE 0.007 to 0.074 after PPO |
| Cross-entropy rewards reporting true probabilities | Gneiting & Raftery (2007): the log score is strictly proper |
| Cheap model first, defer when unsure | Selective classification (2017); FrugalGPT |

Kadavath et al. (2022) found that one temperature of 2.5 largely fixes an RLHF policy's miscalibration. The temperatures reported for Jev and its replicas fall in the same range.

TypeSafe's contribution is the packaging: a typed API, one hosted endpoint, output priced at zero, and many questions evaluated against one input in a single request.

---

#### What About the Open Alternatives?

Each trained replica is built on Qwen or an encoder, and each publishes its own evaluation.

| Project | Base | Own headline | Independent evidence |
|---|---|---|---|
| Nimble-9B (Bespoke Labs) | Qwen3.5-9B + LoRA | 75.9% vs Jev 77.3% on 13 public sets | none; run by the seller, at temperature 1.0 |
| Kev (Jared Palmer) | Qwen3.5 + LoRA + pointer head | Kev-9B 0.852 test vs Jev 0.857 dev | none |
| Luce (scienthoon) | Qwen3-4B + LoRA | 86.9% vs Jev 84.7% on 500 kubernetes issues | none |
| Laya (Convai) | ModernBERT-large 421M | 0.766 vs 0.727 after training on the benchmark | 0.362 zero-shot, below the 0.461 majority baseline |
| SimpleJev (Featherless) | any HF model; demo serves Gemma 4 26B | no accuracy published | none |
| DiffusionGemma via vLLM PR #57250 | one denoise step of `diffusiongemma-26B-A4B-it` | 198/201 vs Jev 191/201 | one synthetic test: 32.2% vs Jev 61.4% |

SimpleJev's README states that its probabilities "are not calibrated probabilities of correctness". The DiffusionGemma result comes from 201 items built by the PR's author, with no items or code published.

The labelled-data result is consistent across authors. On phishing, Luce trained on 1,000 labels reached 97.4% against Jev's 62.6% on the same benchmark. On that benchmark, a two-line regex scored 91.6%, five narrow Jev questions combined by logistic regression reached 95.0% on a held-out half, and Haiku 4.5 asked the same five questions reached 93.2%, a gap within the run-to-run noise (p = 0.063).

---

#### Who Ran Each Study?

The studies with no identifiable stake produce the least favourable accuracy results or the most qualified favourable ones: Ibrahim and Zaki, Rafe and Das, Janardhan, sanand0, SamuelSacco and OmarMujahid.

Favourable figures come more often from parties with a stake. Every's review was written by a launch partner. LangChain, which ships the `langchain-typesafe` integration, reported 100% on 500 judgments that are 5 test cases repeated 100 times. The "15.9% faster pipeline" headline came from a TypeSafe employee's three-case demo.

Undisclosed interests appear on both sides. synthorai, which found Jev behind flash LLMs, runs a gateway selling the rival models and timed them through its own proxy. A SNIPS study where Jev narrowly trails Gemini is by a Google employee. Bespoke Labs sells Nimble. The replica authors test their own models. Matt Mastracci wrote the DiffusionGemma PR he compares with Jev.

`thejevai.com`, registered on September 20 and promoted on Hugging Face as the endpoint for API keys, is unaffiliated with TypeSafe. TypeSafe's API is `api.typesafe.ai`.

---

#### What Circulated, and What the Sources Show

| Claim as circulated | Primary source |
|---|---|
| "Its own employee measured 15.9% on a real pipeline" (Cherry Creek News) | 3 demo cases; 2.329 s / 1.958 s = 1.19x |
| 194x faster and 445x cheaper against GPT-6 Astra (Tom's Hardware) | Astra is a reference labeller with no scored setup; 444.6x fits Opus 5 |
| Jev "cannot hallucinate" (TechCrunch) | TypeSafe calls its 0% "not empirical" |
| Jev flips 70.4 of 100 answers when options are renamed | That figure is an open ModernBERT model's; Jev's is 32.5% |
| Jev matched or beat Luna on 42 of 49 tasks, at a seventh of the latency | 43 under its own tie rule; 6 leads outside the noise; 975 vs 1,018.5 ms on the client clock |
| 0 wrong verdicts in 1,080 rounds, error bound 0.28% | 9 scenarios repeated 120 times; counted by scenario, the bound is 33% |
| Jev needed a refit temperature of 2.74 on an unseen task | Author's correction: 1.30 for `choice`, 1.92 for `score` |
| Jev's ECE is 0.246 against Laya's 0.081 (Laya card) | 0.246 unsourced; the same card lists Jev at 0.144, and compares refit Laya with raw Jev |

---

#### Compare and Contrast

| | Jev | Trained open replica | Plain open model, label scores read |
|---|---|---|---|
| Labels needed | 🥇 none to run; a few hundred to trust its probabilities | 🥉 hundreds to thousands, plus a training run | 🥈 none to run; a few hundred to fit a temperature |
| Accuracy on human labels | mid-size LLM level | beats Jev where labels carry the rule | Gemma 4 31B generative: 0.611 vs Jev 0.581 median F1 |
| Calibration published | none by the vendor | own evaluations | none for Gemma |
| Where it runs | TypeSafe, OpenRouter, Vercel | your hardware | your hardware |
| Cost | $0.042 per million input tokens | your hardware | your hardware |

---

#### So, Which One?

For binary and few-class decisions in English, at high volume, with no infrastructure to run, Jev is fast and cheap, and its accuracy sits with mid-size LLMs. Plan on labelling a few hundred decisions to set thresholds and fit its probabilities.

For multi-class work with many options, non-English text, or decisions where the rule lives outside the input, a model trained on your own labels wins on the current evidence.

For data that cannot leave your network, any open model read by label scores gives the same typed interface. The open question is how much fitting it needs.

---

#### What Is Still Unmeasured?

For Gemma, the largest gap. The published Gemma data points are all generative read-outs, where the model writes its answer and a parser reads it.

- Gemma 4 31B beats Jev's median F1 in the pre-registered social-science study, 0.611 against 0.581
- Gemma 4 31B scores 98.86% on SNIPS against Jev's 97.14%, after stripping code fences from its output
- Gemma 4 26B-A4B ties Jev on two Japanese sentiment tasks
- Gemma 4 E4B, stating its confidence as text, scores 81.0% against Jev's 95.5% on synthetic emails

Plain Gemma read by its label probabilities, with calibration measured, is unpublished. So are DiffusionGemma's calibration, which Google's Gemma account has described as well calibrated, Jev's behaviour across model versions, calibration across difficulty on real data, and whether a generative LLM shows the same option-name sensitivity.

---

#### Summary

The goal of this article was to establish what independent evidence shows about TypeSafe's Jev and its open alternatives. The key to the solution was tracing every figure to its primary source and re-scoring from published per-item outputs wherever they existed. The results were:

- 🟢 Jev returns a valid option on every call, and its server time is about 0.1 s
- 🟢 Accuracy sits with mid-size LLMs: 72.5% against a 74.5–76.0% mid-size range in one six-model study, and a median 11.6 F1 behind the best LLM in the largest pre-registered one
- 🟢 Strong on binary and few-class work: 98.33% on 18,514 emails, and level with a dedicated reranker
- 🟢 A few hundred of your own labels and one fitted temperature fix most of its calibration error
- ⚠️ Measured speed and cost gains run from 0.5x to 478x; TypeSafe's 193.6x and 444.6x have no named comparator
- ⚠️ No calibration metric is published by TypeSafe, and the direction of its calibration error changes by domain
- ⚠️ Swapping option names changes about a third of its answers
- ❌ RLCD has no published method, and the model's size, base and training data are undisclosed
- ❌ Plain Gemma read by label probabilities, with calibration measured, has no published result

Scope: sources read on September 23, 2026, eight days after launch: 14 arXiv preprints, 104 GitHub repositories, 33 dev.to and Medium posts, and TypeSafe's own pages. This review runs no model; every accuracy, calibration and timing figure is a third party's measurement, and each is marked in the full report as re-scored, checked or reported. Every independent run used `jev-1.13.0`.

The strategy for using primary sources to evaluate Jev was validated with an incremental step by step approach.

---

#### References

- Full evidence review and per-source audits: https://github.com/xbill9/gemma4-dev/tree/main/jev
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
- SimpleJev: https://github.com/featherless-ai/simple-jev
- vLLM PR #57250: https://github.com/vllm-project/vllm/pull/57250
- Guo et al., On Calibration of Modern Neural Networks: https://arxiv.org/abs/1706.04599
- Schick & Schütze, verbalizers: https://arxiv.org/abs/2001.07676
- Zhao et al., Calibrate Before Use: https://arxiv.org/abs/2102.09690
- GPT-4 technical report: https://arxiv.org/abs/2303.08774
- InstructGPT: https://arxiv.org/abs/2203.02155
- Christiano et al. 2017: https://arxiv.org/abs/1706.03741
