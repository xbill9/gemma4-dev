# Jev and the "System One" wave: what's old, what's new, what's measured

Internal note, 2026-09-23. Written from the sources listed at the bottom, read on this date. Every number below is quoted from a source or computed in a script from a source's published figures, and says which. The headline figures of the independent studies were re-checked against their primary sources (repos, raw logs, post bodies) on 2026-09-23; audit notes, commit SHAs and recomputation scripts are kept privately in `research_notes/` (gitignored) and are not published. The 14 arXiv preprints were read in full. The publication-grade write-up of the same evidence is `reports/Jev independent evidence review.md`; where the two differ, the report is the checked version.

## Bottom line

- **Jev is a fast, cheap, mid-tier classifier.** On human-labelled public data it scores level with mid-size LLMs (Kimi, MiniMax, DeepSeek) and 6.5–11.5 points below frontier ones. In the largest pre-registered study (arXiv 2609.24574, 7,977 human-labelled items) it trails the best LLM on 14 of 15 tasks by a median 11.6 F1. On Banking77, eight independent studies put it at 0.753–0.840 (median 0.809), below the LLMs run beside it. Supervised baselines beat zero-shot Jev wherever one was run.
- **Speed and cost depend on what is timed and priced:** service time is about 0.1 s; measured gains run from parity to 478x. Client wall-clock against a reasoning-off LLM was 1.04x; Jev was slower than a local Gemma 4 26B and than Luna on long contexts, near parity on cost with DeepSeek on Banking77, and dearer in one edge-orchestration paper (1.7x). The large cost gaps come mostly from free output against reasoning models. TypeSafe's 193.6x / 444.6x have no stated comparator; averaged over all eight of TypeSafe's own published workflow setups they are 97.8x faster and 149.2x cheaper.
- **Its probabilities need fitting on each task.** Against LLMs that return full probability distributions, its calibration error is at or near the worst (Janardhan, sanand0, phishing); against LLMs that write their confidence as text it is usually better. The direction of error flips by domain: overconfident on public multi-class data, under-confident on crash narratives and synthetic items; refit temperatures across studies run 0.65–4.45. Ranking quality (AUROC) runs 0.56–0.998: 0.97–0.99 on binary and spam tasks, 0.69–0.81 on the phishing verdict and Banking77. One run of a difficulty gradient (800 synthetic items) found a single fixed miscalibration curve, compressed toward the middle, in every tier. Good calibration figures come from synthetic or author-built data, or from public datasets that may overlap its training data. TypeSafe publishes no calibration metric.
- **Good ranking with poor calibration is what a few of your own labels fix.** Independent refits bring calibration error under 0.025; transferring a fitted slope plus a 50-label intercept cut it 74%. Every strong result found used the task's own labels.
- **RLCD's method is undisclosed.** No paper, patent or write-up exists; the CEO has said the architecture is kept "close to the chest" and that they "have talked about writing a paper".

Of 33 dev.to and Medium posts read in full, 5 are careful measurements on human or public labels, 11 are real measurements that are small, synthetic, or scored by agreement between models, and 17 restate vendor claims. Of 104 further evaluation repos found on GitHub and awesome-Jev lists, 33 grade A, 57 grade B, 5 grade C, and 9 are not evaluations.

## The mechanism in one line

End the prompt where the answer would start, take the model's logits for only the allowed label tokens, and softmax over those. One forward pass, no generated text. The answer is always one of the declared options; whether it is the right one is a separate question.

Worked example (Fareed Khan, Level Up Coding): logits 25.28 / 24.50 / 21.19 for A/B/C → softmax over the three → billing 0.678, technical 0.311, account 0.011.

## What's old

Citations verified against the primary papers.

| Piece | Prior art |
|---|---|
| Score a fixed answer set by label likelihood | GPT-3 (Brown et al., 2020) §2.4 compares "the LM likelihood of each completion"; MMLU (Hendrycks et al., 2021) §4.1 reads the probabilities of "A"–"D"; lm-evaluation-harness `output_type: multiple_choice` |
| Map labels to single tokens, read their probabilities | Verbalizers, introduced in Schick & Schütze, arXiv:2001.07676 (EACL 2021); multi-token labels in arXiv:2009.07118 (NAACL 2021) |
| Masked-token prediction; zero-shot classification as entailment | BERT (Devlin et al., NAACL 2019); Yin, Hay & Roth (EMNLP 2019) |
| Label probabilities carry a prior bias that needs removing | "Calibrate Before Use" (Zhao et al., ICML 2021): a content-free input corrects label-prior bias. It measures accuracy and variance, not calibration error |
| Modern networks are overconfident; one fitted temperature fixes most of it | Guo et al., ICML 2017 (arXiv:1706.04599) §4.2: temperature scaling "does not affect the model's accuracy" |
| Post-training worsens calibration | GPT-4 technical report, Fig. 8: MMLU ECE 0.007 pre-trained against 0.074 after PPO. Kadavath et al., 2022 §3.3: RLHF policies "appear very miscalibrated"; one temperature (2.5) "largely fixes" it |
| Training with a proper scoring rule rewards honest probabilities | Gneiting & Raftery, JASA 2007: the log score is strictly proper. Cross-entropy training is the log score. Guo §3 is the caveat that this does not guarantee held-out calibration |
| Cheap classifier first, expensive model for the unsure cases | Selective classification (Geifman & El-Yaniv, NeurIPS 2017); FrugalGPT (Chen, Zaharia & Zou, TMLR) |

## What's new

- **Packaging.** TypeSafe's Jev (launched 2026-09-15): a typed API (`choice`, `score`, `noul`), `POST /v1/systemone`, $0.042 per million input tokens, output free, waitlisted early access. Official endpoint `api.typesafe.ai`, docs `docs.typesafe.ai`. The API returns probabilities rounded to 0.01 without saying so: 70% of choice probabilities come back as exactly 0, and `noul` is clamped to 0.01–0.98 (SamuelSacco/jev-exploration#10).
- **RLCD**, TypeSafe's name for how Jev is trained. TypeSafe's pages expand it as "reinforcement learning *for* calibrated decisions"; TechCrunch writes "*from*". No paper, training details or calibration figures from TypeSafe exist; Latent Space calls it "a novel, unpublished technique", and on that podcast Almeida describes it as a new task, "another North Star", rather than an algorithm. The only public description of an RLCD-style objective is Laya's own: a proper-scoring-rule reward with Gaussian exploration noise on the logits, close to ordinary cross-entropy training in RL vocabulary.
- **Training data.** TypeSafe: "We make all the data ourselves." On Latent Space, asked whether all their data is synthetic, Almeida answered "Yep". TechCrunch's "trained exclusively on synthetic data" is the reporter's paraphrase. Size and architecture are undisclosed; TypeSafe calls Jev "neither small nor an LLM".
- **The founder credential.** TypeSafe's docs say RLHF "was co-invented by Diogo Almeida"; its team page says "co-invented RLHF and InstructGPT"; its press release "co-inventor of RLHF/ChatGPT". Almeida is not an author of Christiano et al., 2017 (arXiv:1706.03741: Christiano, Leike, Brown, Martic, Legg, Amodei). He is 4th of 20 authors of InstructGPT (Ouyang et al., 2022, arXiv:2203.02155), marked as a primary author, and is on the GPT-4 report's author list. On Latent Space (2026-09-21) he separates the robot-backflip and learning-to-summarize RLHF, "which I did not co-author", from the instruction-following work he means.
- **DiffusionGemma as a reader.** vLLM PR #57250 (Matt Mastracci, merged 2026-09-22T20:51Z) adds a one-step read-only mode to `google/diffusiongemma-26B-A4B-it`: seed the canvas with the answer template, put random vocabulary tokens in the answer slots, run one denoise step, return the logits at the slots. The PR's `structured_server.py` implements Jev's API shape. It is vendored in `vendor/` at merge commit `1b3b88ec`.

## What's measured

### Independent studies on human or public labels (grade A)

| Study | Data and labels | n | Jev | Comparators | Calibration |
|---|---|---|---|---|---|
| Janardhan (Medium, Data Science Collective; repo `manjunathshiva/jev-frontier-bench` @ `da39111c`) | BANKING77, BoolQ, Yelp stars, ChaosNLI; human labels | 200 items, 6 models, 1 run; raw logs published and re-scored | **72.5%** (95% CI 66–78.5) | Fable 5.1 84.0, GPT-6 Astra 79.0, DeepSeek 76.0, MiniMax 75.5, Kimi 74.5 | **Jev ECE 0.161, worst of six** (Fable 0.064) |
| ikkun1222, Jev vs trained encoder (dev.to) | 3 Japanese public datasets, dataset labels | 250 per task | 76.8 / 94.4 / 74.0 | ModernBERT-ja-310m trained on 200 rows 88.8 / 92.8 / 75.2; Gemma 4 26B-A4B local Q4_K_M 65.2 / 94.4 / 75.2 (runtime and scoring method not stated); Laya 29.2 / 60.0 / 40.0 | not reported for Jev |
| ikkun1222, one question vs 12 dimensions (dev.to) | style (synthetic), ledger (own), injection sets, Japanese NLI (human) | 25,174 calls | 12-way ledger choice **0.40**; as scored dimensions 0.91 | word bigrams 0.949 on the ledger | at confidence ≥0.9, **72.2% right** (91/126, synthetic style task) |
| aitejiu, agent harness (dev.to; ships jev-mcp; repo `Aitejiu/jev-harness-lab` @ `29ddf82`) | 10 public datasets | 17,616 result rows (post says ~22,500 calls) | SNIPS 97.9, Banking77 80.3 (92.8% on the 67.8% at ≥0.9), Who&When AUROC 0.56 | — | wrong routings at confidence 1.0; several thresholds chosen on the test rows |
| synthorai, Jev vs other LLMs (dev.to; runs an API gateway for the rival models, undisclosed, and timed them through it) | Banking77, GoEmotions (human), deepset injection | 308 / 200 / 116, 1 run | Banking77 80.2, GoEmotions 0.228 (lowest) | 70.8–87.3 on Banking77 (Seed 2.0 Mini to GPT-5.6 Terra) | GoEmotions: scored 0.80–0.95, **15% right**; ranking AUC 0.98–0.99 |
| OmarMujahid/jev-decision-bench | 49 tasks, mostly public benchmarks | 8,225 outputs per model, committed; reproduces its report | leads Luna on 29, ties 14, trails 6 under its own ±0.005 tie rule (it claims 42 of 49) | GPT-5.6 Luna with reasoning off or low; LLM confidence written as text | Jev mean ECE 0.070 against Luna 0.178. Only 6 of 29 leads have non-overlapping 95% intervals; the big leads are on the most exposed public sets (MMLU, ARC, HellaSwag, WinoGrande, TruthfulQA); the one fresh-data control (new maths) went to Luna. "A seventh of the latency" is server-header time; client wall-clock medians are 975 against 1,018.5 ms |
| bitnovus/jev-spam-eval | 18,514 emails | recomputed | 98.33% | TF-IDF 98.39%; enriched rerun 98.64% against TF-IDF 98.87% | ECE 0.0505, Brier 0.0174; prompt wording tuned on labelled errors (stated) |
| Aman Kumar (amankumar.ai) | Enron, SST-2, AG News, Banking77; human labels; plus private outcome data | 300 per set | ahead of small GPTs on three; Banking77 76% | small GPTs ~80% on Banking77 | verdict: "a filter, not a replacement" |

Across the grade-A studies: ranking quality varies by task (AUROC 0.56–0.998), stated probabilities overstate certainty on data it was not built around, accuracy drops and calibration error roughly doubles in Russian and Spanish, and it is badly miscalibrated on bias questions with no abstain option (KoBBQ: 0% accuracy, ECE 0.79), random outcomes, and risk prediction (probabilities about 3x too high). It ties dedicated rerankers (0.692 against Cohere 0.691). Yes/no questions are its strongest type; on BoolQ in Janardhan every model scored 90–96 and Jev is within noise of all of them.

Janardhan also reports: on ChaosNLI (100 human labels per item) Jev's probabilities matched the human split worse than a flat one-third guess (Jensen-Shannon 0.149 against 0.127); a cascade taking Jev at ≥0.9 and sending the rest to Fable scored 82.5% at 37% of Fable's cost, with the threshold chosen after seeing the data; a 2026-09-20 rerun of 20 items gave OpenRouter median 384 ms against 874 ms direct, 19 of 20 answers agreeing. His cost multiple is 478x (11.81 / 0.0247).

sanand0's BANKING77 pilot (n=77, one per intent): Jev 75.3%, lowest of 9 models, with the highest ECE (0.138); cheapest of the nine ($0.0717 per 1,000, DeepSeek next at $0.0849).

### Phishing, and open recipes that beat Jev with labels

- **jev-phishing-bench** (anisselbd): 2,000 emails, PhishNChips v5.2; bodies are synthetic LLM text, labels from URL reputation feeds; per-item outputs not published.
  - Single verdict: Jev 62.6% against Haiku 4.5 81.3% (Jev ECE 0.154). Here Jev is 2.9x faster and 12x cheaper.
  - A hand-written regex: 91.6% on all 2,000, 91.8% on the 1,000-email test half.
  - Five narrow Jev questions combined by a logistic regression: 95.1% (AUROC 0.988, ECE 0.027) by five-fold cross-validation on all 2,000 with the verdict probability as a sixth input; **95.0% (AUROC 0.982, ECE 0.024) fitted on 1,000 and tested on the other 1,000**. Haiku asked the same five questions: 93.2% (p = 0.063, not significant), with the higher AUROC. In this comparison Jev is 5x faster and 27x cheaper.
- **Luce** (scienthoon, created 2026-09-20, written by Luce's author): Qwen3-4B-Base + LoRA + decision head, temperature fitted on real labels. Its Jev raw logs are not in the repo tree.
  - Phishing, trained on 1,000 labels: 97.4%, ECE 0.010 (Jev 62.6 from the phishing bench's full set, not item-matched).
  - Rule-generated tickets: 91.1%, ECE 0.022; Jev's 75.1 is on a different 900-item validation set. Luce's training labels were written by an LLM teacher (DeepSeek V4.1 Flash).
  - GitHub issue type, kubernetes maintainer labels, 500 labels: 86.9% against Jev 84.7%. The only item-matched comparison on human labels; Jev answered 478 of the 500 (22 exceeded max_tokens), and 86.9 comes from a later retrain while its ECE 0.044 matches the earlier run (85.9%).
  - On tasks it was not trained on, Luce needed a refit temperature of 2.5–3.3. Its README still prints Jev's 2.74 on an unseen ticket task; the author partly retracted that on 2026-09-22 (jev-ood-calibration @ `914d87ab`): with a sensible floor for exact zeros, Jev's refit temperature is 1.30 for Choice and 1.92 for Score, still overconfident but less so.
  - Its rule: ship with 100–300 real labelled items and fit the temperature on them.
- **Kev** (Jared Palmer), retrained 2026-09-21: Kev-9B 0.822 (dev) / 0.852 (test) on unseen sources, against Jev's 0.857 (dev only).

### Jev on public data, by Nimble (a competitor; possible overlap with training data)

Bespoke Labs, which sells the competing Nimble-9B, ran Jev 1.13.0 and Nimble on 13 public human-labelled datasets, 3,880 records (`bespokelabsai/nimble`, `docs/PUBLIC_BENCHMARKS.md`, contributed by Edgar Dyck). Per-record outputs are not committed.

- Accuracy, pooled over records: Jev 77.3%, Nimble 75.9% (macro 76.0 against 74.8).
- By question type (macro): Jev leads on yes/no (84.6% against 80.2%) and slightly on choice (82.9% against 81.6%); Nimble leads on rating scales (54.6% against 50.1%).
- Calibration error: Jev lower on 11 of 13 datasets; 0.038–0.075 on most yes/no and choice sets, 0.23–0.26 on two of the three rating sets. Nimble ran uncalibrated (temperature 1.0) in this comparison, which flatters Jev's 11 of 13.
- English → German on the same 350 utterances: Nimble −3.5 points, Jev −0.5.
- Jev's probabilities arrive rounded to 2 decimals, so its log loss is not comparable.

Read with the grade-A results above: calibrated on familiar ground.

### Calibration across difficulty

SamuelSacco/jev-exploration#1 designed a four-tier test (trivial, ordinary, hard, adversarial). It was run on 2026-09-18 with 800 synthetic items and 3 passes, and reproduces offline: one fixed miscalibration curve compressed toward the middle, in every tier, with precision 1.000 at p ≥ 0.9 in every tier. The cross-domain transfer experiment (#9) and a head-to-head against a local model are built but not run.

### Smaller measurements (grade B)

- **devopsdaily** (dev.to), 50 real accounts, real admin decisions, 47 complete timing pairs: median 605 ms against 7,326 ms (12.1x) for a mid-size open model on a serverless endpoint; about 7x cheaper per 1,000 reviews, 83% of the old bill being output. No accuracy result survived (only 2 fair labelled cases). Linked questions in one request produced incoherent pairs. No Jev version given; harness not linked.
- **copyleftdev** (dev.to; repo `copyleftdev/jev-labs` @ `66933dd`), TLA+ pharmacy, 14 synthetic scenarios the author labelled: 0 wrong verdicts in 1,080 easy rounds covering 9 scenarios repeated 120 times each. Jev alone on 240 constructed items: 0.979 accuracy, Brier 0.0187, ECE 0.0752 (ECE from the repo). Observed largest deviations across identical, reordered and paraphrased requests: 0.02 / 0.03 / 0.04.
- **kunko judge-audit** (dev.to; repo @ `665ee4f`), synthetic seeded data: clean emails 200/200 (ECE 0.0036), adversarial 95.5% (ECE 0.039). A task router with bare option labels sent all 120 tasks to the easy model; on the 40 hard tasks its confidence ran 0.56–1.00 (median 0.96). One-line option descriptions fixed 37 of 40. The repo's later comparison page (`docs/arena-2026-09.md`), same data: Gemini 3 Flash 97.0%, Gemma 4 E4B 81.0%. Gemma there is `gemma4:e4b` on Ollama at temperature 0, its confidence a number written in its reply rather than a logit; on the bare-label router it sent 28 of 40 hard tasks to the strong model against Jev's 0.
- **themsquared/jev-benchmark** (n=60): 55/60, with two wrong answers at confidence 0.97 and 0.98.
- **Near Here** (Jon Reed; had early access), 50 real listings with labels "written by the assistant": Jev 48/50 on the set used to choose prompts. The 21 "held-out" listings were inspected before labelling (the author says it was not blind) and hold 2 approvals: Jev 19 = Mistral 19 < Gemini 20.
- **Paweł Józefiak** (thoughts.jock.pl; sells a related kit): 40 self-labelled tickets, 39/40, level with Haiku.
- **LessWrong trusted-monitor post** (Ventak T): AUROC 0.976 and 0.970, about 90% of backdoors caught at 2% false positives, about $0.04 per 1,000 submissions; a best-of-5 attack cut the catch rate from 90% to 60%.
- **Every** (Mike Taylor, a launch partner): 777 judgments (37 × 21) in under 0.7 s; speed only, no accuracy. Dan Shipper: 12 synthetic passages, 6 of 7 planted defects caught against Fable's 7 of 7, 0.35 s against 8.83 s.
- **OpenPoke** (0xshin0221): 5.8x faster than Claude Sonnet 4 on 12 emails; agreement, no labels.
- **Robot fleet** (chorylee): 300 simulated incidents, team 91.3% against template labels, p50 0.53 s, $0.00737 total.
- **Ariful Islam** (dev.to; helpdesk vendor ad): 100 synthetic tickets, Jev's accuracy not broken out; "calibration" measured as agreement with other models.
- **Joe Njenga** (Medium; sells a course): a few playground calls. "Are you a bot?" scored 0.26 against the 0.40 in TypeSafe's docs; the other two documented examples matched.
- **The "15.9%" story.** Cherry Creek News (syndicated, word for word on the North Denver Tribune): "Its Own Employee Measured 15.9% on a Real Pipeline." The source is `typesafeainate/dspy-typesafeify` by Nathan LeClaire, a TypeSafe employee: three cases of a demo support-ticket example, 2.329 s against 1.958 s, which is 1.19x end to end; the 30.1% cost cut is modelled from a price the README marks as a "supplied estimate".
- **JevBench**: run 001 Banking77 80.3% on 3,080 items; run 003 94% against AI-written labels. By its own rules its runs are not independent evidence, and its artifact repo returns 404.

No source measured Jev's answers changing over time. The one post claiming it (jev-watch @ `a0ddd8f`) made its failing example by editing an expected value on purpose, and its 20,000-case benchmark never calls Jev.

### TypeSafe's own claims

- **193.6x / 444.6x**: from TypeSafe's four in-house workflow evals (evals.typesafe.ai), "on the higher end of real world gains". The comparator is never stated. 444.6x fits only Jev against Opus 5 in the workflow setup; 193.6x fits several comparisons. Averaged over all eight workflow setups: 97.8x faster, 149.2x cheaper. Workflows written by TypeSafe's team, timed from laptops on the US West Coast, LLMs run through TypeSafe's own wrapper; no case counts, run counts, variance or Jev version published.
- **Its speed claims differ across its own releases**: press release "under 100 ms, up to 100 times faster and less expensive"; founder's X post 20–200x faster, 40–400x cheaper; launch post 70–500 ms, 40–200x faster; home page 193.6x / 444.6x.
- **Reference labels**: "an average of the responses of GPT-6 Astra and Claude Fable 5.1, both at high thinking"; the CEO confirmed on HN "we actually use astra (and fable) … for our evals". Accuracy there means agreement with two LLMs. Jev 67.8% overall, equal to Sonnet 5 and just under GPT-5.6 Terra (67.9%); Sol leads at 74.1%. Invoice processing: Jev 61.8% against Sol 79.1%; only Haiku 4.5 scores lower.
- **Calibration**: no ECE, reliability plot, Brier score or log loss published. The only confidence-accuracy numbers are one cookbook: 60 SEC filings, 90% right at confidence ≥0.9, 40% below. TypeSafe says it deliberately does not publish public-benchmark results.
- **"0% hallucination"**: "Our number is not empirical. Schema matching is guaranteed, thus we can confidently add 0% into the plots." The home-page FAQ says Jev "can choose the wrong one".
- **Pricing**: the launch post says "We can't prove it isn't subsidized"; the home-page FAQ says "We can serve Jev profitably at our current prices."
- **Versions**: `jev-latest` and `jev-preview` both point to `jev-1.13.0`. `jev-1.12` appears in 16 of TypeSafe's cookbooks and was "already gone" by 2026-09-17 per a third-party client. No model changelog.
- **Company**: founded 2024, San Francisco; $40M seed led by DCVC (Business Wire, 2026-09-15); Almeida CEO, Sasha Sheng COO, Erik Gafni CTO. No other investors, headcount or customers named by TypeSafe.
- **Forums**: to an HN commenter's "basically a zero-shot classifier", the CEO replied "exactly right!" (comment 49718727). The CEO conceded it can be "confidently wrong"; a TypeSafe engineer in the same thread said it "will never produce unreliable outputs".

### Jev against DiffusionGemma, hand-built items

Mastracci's thread (2026-09-17), read from his chart images. He wrote the PR he is comparing; items are hand-built; no eval code, items or per-item outputs are published (PR #57250, `djev`, `djev-spark`), so the results cannot be reproduced. Totals computed in a script from his per-set counts:

- 201 items over 8 sets: DiffusionGemma 198 (98.5%), Jev 191 (95.0%).
- Both are perfect on 5 of the 8 sets (80 of 201 items). 7 of Jev's 10 errors are in one set, PII per word: 89 words drawn from 5 sentences.
- Follow-up table adds a plain left-to-right Qwen (size not stated). On the 181 items it covers: DiffusionGemma 177, Jev 172, Qwen 169. His own images disagree on DiffusionGemma's PII count (87/89 in the follow-up table, 88/89 in the first chart, which would make 178).
- Coupled decisions: DiffusionGemma's joint read gave an inconsistent pair once in 3 (restaurant "luigis" 0.89, cuisine "thai" 0.62). Its sequential mode and Jev were consistent 3/3. That matches the PR's design: during a joint read each answer slot sees random tokens where the other answers go.
- Raw maze moves: DiffusionGemma hit walls on 80 of 84 moves (9×9) and 186 of 192 (15×15); Jev on 13 and 26.
- Latency: DiffusionGemma single reads measured warm on a local DGX Spark; Jev over the internet, including a 46 ms round trip. Single reads were faster on 7 of 8 sets. With automatic re-reads on, DiffusionGemma was slower than Jev on all 8.
- No calibration metric published.

### Nimble's own set (synthetic labels)

324 held-out examples, labels synthetic, 162 contrastive pairs, 6 source families. Quoted: Qwen3.5-9B untrained 66.4%, Qwen3.8-27B untrained 84.9%, Nimble-9B 90.1%, Jev 93.2%. Nimble's raw probabilities were overconfident: a fitted temperature of 2.179 took calibration error on a second set from 0.128 to 0.066, but did not improve it on the 324 (0.052 → 0.054).

### Laya's own card

- Base checkpoint zero-shot on their typed-decisions set: 0.362, below the 0.461 majority-class baseline. The 0.766 that "beats Jev" comes from a checkpoint fine-tuned on that benchmark's own training split.
- Its calibration figures disagree with each other. The headline table gives Jev 0.246 against Laya 0.081 (after temperature refit); no source is given for 0.246. The detailed tables (card and BENCHMARKS.md) give Jev 0.144. Laya's own raw ECE is 0.213 on the card and 0.466 as shipped in BENCHMARKS.md.
- English checkpoint on Khmer: 0.000 accuracy at 0.952 confidence (quoted in the AI Engineering article).

### Speed

The one like-for-like measurement: Qwen2.5-0.5B on SGLang, 100 cases, label scoring at 463 ms per case against 848 ms for generating up to 32 tokens, 1.83x (Fareed Khan). Client-side wall-clock in OmarMujahid's run: 975 against 1,018.5 ms (1.045x). Vendor figures: see TypeSafe's own claims.

### LangChain's judge comparison

Jev matched a human reference on 100% of 500 judgments, against GPT-5.6 Terra 99.8%, Luna 96.4%, Claude Sonnet 4.6 80.0%, with the lowest run-to-run variance. The 500 judgments are 5 test cases × 100 repeats. LangChain ships the `langchain-typesafe` integration. The result shows repeatability on 5 items.

### arXiv

14 preprints dated 2026-09-19 to 2026-09-22, read in full. 10 call TypeSafe's hosted Jev (version 1.13 where stated); 4 do not (2609.25845 is the authors' own Qwen-VL method, 2609.23959 never ran Jev, 2609.23886 is FLock.io's competing model quoting Jev replies recorded by others, 2609.25498 never runs Jev). None describes RLCD or how Jev was trained; none carries a conflict-of-interest statement. Grades: A 2609.24574; A− 2609.24052; B 2609.26550, 2609.24395; C 2609.26758 (as evidence about Jev), 2609.26532, 2609.24965, 2609.22753, 2609.23136, 2609.23986, 2609.23886.

- **2609.24574** (social-science annotation, grade A): pre-registered, 7,977 human-labelled items, per-item release that reproduces the paper's figures. Jev trails the best LLM on 14 of 15 tasks by a median 11.6 F1, at 44x lower cost than the best LLM per task (18.6x median against all 19 LLMs). Jev's ECE 0.157 beats most LLMs' stated confidence; after fitting a temperature on the pilot tasks (Jev T = 2.66), 15 LLMs beat it.
- **2609.24052** (crash narratives, A−): F1 Jev 0.908, Fable 0.967, Sol 0.885. Jev is under-confident and over-states prevalence; recalibration cuts its ECE 3.35x. The human sample was drawn using Jev's own probabilities (effective n ≈ 1,265); per-item outputs withheld.
- **2609.26550** (Jev as a judge, B; partly funded by OpenAI API credits, whose model is its main comparator): the "blinded human adjudication" is one member of the research team on 183 disputed items, with a Claude pass standing in for the second human. Under those labels Jev is −3.0 on RewardBench and −14.6 on JudgeBench. "Retains 99%" holds for GPT-6 (99.4%) and not for GPT-5.6 Sol (97.5%).
- **2609.26758** (option naming, C as evidence about Jev): with question, state, rubric and option set held fixed, only the pairing of names to rubric changes. **The headline "70.4 of every 100 answers" and "AUC 0.94 → 0.23" are an open ModernBERT model scored on [MASK] tokens (Laya's architecture), not Jev.** Jev's own result (n = 1,200): the 0/1 → no/yes swap changes 32.5% of answers against about 2% with neutral names (difference 30.4 points, 95% CI 27.6–33.3; 24x Jev's repeat noise of 1.33%); AUROC falls from 0.81 to 0.58. No LLM was given the same swap. The dataset is unnamed (likely the synthetic `LocalLLaMA/typed-decisions`, whose gold answers come from a ~4B teacher), the Jev version and access path are not stated, the promised code has no URL, and its exclusion counts contradict its tables.
- Others: two papers by the same UTS authors contradict each other on Jev's fees (69–70% cheaper than DeepSeek in one, 1.7x dearer in the other); REFLEX picked its headline threshold on the test tasks; Jev-Mem's text contradicts its own table in four cells.

The arXiv API searches metadata only and may lag a day; a paper naming Jev only in its body, or announced late on 2026-09-23, may be missing.

## Who's who

| Project | What it is | Evidence |
|---|---|---|
| **Jev** (TypeSafe) | Closed, hosted, RLCD-trained, synthetic training data, size undisclosed | Mid-tier accuracy on human labels; calibrated on familiar public data, miscalibrated elsewhere; fast and cheap |
| **Nimble-9B** (Bespoke Labs) | Qwen3.5-9B + LoRA, cross-entropy over the allowed label logits, synthetic contrastive data, not distilled from Jev; Apache-2.0 on the HF card | Public data, recipe and 13-set comparison (run by the company that sells it) |
| **Kev** (Jared Palmer) | Qwen3.5 0.8B/4B/9B-Base + LoRA rank 16 + pointer head, cross-entropy, no Jev outputs in training; Apache-2.0 | Kev-9B 0.822 dev / 0.852 test on unseen sources against Jev 0.857 dev (as reported); no independent measurement |
| **Luce** (scienthoon) | Qwen3-4B-Base + LoRA + decision head, temperature fitted on real labels; Apache-2.0 | Author's own; see above |
| **NanoJev** (TianyuCodings) | Qwen3-0.6B + decision heads; MIT | Games only; Jev beats it on Maze (7/10 against 4/10) and scores the same as untuned Qwen on ViZDoom Basic and Predict Position |
| **Laya** (Convai Innovations; code at `NandhaKishorM/laya`) | ModernBERT-large 421M or mmBERT-base 322M, option `[MASK]` scoring, REINFORCE with a proper-scoring-rule reward; Jev-compatible server | Own card only; zero-shot near chance; HF shows 3,002 likes and 0 downloads while PyPI shows 46,765 downloads last month |
| **SimpleJev** (Featherless AI) | Any HF model, next-token label logits, shared-prefix KV reuse; free demo API serving Gemma 4 26B | README: its probabilities "are not calibrated probabilities of correctness"; no accuracy published |
| **DiffusionGemma-as-Jev** (vLLM PR #57250) | One-step read of DiffusionGemma's canvas (25.2B total, 3.8B active) | Mastracci's 201 items, not reproducible; no calibration measurement |

Every trained open replica is built on Qwen or an encoder. No labelled accuracy or calibration results exist for plain Gemma read by label logits.

## Hype signals seen

- "14 projects in 48 hours"; clones and explainers within days of launch.
- `thejevai.com`, registered 2026-09-20, promoted by a Hugging Face account named `sora-2` as the endpoint for your API key. TypeSafe's API is `api.typesafe.ai`. **Do not send keys there.**
- Laya: HF likes against 0 recorded HF downloads on a five-day-old repo; headline table compares tuned Laya with untuned Jev and cites an unsourced Jev ECE.
- Google's Gemma account: bidirectional attention "yielding well-calibrated decision distributions", with no data.
- Vendor multiples (193.6x, 444.6x) with no stated comparator, against measured parity to 478x depending on comparator; "never hallucinates" for a model that can still pick the wrong valid option.
- Partner-published "independent" evaluations (LangChain, Every, Valyu-tagged guides) and competitor-run ones (Bespoke Labs, synthorai).
- About 60 dev.to posts on Jev in eight days, nearly all with zero reactions; 17 of the 33 read in full restate vendor claims.
- Syndicated "analysis" (Cherry Creek News / North Denver Tribune) overstating a three-case demo.
- A NanoJev write-up (wonderlab) whose table does not match NanoJev's own README.
- Several self-published evaluations chose thresholds on their test data or state more than their own data supports; per-study details stay in the private working notes.

## What holds up

1. Asking bounded questions and reading label scores is cheaper and more reliable than parsing generated text, on any model.
2. Splitting a task into narrow typed questions improves results whatever the model: TypeSafe reports Haiku 4.5 from 18.1% as one prompt to 53.6% in a structured workflow, Opus 5 from 64.8% to 73.1%; independently, a 12-way choice at 0.40 became 0.91 as scored dimensions.
3. Thresholds must come from your own labelled data. TypeSafe, Vercel, Laya, Nimble and SimpleJev all say so.
4. Wording decides: bare option labels sent 0 of 40 hard tasks to the strong model; one-line descriptions fixed 37 of 40.
5. Option names matter: swapping 0/1 for no/yes with everything else fixed changed 32.5% of Jev's answers (arXiv 2609.26758, n = 1,200).
6. TypeSafe's own list of Jev's failure modes (`docs.typesafe.ai/model-jaggedness/jev-1.13`): literal reading, no arithmetic or counting, dates, long irrelevant state, adversarial content in the state, and answers that change with phrasing. Its examples: a refund question scored 0.22 as a yes/no and 0.01 (confidence 0.97) as a yes/no choice; "refund" 0.72 and "not a refund" 0.47, summing to 1.19.

## The open question

Luce and others have answered the Qwen version: a few hundred to a thousand labels beat Jev where the label follows from the input, and a small refit fixes calibration. What remains open:

1. **Gemma.** Every trained open replica is built on Qwen or an encoder. The Gemma data points in the literature are all generative read-outs: Gemma 4 31B beats Jev's median F1 in arXiv 2609.24574 (0.611 against 0.581, grade A); heiko-hotz has Gemma 4 31B at 98.86% on SNIPS after stripping code fences against Jev 97.14%; ywchiu (synthetic routing) has Gemma 4 31B QAT ahead on joint fields and the DiffusionGemma-based `djev-spark` at 32.2% against Jev's 61.4%; ikkun1222's Gemma 4 26B tied Jev on two Japanese sentiment tasks; kunko's Gemma 4 E4B wrote its confidence as text. Plain Gemma read by label logits, with calibration measured, is unpublished.
2. **Data from after the training cutoffs.** Public datasets may be in Jev's training mix, and OmarMujahid's biggest leads are on the most exposed sets. Private data from these rigs (tool calls, startup logs, zone-failure messages) cannot be.
3. **Calibration across difficulty on real data.** The one gradient run used synthetic items.
4. **DiffusionGemma's calibration**, claimed by Google, is unmeasured.

## State of this directory

- `data/`: 1,200 labelled examples (sst2, ag_news, emotion, tweet_eval irony; 300 each, stratified, fixed seed), from `build_eval_set.py`.
- `run_eval.py`: two arms on identical prompts, templates and label tokens, reusing the PR's code: DiffusionGemma one-step reads (4 noise draws) and Gemma 4 26B next-token label logits. Checks that both chat templates render the same prefix before sending anything.
- `score.py`: accuracy, calibration error, Brier, log loss, confidence-vs-correctness ranking, escalation rule at the PR's 0.1 threshold, noise-draw spread, paired differences with resampled 95% ranges. Unit-tested (`tests/test_score.py`); pipeline dry-run against `tests/fake_vllm.py`.
- `research_notes/` (gitignored, private): primary-source audits behind this note and the report.
- **Not run on a real model.** Needs a GPU with ≥80 GB for both 26B models in bf16, or the SimpleJev demo API for plain Gemma 26B.

Proposed next steps, cheapest first:

1. Label a few hundred real decisions from these rigs (tool-call risk, log-line class, zone-failure cause) across difficulty tiers, labels committed before any model call, with a regex baseline per tier.
2. Plain Gemma 4 (E2B, E4B, 26B) reading label logits on that set, plus the four public tasks already in `data/` and, for comparison with published Jev numbers, Nimble's 13-dataset suite.
3. Calibration against labels used for temperature fitting (N = 0, 25, 50, 100, 300): arithmetic on stored logits, no further model calls.
4. Consistency checks: identical input repeated, options reordered, option names changed (0/1 against no/yes), yes/no against choice, a question against its negation.
5. DiffusionGemma as an arm on one 80 GB GPU; Laya as a CPU arm; Jev through `api.typesafe.ai` if waitlist access arrives.

Not supported on TPU as of this date: tpu-inference has DiffusionGemma config scaffolding only. The base checkpoint shares one set of 30-layer weights between encoder and decoder, so a one-step read is a small addition to the `~/tpu-jax-26b` engine; that engine has never run the 26B on a TPU.

## Sources

- TypeSafe: [announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [System One](https://docs.typesafe.ai/concepts/system-one), [Confidence](https://docs.typesafe.ai/confidence), [AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer), [Models](https://docs.typesafe.ai/models), [Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13), [evals](https://evals.typesafe.ai)
- [LangChain: Building a harness with Jev](https://www.langchain.com/blog/building-a-harness-with-jev) · [Vercel: What is Jev](https://vercel.com/i/what-is-jev)
- Karl Weinmeister, [How to build a Jev-style classifier with DiffusionGemma and vLLM](https://medium.com/google-cloud/how-to-build-a-jev-style-classifier-with-diffusiongemma-and-vllm-ef2e0bfa9ad7) · [vLLM PR #57250](https://github.com/vllm-project/vllm/pull/57250) · [DiffusionGemma model card](https://huggingface.co/google/diffusiongemma-26B-A4B-it)
- [Google Gemma post](https://x.com/googlegemma/status/2101069861598482817) · [Matt Mastracci's evals](https://x.com/mmastrac/status/2100626193943052784)
- [bespokelabsai/nimble](https://github.com/bespokelabsai/nimble) · [jaredpalmer/kev](https://github.com/jaredpalmer/kev) · [featherless-ai/simple-jev](https://github.com/featherless-ai/simple-jev) · [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) · [convaiinnovations/laya (HF)](https://huggingface.co/convaiinnovations/laya) · [TianyuCodings/NanoJev](https://github.com/TianyuCodings/NanoJev)
- Independent evaluations: Manjunath Janardhan, [I Tested TypeSafe's Jev … Only Two Beat It](https://medium.com/data-science-collective/i-tested-typesafes-jev-a-470-cheaper-decision-model-against-claude-gpt-6-kimi-minimax-and-d36ed152e861) and [manjunathshiva/jev-frontier-bench](https://github.com/manjunathshiva/jev-frontier-bench) · [anisselbd/jev-phishing-bench](https://github.com/anisselbd/jev-phishing-bench) · [scienthoon/luce](https://github.com/scienthoon/luce) · [scienthoon/jev-ood-calibration](https://github.com/scienthoon/jev-ood-calibration) · [OmarMujahid/jev-decision-bench](https://github.com/OmarMujahid/jev-decision-bench) · [bitnovus/jev-spam-eval](https://github.com/bitnovus/jev-spam-eval) · [themsquared/jev-benchmark](https://github.com/themsquared/jev-benchmark) · [SamuelSacco/jev-exploration](https://github.com/SamuelSacco/jev-exploration) · [sanand0 BANKING77 pilot](https://sanand0.github.io/llmevals/jev/) · [JevBench](https://jevbench.xyz/) · [Aman Kumar](https://amankumar.ai/blogs/jev-measured) · [Paweł Józefiak](https://thoughts.jock.pl/p/jev-typesafe-system-one-model-benchmark-2026)
- dev.to measurements: [Jev vs a 310M encoder](https://dev.to/ikkun1222/jev-vs-a-310m-encoder-i-trained-myself-750-rows-three-tasks-two-different-winners-242e) · [one judgment or twelve dimensions](https://dev.to/ikkun1222/jev-one-judgment-call-or-twelve-dimension-scores-i-measured-both-on-three-classification-tasks-31fd) · [agent harness](https://dev.to/aitejiu/benchmarking-jev-what-a-decision-model-can-and-cant-do-in-an-agent-harness-20po) · [Jev vs flash LLMs](https://dev.to/synthorai/jev-vs-flash-llms-7x-cheaper-on-workflows-same-cost-on-one-label-1e2j) · [the 200x claim](https://dev.to/devopsdaily/we-measured-the-200x-claim-and-got-it-wrong-twice-first-5ch5) · [TLA+ pharmacy](https://dev.to/copyleftdev/i-put-jev-behind-a-tla-spec-and-ran-1680-chaos-tested-pharmacy-decisions-zero-wrong-verdicts-1ij8) · [judge-audit](https://dev.to/kunko_ai_labs/your-ai-judge-says-98-confident-does-it-mean-it-10lj) · [event validation](https://dev.to/jonreed/testing-typesafe-jev-mistral-and-gemini-for-local-event-validation-5697)
- Press and forums: TechCrunch (Tim Fernholz, 2026-09-18) · Forbes (Josipa Majic Predin, 2026-09-19) · Tom's Hardware · [Every: Mini-Vibe Check](https://every.to/vibe-check/mini-vibe-check-typesafe-s-jev-judged-everything-i-ve-written-in-0-7-seconds) · [Cherry Creek News](https://thecherrycreeknews.com/jev-typesafe-benchmark-checked-explainer-wave-cherry_creek/) · Latent Space interview with Diogo Almeida (2026-09-21) · Hacker News launch thread (1,976 points, 510 comments) · LessWrong trusted-monitor post (Ventak T)
- Prior art: Guo et al. 2017 (arXiv:1706.04599) · Schick & Schütze (arXiv:2001.07676, 2009.07118) · Zhao et al. 2021 · Brown et al. 2020 · Hendrycks et al. 2021 · Devlin et al. 2019 · Yin, Hay & Roth 2019 · Gneiting & Raftery 2007 (doi:10.1198/016214506000001437) · OpenAI GPT-4 technical report (arXiv:2303.08774) · Kadavath et al. 2022 · Christiano et al. 2017 (arXiv:1706.03741) · Ouyang et al. 2022 (arXiv:2203.02155) · Geifman & El-Yaniv 2017 · Chen, Zaharia & Zou (FrugalGPT)
- Roundups: Tushar Kanjariya, [Jev Is Everywhere This Week](https://medium.com/@TusharKanjariya/jev-is-everywhere-this-week-is-it-faster-or-just-loud-efb67b6308ac) · Joe Njenga, [JEV AI Is Here](https://medium.com/ai-software-engineer/jev-ai-is-here-i-tested-it-and-debunked-the-new-model-hype-fec4028e04e9) · Fareed Khan, [Jev Clearly Explained](https://levelup.gitconnected.com/jev-clearly-explained-3203d6d48de3) · unicodeveloper, [The Ultimate Guide to Jev](https://medium.com/@unicodeveloper/the-ultimate-guide-to-jev-the-new-frontier-ai-for-faster-decisions-acd78e5f4c56) · AI Engineering: [two open-source replicas](https://medium.com/@ai-engineering-trend/just-two-days-after-jev-went-viral-two-open-source-replicas-are-here-one-9b-one-0-5b-57024ceef03e), [Laya](https://medium.com/@ai-engineering-trend/laya-the-open-source-jev-alternative-thats-7x-faster-and-runs-under-1gb-of-ram-49c05ff410b5) · Christian Graham, [Laya plays Doom](https://medium.com/@christian.graham_49279/laya-a-free-local-alternative-to-jev-and-it-can-even-play-doom-ish-42e2292e541e)
- Not TypeSafe's: [HF post pointing at thejevai.com](https://huggingface.co/blog/sora-2/how-to-use-the-jev-ai-model-a-step-by-step-develop)
