---
title: "Gemma 4 E2B and Long Documents on an 8 GB RTX 3050: Keep It Short or Turn Thinking On"
published: false
description: "Gemma 4 E2B loads a 128k-token document on an 8 GB RTX 3050 in 3.8 GB, but as it serves by default it found 52 of 80 hidden facts at 16k tokens and 14 at 32k. Thinking on with Google's sampling settings found 38 at 32k and 34 at 64k. Keep E2B's document under 16k tokens, or turn thinking on."
tags: gemma, llm, benchmark, ai
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/docs/gemma4-e2b-long-context-3050/devto-cover.ef55dd33.jpg
---

Gemma 4 E2B loads a 128k-token document on an 8 GB RTX 3050 in 3.8 GB and reads it in under two minutes. What it finds there depends on the length. At temperature 0 with thinking off, the way the rig serves it, E2B found 52 of 80 hidden facts in a 16k-token document, 14 at 32k and 7 at 64k.

Thinking on with Google's sampling settings finds more: 38 of 80 at 32k and 34 at 64k, and on a harder task that joins two facts, 42 and 32 against 14 and 1. Over 16k to 98k tokens it found 264 facts of 640 against 124, ahead in 55 trials and behind in 9 (p = 3.54e-9). Google's sampling with thinking off found no more than temperature 0, and the drafter changed nothing beyond chance.

On a small card, give E2B only the part of the document that matters, under 16k tokens, or turn thinking on and pay 5 to 11 seconds more per request. On an 8 GB card the 12B also fits with a 131k-token context, and an earlier run of the same recall test found 48 to 59 of 80 facts there from 32k to 127k.

This article provides a step by step test of Gemma 4 E2B with a long document on one local GPU: a 4-bit QAT build on an 8 GB RTX 3050 through llama.cpp. Facts were hidden in a technical manual cut to five lengths from 16k to 127k tokens, and E2B was asked to find them in six server settings, 580 requests in all.

https://github.com/xbill9/gemma4-dev/tree/main/docs/gemma4-e2b-long-context-3050

---

#### The Findings

**Past 16k tokens, E2B as served finds few facts.** At temperature 0 with thinking off it found 52, 14, 7, 4 and 11 of 80 codes at 16k, 32k, 64k, 98k and 127k tokens. The 12B, in an earlier run of the same test, found 67, 58, 59, 48 and 56.

**It finds what is near the start.** In replies that ended on their own, E2B found 50 of 73 codes placed in the first third of the document, 22 of 63 in the middle third and 5 of 64 in the last third.

**A miss is a fact left out.** In no setting did E2B give a wrong code: every recall miss was a code missing from its list.

**Some replies list the manual instead.** 28 of 100 replies ran to the 256-token reply limit, none at 16k. 27 of them named none of the hidden projects: they listed names from the manual itself, "NVIDIA Corporation", "Fedora", "OpenGL", each with "N/A", and 15 repeated one line five times or more.

**Thinking on with Google's sampling settings found about twice as many.** 138 against 77 codes and 126 against 47 links over 16k to 98k tokens, with no reply at its limit. Sampling with thinking off was level with temperature 0, so the gain comes with thinking.

**Keep the f16 cache.** q8_0 was level with f16: 131 facts against 138, 8 trials ahead and 9 behind. q4_0 found fewer, though not beyond chance once every comparison here is counted. Both generate slower with the drafter at 127k: 51.6 and 48.9 tokens per second against 83.6.

**Reading the document is the wait.** 5.6 seconds for 16k tokens and 115 seconds for 127k, before the first word of the reply.

**What to do with it.**

- 🟢 Under 16k tokens, E2B as served finds most facts and answers in about 6 seconds.
- 🟢 Longer than that, turn thinking on with Google's sampling settings.
- ❌ Beyond 64k tokens even that finds fewer than one fact in five on recall. On an 8 GB card, load the 12B for long documents.

---

#### What Was Measured

| | |
| :--- | :--- |
| Model | Gemma 4 E2B, exact Q4_0 repack of Google's QAT weights |
| Card | RTX 3050 OEM, 8 GB, driver 615.78.08 |
| Server | llama.cpp `7182ef71e`, the rig's serving flags, one request at a time |
| Context | 131,072 tokens |
| Document lengths | 16,384, 32,768, 64,000, 98,304 and 126,976 tokens |
| Tasks | recall (8 codes) and linking (8 cities, two facts each) |
| Trials | 10 per length and task, the same 10 in every setting |
| Settings | 6, listed in Step 2 |

---

#### At This Point You Should Have…

- An NVIDIA card with 8 GB, here an RTX 3050 OEM
- llama.cpp built with CUDA, here commit `7182ef71e`
- Python 3
- A clone of [gemma4-dev](https://github.com/xbill9/gemma4-dev)

---

#### Step 1 — The Test

The document is the NVIDIA Linux driver README, a long technical manual, tokenized and cut to each length. Into it go short sentences at line breaks spread evenly through the text, with random names and numbers drawn from a fixed seed, so every setting sees the same documents.

**Recall** hides eight sentences such as:

> The access code for project Cobalt is 858622.
>
> The access code for project Lumen is 532344.

and asks: *"List the access code of every project mentioned in the document above, as a JSON object mapping project name to code. Output only the JSON."*

**Linking** hides sixteen sentences, two per project, in shuffled places, so the two halves of each answer sit apart:

> Project Ember is led by Mei Chen.
>
> Mei Chen works from the Cusco office.

and asks for each project's office city. To answer, the model has to find the project, find its lead, then find the lead's office.

A reply is scored by code: the JSON is read, and each of the 8 codes or cities counts as right only if it matches exactly. A trial scores 0 to 8, and each length gives 10 trials of 8 facts, 80 in all.

The test is the one an earlier 12B run on this card used, with the same document and the same recall sentences, so its numbers can sit beside these.

The prompt cache is off for every request, so each one reads its whole document, and the time to read it is measured every time.

---

#### Step 2 — Six Settings

| Setting | Cache | Drafter | Sampling | Thinking |
| :--- | :--- | :--- | :--- | :--- |
| **As served** | f16 | 2 tokens | temperature 0 | off |
| No drafter | f16 | none | temperature 0 | off |
| q8_0 cache | q8_0 | 2 tokens | temperature 0 | off |
| q4_0 cache | q4_0 | 2 tokens | temperature 0 | off |
| Google's sampling | f16 | none | temperature 1.0, top-k 64, top-p 0.95 | off |
| Google's sampling, thinking on | f16 | none | temperature 1.0, top-k 64, top-p 0.95 | on |

"As served" is how the E2B rig on this card runs. Google's sampling settings come from the model's own `generation_config.json`, with the trial number as the seed.

Replies are capped at 256 tokens, enough for eight JSON entries, and at 16,384 with thinking on. With thinking on, the 127k length leaves too little room in the context for the thinking, so that setting stops at 98k.

---

#### Step 3 — Recall by Length

Codes found, of 80:

| Length | 16k | 32k | 64k | 98k | 127k |
| :--- | ---: | ---: | ---: | ---: | ---: |
| As served | 52 | 14 | 7 | 4 | 11 |
| No drafter | 49 | 20 | 11 | 5 | 8 |
| Google's sampling | 51 | 32 | 10 | 3 | 10 |
| Google's sampling, thinking on | 🥇 53 | 🥇 38 | 🥇 34 | 🥇 13 | not run |
| 12B, earlier run, q4_0 cache | 67 | 58 | 59 | 48 | 56 |

At 16k all four E2B settings find about two codes in three. From 64k on, the settings without thinking find at most 11 of 80; thinking on still finds 34 at 64k.

The 12B row comes from an earlier run on this card: the same document and sentence design with its own random draws, temperature 0, the prompt cache on and recall only. It is the same test on different trials, so it shows the size of the gap; the tests in Step 6 pair E2B settings only.

---

#### Step 4 — Linking Two Facts

Cities found, of 80:

| Length | 16k | 32k | 64k | 98k | 127k |
| :--- | ---: | ---: | ---: | ---: | ---: |
| As served | 21 | 14 | 1 | 11 | 3 |
| No drafter | 19 | 13 | 1 | 12 | 2 |
| Google's sampling | 12 | 12 | 3 | 9 | 7 |
| Google's sampling, thinking on | 🥇 32 | 🥇 42 | 🥇 32 | 🥇 20 | not run |

Linking is harder than recall at every length: as served, E2B joined 21 of 80 at 16k, where it recalled 52 codes. Thinking on finds 42 of 80 at 32k, three times as many as the settings without it.

Thinking also changes the kind of mistake. As served, of 400 cities asked for, E2B gave 50 right, 35 that belonged to another project, 41 that did not belong to any of the 8, and left 274 out. With thinking on, of 320 it gave 126 right, 30 from another project, 2 from none of them, and left 162 out.

One real trial at 32k, as served and with thinking on, both 5 of 8:

```
As served:
  "NVIDIA Accelerated Linux Graphics Driver README and Installation Guide": "Santa Clara, CA",
  "Project Cobalt": "Osaka", "Project Quartz": "Hanoi", "Project Garnet": "Tunis",
  "Project Ember": "Cusco", "Project Kestrel": "Gdansk"

Thinking on:
  "Project Cobalt": "Osaka", "Project Tundra": "Lisbon", "Project Kestrel": "Gdansk",
  "Project Quartz": "Tunis", "Project Garnet": "Tunis", "Project Ember": "Cusco"
```

---

#### Step 5 — What the Misses Look Like

**Left out, never wrong.** Across all six settings E2B never gave a wrong recall code. When it misses a fact, the fact is absent from the list. A reply at 32k, as served, 3 of 8:

```
{ "Cobalt": "858622", "Lumen": "532344", "Alder": "455157" }
```

**The start of the document wins.** Codes found by where they sat, 16k to 98k tokens:

| | First third | Middle third | Last third |
| :--- | ---: | ---: | ---: |
| As served, replies that ended on their own | 50 of 73 | 22 of 63 | 5 of 64 |
| Google's sampling, thinking on | 76 of 111 | 39 of 108 | 23 of 101 |

Thinking on lifts every third, the last one most, from 5 of 64 to 23 of 101.

**Listing the manual.** As served, 28 of 100 replies ran to the 256-token limit, 11 at 32k and 11 at 64k. They found no fact, and 27 named none of the hidden projects. They list the manual's own names, then repeat one:

```
"NVIDIA Accelerated Linux Graphics Driver README and Installation Guide": "N/A",
"NVIDIA Corporation": "N/A", "Linux": "N/A", "Fedora": "N/A", "Red Hat": "N/A",
"SuSE": "N/A", "Mandriva": "N/A", "Intel": "N/A", "OpenGL": "N/A", ...
"NVIDIA Corporation": "N/A", "NVIDIA Corporation": "N/A", "NVIDIA Corporation": "N/A", ...
```

These replies carry none of the 8 project names, so a longer limit would let them list more of the manual. The position pattern above counts only replies that ended on their own, so the limit does not explain it. With thinking on, no reply reached its limit, and the longest was 2,022 tokens.

---

#### Step 6 — The Drafter and the Cache

Every comparison here is paired: the same trials in two settings. The 8 facts in one trial share a document and a reply, so they are not independent; the p-values quoted count trials, an exact sign test on the trials where one setting found more facts than the other. Against the setting as served, recall and linking together:

| Setting | Lengths | Trials ahead / behind | p | After correcting for 13 tests |
| :--- | :--- | :--- | ---: | :--- |
| No drafter | all five | +9 / −9 | 1 | level |
| q8_0 cache | all five | +8 / −9 | 1 | level |
| q4_0 cache | all five | +7 / −22 | 0.00813 | not beyond chance |
| Google's sampling | 16k to 98k | +13 / −13 | 1 | level |
| Google's sampling, thinking on | 16k to 98k | +55 / −9 | 3.54e-9 | holds |

Two more comparisons separate the parts. Thinking on against off, both with Google's sampling: +50 / −13 (p = 3.02e-6). Thinking on with Google's sampling against temperature 0, neither with the drafter: +56 / −10 (p = 6.90e-9). Both hold after correcting for 13 tests (Holm).

The drafter and the cache change speed. Generation in tokens per second, at 16k and 127k tokens held:

| | f16 | q8_0 | q4_0 | f16, no drafter |
| :--- | ---: | ---: | ---: | ---: |
| 16k | 🥇 161.7 | 130.6 | 130.9 | 96.2 |
| 127k | 🥇 83.6 | 51.6 | 48.9 | 55.2 |
| Card memory at start | 3,802 MiB | 3,880 MiB | 3,684 MiB | 3,610 MiB |

E2B's cache at 131,072 tokens is small enough in f16 that q8_0 and q4_0 save almost nothing: q4_0 starts at 3,684 MiB against 3,802, and q8_0 at 3,880.

---

#### Step 7 — Time

Medians over the 20 requests at each length, as served:

| Length | Reading the document | Reading speed | Generation | Whole request |
| :--- | ---: | ---: | ---: | ---: |
| 16k | 5.6 s | 2,963 tok/s | 161.7 tok/s | 6.2 s |
| 32k | 13.9 s | 2,378 tok/s | 153.7 tok/s | 15.6 s |
| 64k | 37.3 s | 1,723 tok/s | 121.2 tok/s | 39.6 s |
| 98k | 74.6 s | 1,320 tok/s | 89.1 tok/s | 76.2 s |
| 127k | 115.0 s | 1,106 tok/s | 83.6 tok/s | 117.2 s |

Reading is most of the wait at every length. With thinking on, replies grew to a median of 626 tokens, and each request took 5.0 to 10.9 seconds longer than with thinking off at the same sampling. The card drew about 91 W under load.

These are cold reads: the prompt cache was off. In a conversation that keeps the document in context, llama.cpp reuses what it has read, and later questions skip most of this wait.

---

#### 🔎 Tip: Test Your Own Document at Your Own Length

A model that loads 128k tokens has not shown it can use them. Hide a few sentences with made-up names and numbers in a document like yours, at the length you plan to send, and score the reply by code. On this card one request took 6.2 seconds at 16k and 117.2 at 127k. If the facts from the last third come back missing, shorten the document or change the setting before you trust the answers.

---

#### Compare and Contrast

| | As served | Google's sampling | Google's sampling, thinking on |
| :--- | :--- | :--- | :--- |
| Recall, 16k | 52 of 80 | 51 | 🥇 53 |
| Recall, 64k | 7 of 80 | 10 | 🥇 34 |
| Linking, 32k | 14 of 80 | 12 | 🥇 42 |
| Replies at the limit | 28 of 100 | 17 of 100 | 🥇 0 of 80 |
| Request at 64k | 🥇 39.6 s | 37.9 s | 46.4 s |
| Drafter | 🥇 yes, 161.7 tok/s at 16k | no | no |

**Pros and cons**

- **As served.** 🟢 Fastest, and finds most facts under 16k tokens. ❌ From 32k on it finds fewer than one in four, and some replies list the manual instead.
- **Google's sampling, thinking off.** 🟢 Fewer replies at the limit. ❌ Level with temperature 0 on facts found.
- **Google's sampling, thinking on.** 🟢 About twice the facts from 32k on, on both tasks, and no reply at the limit. ❌ 5 to 11 seconds more per request, no drafter in this run, and still under half the facts beyond 32k on recall.

---

#### So, Which One?

**Under 16k tokens: E2B as served.** It finds about two facts in three, with the drafter, in about 6 seconds a request.

**Longer on a 4 GB card: E2B with thinking on and Google's sampling settings,** and cut the document down where you can. It finds about twice as many facts as temperature 0, and still loses most of them beyond 64k.

**Longer on an 8 GB card: the 12B.** It fits with a 131k-token context, and in an earlier run of the same recall test found 48 to 59 of 80 codes from 32k to 127k, where E2B as served found 4 to 14.

**Cache: f16.** q8_0 and q4_0 save E2B no memory worth having and generate slower.

---

#### Summary

The goal of this article was to find how much of a long document Gemma 4 E2B uses on an 8 GB card. The key to the solution was hiding facts in the same documents for every setting and scoring the replies by code, trial by trial. The results were:

- ❌ As served, E2B found 52 of 80 codes at 16k tokens, 14 at 32k and 4 to 11 from 64k on.
- ⚠️ It finds what is near the start: 50 of 73 codes in the first third, 5 of 64 in the last.
- 🟢 It never gave a wrong code; a miss is a code left out.
- 🟢 Thinking on with Google's sampling settings found 264 facts of 640 against 124 over 16k to 98k (p = 3.54e-9).
- ⚠️ Google's sampling with thinking off, the drafter and a q8_0 cache changed nothing beyond chance.
- 🟢 Reading takes 5.6 seconds at 16k and 115 seconds at 127k.

Scope: one RTX 3050 OEM with 8 GB in one desktop, llama.cpp `7182ef71e`, Gemma 4 E2B as an exact Q4_0 repack of Google's QAT weights, one request at a time, 2026-11-25 and 26. One document, the NVIDIA Linux driver README, with short made-up sentences inserted at line breaks; documents of other kinds, and facts phrased like the text around them, are untested. Ten trials per length and task, one seed per trial for the sampled settings. With thinking on, the 127k length was not run, and thinking on at temperature 0 was not run. The 12B figures come from an earlier run of the same test on this card with its own random draws, temperature 0, the prompt cache on and recall only, so they are not paired with these. The article reports 13 paired tests; corrected for that number (Holm), the gains from thinking on with Google's sampling hold, and the q4_0 cache's loss does not. "Level" means no gap beyond chance.

The strategy for testing a small model with a long document on a small GPU was validated with an incremental step by step approach.

---

#### References

- [gemma4-dev on GitHub](https://github.com/xbill9/gemma4-dev)
- [This article's evidence: scripts and outputs](https://github.com/xbill9/gemma4-dev/tree/main/docs/gemma4-e2b-long-context-3050/evidence)
- [The run: replies, logs and the test script](https://github.com/xbill9/gemma4-dev/tree/main/local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-11-25-longctx-e2b-3050oem)
- [The 12B's recall run on the same card](https://github.com/xbill9/gemma4-dev/tree/main/local-llamacpp-3050oem-12b-q4_0/benchmarks/runs/2026-10-17-kvcache-12b-3050oem)
- [E2B, E4B or 12B? On an 8 GB RTX 3050, Serve Gemma 4 E4B](https://dev.to/gde/e2b-e4b-or-12b-on-an-8-gb-rtx-3050-serve-gemma-4-e4b-1eji)
- [Exact Q4_0 repack of Gemma 4 E2B QAT](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)

Hope you enjoyed it!
