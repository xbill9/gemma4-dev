# Plan — EmbeddingGemma 2 on jev's four tasks (written before any model call, 2026-10-08)

**Question.** On the 1,200 examples jev scored (`../../../../jev-tpu/data/*.jsonl`, 300 each of
sst2, AG News, DAIR Emotion, tweet_eval irony; byte-identical to `../../../../jev/data`), how does
`google/embeddinggemma-2` on the local T4 classify, against Gemma 4's label read on one v6e chip
(`jev-tpu/results/2026-09-24-v6e1-{e2b,e4b,12b,26b-fp8}/autoregressive-<task>.jsonl`)?

**Not a speed comparison.** Different model, task, output and chip. No latency is compared.

**Not like-for-like on supervision either.** Jev's read is zero-shot; its "after 50 labels" uses
labels only to *calibrate* (fit a temperature). Readouts B and C below *train* on labels. Only A
is zero-shot like jev's.

## Readouts (fixed now)

Server: the sweep's server, unchanged (`--dtype float32`, vision on, revision `914f7f89`). Full 768 d.
Texts are embedded with the model's `Classification` prompt, `task: classification | query: `
(`config_sentence_transformers.json`).

- **A, zero-shot (primary).** Each label is a fixed sentence (below), embedded with the same
  `Classification` prompt; predict the label with highest cosine. **A2 (secondary):** text with
  `task: search result | query: `, labels with `title: none | text: `.
- **B, 50 labels, nearest centroid.** 200 draws (seeded) of 50 training examples uniformly at random;
  predict the other 250; report mean accuracy and the 2.5–97.5% range over draws. A class absent
  from a draw gets no centroid.
- **C, 299 labels, nearest centroid, leave-one-out.** Every example predicted from the centroids of
  the other 299. An upper reference, not a deployable number.

Comparison: accuracy per task; for A, A2 and C, paired bootstrap of (embed − jev arm) over 2,000
resamples of examples, 95% range. Jev's prediction is the argmax of `reads[0].probs` over `names`
in the non-reversed file.

## Label sentences (fixed now, not tuned)

| task | label | sentence |
|---|---|---|
| sst2 | negative | A negative movie review. |
| sst2 | positive | A positive movie review. |
| ag_news | world | World news about international politics and conflicts. |
| ag_news | sports | Sports news. |
| ag_news | business | Business and financial news. |
| ag_news | scitech | Science and technology news. |
| emotion | sadness | The writer feels sadness. |
| emotion | joy | The writer feels joy. |
| emotion | love | The writer feels love. |
| emotion | anger | The writer feels anger. |
| emotion | fear | The writer feels fear. |
| emotion | surprise | The writer feels surprise. |
| irony | yes | An ironic or sarcastic tweet. |
| irony | no | A sincere tweet that is not ironic. |

Any change after the first model call is recorded as a dated deviation below.

## Deviations

- **2026-10-08, start time only.** The run was queued behind the sweep with a `pgrep -f scripts/sweep.py`
  wait loop, which matched its own command line and never exited. It was killed and the script run by
  hand at 16:40 UTC against the same server, GPU idle (0%). No readout, label sentence or prompt changed.
