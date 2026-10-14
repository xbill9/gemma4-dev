# Notes — BEIR retrieval on the T4 (2026-10-08)

Design is fixed in `scripts/beir_eval.py`'s docstring, written before the first model call.
Datasets: BEIR zips from public.ukp.informatik.tu-darmstadt.de, sha256 in `evidence/dataset-sha256.txt`.

## Deviation: nDCG gain

`beir_eval.py` computes nDCG@10 with exponential gain (2^g − 1); its docstring wrongly calls that
"BEIR's definition". BEIR (pytrec_eval) uses linear gain. SciFact's qrels are binary, so its numbers are
identical either way. NFCorpus's are graded 1–2: `scripts/ndcg_linear.py` recomputes it from the saved
rankings (`evidence/ndcg-linear.json`): embed2 768-d **0.369** (0.370 exponential), BM25 **0.307**
(0.308). The Matryoshka 512/256/128 rows for NFCorpus in `evidence/summary.json` remain exponential-gain.

## Sanity check

Stdlib BM25 (no stemming, no stopwords, k1=1.2, b=0.75) scores SciFact 0.662 against 0.665 published
in the BEIR paper; NFCorpus 0.307 against 0.325 (tokenisation differs from Elasticsearch's).

## Not established

Whether either set appears in EmbeddingGemma 2's training data. SciFact's 0.851 should be read with that open.
