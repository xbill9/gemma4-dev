# 2026-10-13 — answer quality, E2B vs E4B, RTX 3050 OEM

**Answer: E4B is better overall and on math, and level on tool calling.** It scores higher on
11 of the 13 classification subsets (7 with a range clear of zero) and lower on 2. Suite +4.8 points, GSM8K +3.5 points, BFCL −1.5 points (not significant). Both Q4_0
builds score within the ranges of their TPU references, so the gap is the model, not the
quantization or the serving stack.

## Setup

Both exact Q4_0 GGUFs (E2B `419db9a6…`, E4B `5462cc10…`) on llama.cpp `77c4540ce`, one at a time,
with identical server flags: `-ngl 99 -c 32768 --parallel 8 -ctk f16 -ctv f16 -fa 1 -t 6 -tb 12
--reasoning off -ot 'per_layer_token_embd\.weight=CUDA0'`, **no drafter**, so the scores are the
models' own. `run_quality.sh` ran, per model:

- **Nimble 13-subset suite**, 3,880 records rebuilt with `jev-tpu-v5e1/nimble_suite/build.sh`
  (every subset's `dataset_sha256` matched), read by `nimble_suite/run_suite.py` through
  `vllm_shim.py` (llama-server `/completion` with `n_probs` 32, `JEV_TOPK=32`), 8 at a time.
- **GSM8K**, all 1,319 test problems, `gen_eval.py`, 2,048-token limit, greedy, 8 at a time.
- **BFCL simple**, 400 tool-calling records, `gen_eval.py`, greedy, 8 at a time.

Scoring is the harnesses' own code (`score_suite.assess`, `gen_eval.py`'s `right`); pairing and
bootstrap ranges are `quant_compare.paired` and `gen_compare.compare`, run by `compare.py`.
Full tables in `COMPARE.md`, numbers in `compare.json`, raw records in the suite directories and
`gsm8k-*.jsonl` / `bfcl-*.jsonl`. No errors in any task.

## Results

| task | E2B | E4B | E4B − E2B | 95% range |
| --- | ---: | ---: | ---: | --- |
| Suite, 13 classification subsets (3,880) | 0.683 | 0.730 | **+0.048** | +0.035 to +0.061 |
| GSM8K, math word problems (1,319) | 0.898 | 0.933 | **+0.035** | +0.020 to +0.050 |
| BFCL simple, tool calling (400) | 0.925 | 0.910 | −0.015 | −0.043 to +0.013 |

On the suite E4B gets 463 records right that E2B gets wrong, against 278 the other way; on GSM8K
74 against 28; on BFCL 13 against 19.

**Per suite subset** (secondary, ranges not corrected for 13 comparisons): the largest gains are
reading comprehension and inference — squad2 +0.171, boolq +0.137, multinli +0.094 — and the
two MASSIVE intent sets (+0.074 German, +0.057 English). E4B is lower on vitaminc-dev (−0.042,
range −0.080 to −0.005) and pubmedqa (−0.044, range includes zero).

## Against the TPU references

| | difference, 3050 Q4 − TPU reference | 95% range |
| --- | ---: | --- |
| E2B suite vs E2B bf16 (v5e) | +0.000 | −0.010 to +0.010 |
| E4B suite vs E4B bf16 (v6e) | −0.000 | −0.007 to +0.006 |
| E2B GSM8K vs E2B bf16 (v5e) | −0.012 | −0.025 to +0.001 |
| E4B GSM8K vs E4B QAT repack W4A16 (v5e) | −0.002 | −0.009 to +0.005 |
| E2B BFCL vs E2B bf16 (v5e) | −0.003 | −0.020 to +0.015 |
| E4B BFCL vs E4B QAT repack W4A16 (v5e) | +0.000 | −0.010 to +0.010 |

The E4B-over-E2B suite gap on the TPU bf16 runs is the same +0.048. E2B's 1.2-point GSM8K
shortfall against bf16 matches what the TPU sweep found for builds from the QAT weights
(`jev-tpu-v5e1/README.md`: E2B QAT bf16 scored 0.897). No bf16 E4B reference exists for GSM8K
or BFCL; the QAT repack on the TPU is the closest, and this build matches it.

"All labels returned" (every answer label among the 32 most likely next tokens) is 96.2% for E2B
and 82.0% for E4B here, against 86.5% and 82.7% on the TPU; the scorer handles a missing label the
same way in every run.

## Run time on this card

Suite 6.4 min (E2B) and 11.2 min (E4B); GSM8K 29.3 and 37.8 min; BFCL 1.3 and 2.1 min.

## Not measured

Answer quality with the drafter on (the E4B drafter left greedy text unchanged on 18 of 18
prompts; E2B's did not), and open-ended chat quality, which none of these tasks scores.
