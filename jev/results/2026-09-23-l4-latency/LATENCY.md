# Latency pass, run 2026-09-23-l4-latency (computed by the script in the commit)
Client on the instance against localhost, concurrency 1, first 100 examples per task. Milliseconds.

| Task | Plain, median | Plain, 90th pct | Diffusion one read, median | Diffusion one read, 90th pct | Diffusion 4 reads (first + 3 parallel), median |
|---|---|---|---|---|---|
| sst2 | 61 | 62 | 118 | 119 | 306 |
| ag_news | 61 | 62 | 119 | 120 | 307 |
| emotion | 61 | 62 | 119 | 120 | 308 |
| irony | 61 | 61 | 121 | 122 | 312 |

## Where DiffusionGemma's probability outside the labels goes

Highest returned token at the answer slot that is not an allowed label, counted over the four reads of each example; share of reads, top 5 per task.

- sst2 (median label mass 0.488): '<eos>' 40%, ' skipped' 0%
- ag_news (median label mass 0.372): '<eos>' 53%, ' sc' 1%, ' Which' 0%, ' business' 0%, ' biology' 0%
- emotion (median label mass 0.393): '<eos>' 64%, ' sadness' 0%
- irony (median label mass 0.548): ' the' 24%, '<eos>' 8%, ' not' 0%
