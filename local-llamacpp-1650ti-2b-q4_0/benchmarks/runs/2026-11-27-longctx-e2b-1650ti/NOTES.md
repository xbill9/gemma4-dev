# E2B long context on the GTX 1650 Ti — run notes

The 3050's E2B long-context study (`local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-11-25-longctx-e2b-3050oem`)
on the `yogagpu` laptop, started 2026-11-27, at the two depths the user chose: 16,384 and 32,768 tokens.
All six arms (f16-draft, q8-draft, q4-draft, f16-sampled-off, f16-sampled-on, f16-none), 10 seeded trials x 2
tasks per depth, prompts byte-identical to the 3050's (its NVIDIA README is copied here as
`nvidia-readme-3050.txt`; this laptop's driver README differs).

Same as the 3050: llama.cpp `7182ef7`, exact Q4_0 E2B (`419db9a6…`), `-c 131072`, `--parallel 1`, prompt cache off,
greedy or Google's sampling as per arm.

Two settings are this card's own: the per-layer table stays in system RAM (no `-ot`), and the drafter drafts 1
token (the 3050 drafted 2). `f16-none` and the two sampled arms use no drafter, so they compare across cards
with nothing changed but the card.

Why only two depths: measured 2026-11-27 with llama-bench, this card reads 241 tok/s at 16k and 185 at 32k,
12-13x slower than the 3050; the full five depths would take about 105 h. These two take about 8-9 h (estimate).

Is it running: `pgrep -af 'run\.sh$'; tail -5 driver.log; wc -l longctx.jsonl`.
Resume: `setsid nohup systemd-inhibit --what=sleep:idle:handle-lid-switch --why="E2B long context" ./run.sh >> nohup.out 2>&1 < /dev/null &`
(longctx.py skips trials already in longctx.jsonl).
