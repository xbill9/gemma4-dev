# 2026-10-15 — one drafted token under concurrency, mixed prompts, RTX 3050 OEM

**Question:** at 2 drafted tokens the drafter loses at 16–64 requests on mixed prompts. Does 1
drafted token lose less?

**Answer: it loses less but still loses.** At 32–64 requests 1 token runs 0.84–0.98x no drafter
against 0.79–0.92x at 2. For one request it matches 2 tokens on E2B (1.34–1.37x against
1.30–1.38x) and falls behind on E4B (1.49–1.51x against 1.61–1.72x). Neither setting gains at
32–64 requests, so the rigs' settings (E2B and E4B at 2) stand for their one-user serving.

| drafter / no drafter | c=1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E2B greedy, 1 token | 1.37 | 0.97 | 0.94 | 1.25 | 0.98 | 0.91 | 0.84 |
| E2B greedy, 2 tokens | 1.38 | 0.93 | 1.09 | 1.17 | 0.95 | 0.86 | 0.79 |
| E2B sampled, 1 token | 1.34 | 0.96 | 0.92 | 1.20 | 0.97 | 0.92 | 0.85 |
| E2B sampled, 2 tokens | 1.30 | 0.92 | 1.07 | 1.14 | 0.96 | 0.87 | 0.79 |
| E4B greedy, 1 token | 1.49 | 1.12 | 1.03 | 1.26 | 1.06 | 0.98 | 0.87 |
| E4B greedy, 2 tokens | 1.61 | 1.09 | 1.18 | 1.24 | 1.05 | 0.92 | 0.79 |
| E4B sampled, 1 token | 1.51 | 1.11 | 0.98 | 1.23 | 1.04 | 0.98 | 0.88 |
| E4B sampled, 2 tokens | 1.72 | 1.08 | 1.16 | 1.20 | 1.03 | 0.92 | 0.81 |

- **Acceptance at 1 token:** 0.56–0.64, against 0.43–0.50 at 2.
- **The level-to-level pattern is uneven at 2–8 requests for both settings** (1 token behind at 4,
  ahead at 8); with 5 repeats and one pass per arm, differences under about 0.1 between the two
  settings at one level are not read as findings.
- VRAM: E2B 3,552 MiB, E4B 6,306 MiB.

## Method

`run.sh`, job 3 of `../2026-10-15-queue-3050oem`, 18:05–18:36. As
`2026-10-15-concurrency-mixed-3050oem` (`mixed.py`, six prompts, `-c 32768 --parallel 64 -kvu`,
rigs' flags, c = 1–64, 5 repeats, 256 tokens, prompt cache off, thinking off, greedy and sampled,
fresh server per mode) with `--spec-draft-n-max 1`. No-drafter and 2-token figures are from that
run, 11:19–12:20 the same day on the same binary. `summarize.py` → `summary.txt`.
