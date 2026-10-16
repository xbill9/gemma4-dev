# 2026-10-15 — mixed-prompt concurrency with thinking on, RTX 3050 OEM

**Question:** the drafter gained more in the repeated-paragraph sweep (job A, thinking on) than in
`2026-10-15-concurrency-mixed-3050oem` (mixed prompts, thinking off). How much of that is the
thinking setting?

**Answer: about half.** With thinking on, the same mixed prompts raise acceptance from 0.46–0.50
to 0.54–0.62, and the drafter's gain with them: E4B 1.29–1.47x at 2–8 requests (thinking off:
1.09–1.24x), E2B 1.04–1.31x (0.93–1.17x). Job A's repeated paragraph, at acceptance 0.73–0.83,
gains more again. The loss at 32–64 requests remains: 0.82–0.98x.

| drafter / no drafter, greedy | c=1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E2B, thinking on | 1.46 | 1.04 | 1.20 | 1.31 | 1.02 | 0.91 | 0.82 |
| E2B, thinking off | 1.38 | 0.93 | 1.09 | 1.17 | 0.95 | 0.86 | 0.79 |
| E4B, thinking on | 1.87 | 1.29 | 1.29 | 1.47 | 1.13 | 0.98 | 0.85 |
| E4B, thinking off | 1.61 | 1.09 | 1.18 | 1.24 | 1.05 | 0.92 | 0.79 |

- **Acceptance with thinking on:** E2B 0.54–0.57, E4B 0.58–0.62 across levels.
- **Every reply ran the full 256 tokens**, most of it thinking text; no-drafter output is within
  3.5% (E2B) and 8.9% (E4B) of thinking off at every level, so the thinking setting mainly changes how well the drafter guesses,
  not the model's speed.
- Time to first token (first thinking or answer token) is lower with the drafter at every level.

## Method

`run.sh`, job 2 of `../2026-10-15-queue-3050oem`, 17:34–18:05. As
`2026-10-15-concurrency-mixed-3050oem` (`mixed.py`, six prompts, `-c 32768 --parallel 64 -kvu`,
rigs' flags, c = 1–64, 5 repeats, 256 tokens, prompt cache off) with `--thinking`, which sends
`chat_template_kwargs: {"enable_thinking": true}` as `sweep.py` does; greedy only, one server per
(model, arm). Thinking-off figures are from that run. `summarize.py` → `summary.txt`.
