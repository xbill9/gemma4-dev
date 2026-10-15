# 2026-10-15 — concurrency with mixed chat prompts, drafter against none, RTX 3050 OEM

**Question:** `2026-10-14-concurrency-drafter-3050oem` found the drafter a clear win up to 8
requests at once, on `sweep.py`'s repeated-paragraph prompt, which the drafter guesses unusually
well (acceptance 0.73–0.83). What does it do on ordinary chat prompts?

**Answer: on mixed prompts the drafter pays clearly only for one request at a time.** E4B gains
1.09–1.24x at 2–8 requests, E2B 0.93–1.17x, and both lose 8–21% at 32–64. Time to first token is
lower with the drafter at 2 requests and up. Acceptance holds at 0.43–0.50 at every level.

**Second finding: llama-server's default per-slot KV cache halves decode speed at random under
load on this build.** With one shared cache (`-kvu`) throughput at 16–64 requests is
higher without the drafter (1.2–1.9x) on these short requests. Every number below is from the
shared-cache rerun. On longer requests the shared cache is slower, and the 2026-10-14 sweep's
drafter ratios hold with it (`../2026-10-15-concurrency-drafter-kvu-3050oem`).

## Drafter / no drafter, shared KV cache

| requests at once | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E2B greedy | 1.38 | 0.93 | 1.09 | 1.17 | 0.95 | 0.86 | 0.79 |
| E2B sampled | 1.30 | 0.92 | 1.07 | 1.14 | 0.96 | 0.87 | 0.79 |
| E4B greedy | 1.61 | 1.09 | 1.18 | 1.24 | 1.05 | 0.92 | 0.79 |
| E4B sampled | 1.72 | 1.08 | 1.16 | 1.20 | 1.03 | 0.92 | 0.81 |

Aggregate output tok/s, greedy (median of 5, spread in `summary.txt`):

| requests at once | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E2B none | 106.1 | 188.7 | 244.6 | 297.1 | 478.1 | 559.9 | 569.5 |
| E2B drafter | 146.7 | 174.9 | 267.3 | 348.6 | 454.1 | 482.2 | 447.6 |
| E4B none | 57.3 | 110.0 | 153.1 | 199.3 | 331.9 | 423.0 | 472.4 |
| E4B drafter | 92.5 | 119.8 | 180.0 | 246.2 | 347.9 | 389.7 | 375.3 |

- **Acceptance** (accepted over drafted tokens, all requests at a level): E2B 0.46–0.50 greedy and
  0.43–0.46 sampled, E4B 0.48–0.50 and 0.46–0.50, against 0.48 on the same six prompts one at a
  time in `2026-10-14-drafter-gaps-3050oem`.
- **Time to first token**, median: the drafted arm is lower at every level from 2 up, by 6–48%
  (E4B greedy at 8 requests: 341 against 655 ms; E2B greedy at 64: 1,381 against 1,697 ms).
- **Spread at one request is the prompt mix** (37–48% drafted): each repeat cycles to a different
  prompt, and the drafter's gain runs from story to code. Spreads at 8 requests and up are 2–11%.
- **VRAM** at `-c 32768 --parallel 64 -kvu`: E2B 3,340 / 3,616 MiB, E4B 6,068 / 6,372 MiB
  without / with the drafter. E4B with the drafter fits 64 slots here.

**Against the repeated-paragraph sweep:** its drafter/none at 8 requests was 1.29–1.53x (E2B) and
1.40–1.77x (E4B); here 1.14–1.24x. Two things differ besides the prompts: `sweep.py` turns thinking
on, and that run used per-slot caches.

## The per-slot KV cache

The first pass of this run (`perslot-kv/`) used llama-server's default, a separate KV cache per
slot (`kv_unified = 'false'` in the log). Its sampled cells ran at half the speed of greedy at the
same level, with spreads up to 81%, and E4B at 2 requests gave 52 tok/s against 57 for one.

Traced in `seedtest/` on E4B with no drafter, `-c 32768 --parallel 64`:

- **While it happens, decoding two requests takes 37.7 ms a step against 17.2 for one, and the
  server stops reusing its compute graph** ("graphs reused" stops climbing in the log).
- **Seeds, sampling and streaming do not cause it** (`seedtest.py`, `isolate.py`): every seed
  setting ran at 112 tok/s for two requests on a fresh server.
- **It depends on which slots the server picks, and switches on and off with request history**
  (`bisect1.txt`): two requests ran at 111 tok/s after 1 and 2 at once, 52 after 4, 8 and 16, and
  110 after 32. A burst of 2 alternating prompts (`order.py`) never triggered it.
- **With `-kvu` it does not happen** (`bisect-kvu.txt`): graphs reused after every level, and the
  aggregate at 16 / 32 / 64 was 356 / 445 / 489 tok/s against 243 / 254 / 270 per-slot. Two
  requests still slowed from 113 to 81 tok/s as old conversations filled the shared cache.

Unified against per-slot, same arms (rerun ÷ first pass), greedy: E2B no drafter 0.95–1.05x up to
8 requests and 1.30–1.38x at 16–64; E4B no drafter 0.94–1.09x and 1.54–1.79x. Sampled cells,
which the stall hit hardest, gain up to 2.13x.

Which part of llama.cpp `77c4540ce` causes it was not traced. **The rigs serve one user
(`--parallel 1`), so serving is unaffected**. Rerunning `2026-10-14-concurrency-drafter-3050oem`
with `-kvu` (`../2026-10-15-concurrency-drafter-kvu-3050oem`) gave 0.55–0.79x its output at 64
requests: with up to 640 tokens per request the shared cache costs more than the stall, so
neither setting is right for every load.

## Method

`run.sh`, 2026-10-15 11:19–12:20, llama.cpp `77c4540ce`. One llama-server per (model, arm, mode),
fresh for each mode, at `-c 32768 --parallel 64 -kvu` (512 tokens per slot; `kvu-server.sh` adds
`-kvu` to `../2026-10-14-overnight-3050oem/lib.sh`'s `serve`) with the rigs' flags: `-fa 1`, f16
KV, `per_layer_token_embd` on the card, `--reasoning off`; drafted arms add the exact Q4_0 drafter
at 2 tokens. `mixed.py`: c = 1–64, 5 repeats, each request one of the six prompts of
`2026-10-14-drafter-gaps-3050oem/speed.py` (rotating by repeat), 256 tokens out, prompt cache off,
thinking at the server default (off), streamed; greedy, and sampled at temperature 0.8 with one
seed per request. Aggregate is total output tokens over the batch's wall time; tokens, draft
counts and timings come from the server's final stream chunk. `summarize.py` → `summary.txt`.
The first pass (`perslot-kv/`, 09:40–10:57) is the same without `-kvu`, one server per arm.
