# 2026-10-15 — concurrency with long mixed prompts, drafter against none, both KV cache layouts, RTX 3050 OEM

**Question:** `2026-10-15-concurrency-mixed-3050oem` measured the drafter on six short chat
prompts (under 300 tokens) and found it pays clearly only for one request at a time, and that
the per-slot KV cache halved decode speed at random. What happens when every request carries
1,000–3,500 tokens of document?

**Answer: with long prompts the drafter pays at every level up to 16 requests and breaks even at
32.** Drafter over none, both passes:

| requests at once | 1 | 2–16 | 32 |
|---|---:|---:|---:|
| E2B, per-slot cache | 1.34 | 1.02–1.14 | 1.02–1.04 |
| E2B, shared cache (`-kvu`) | 1.26 | 1.04–1.13 | 0.96–0.97 |
| E4B, per-slot cache | 1.54 | 1.08–1.16 | 1.03–1.05 |
| E4B, shared cache (`-kvu`) | 1.61 | 1.08–1.18 | 0.99 |

Acceptance is 0.56–0.62 at every level, against 0.43–0.50 on the short prompts: the drafter
guesses better with a document in context.

**Second finding: on long requests the per-slot cache is the faster layout.** Shared over
per-slot output is 0.92–1.01x for one request, 0.78–0.87x at 8, 0.68–0.82x at 16 and 0.49–0.66x
at 32, and time to first token is 1.39–2.00x per-slot's from 8 requests up. The random per-slot
slowdown seen on short prompts did not show here: per-slot beat shared in 204 of 232 paired
batches (same prompts, same level, same repeat), and every batch where it lost is a batch with
prompts 5 and 0 together at 2 or 4 requests, losing by the same amount in both passes and both
models.

## Aggregate output tok/s, pass 1 (median of 6; pass 2 in `summary.txt`)

| requests at once | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| E2B per-slot, none | 81.0 | 113.1 | 126.7 | 140.8 | 157.8 | 168.6 |
| E2B per-slot, drafter | 108.5 | 124.5 | 136.5 | 160.0 | 165.1 | 175.8 |
| E2B shared, none | 79.3 | 105.6 | 114.0 | 112.8 | 109.6 | 89.6 |
| E2B shared, drafter | 99.8 | 112.8 | 128.8 | 125.4 | 116.3 | 86.4 |
| E4B per-slot, none | 45.0 | 69.1 | 76.6 | 90.2 | 101.6 | 111.7 |
| E4B per-slot, drafter | 69.4 | 75.9 | 89.0 | 103.9 | 109.7 | 116.7 |
| E4B shared, none | 43.8 | 64.5 | 75.3 | 77.3 | 82.6 | 72.2 |
| E4B shared, drafter | 70.4 | 74.5 | 87.0 | 90.2 | 89.3 | 71.4 |

- **Shared cache output falls past 4–16 requests**, while per-slot keeps climbing to 32. At 32
  requests per-request decode on E4B is 3.5–3.7 tok/s shared against 5.5–6.5 per-slot.
- **Spread at 1–4 requests is the prompt mix** (up to 49%): each repeat starts from a different
  prompt, and prompt lengths run 1,025–3,537 tokens. The same repeat gives the same rate in both
  passes (per-repeat pass 2 / pass 1 within 4.3% at 2–4 requests). Spreads at 32 are 0.2–5.8%.
- **Order does not matter:** pass 2 ran the arms in reverse order and lands at 0.96–1.03x pass 1
  in every cell.
- **Time to first token** grows with the queue as before (prefill serves one request at a time):
  E4B per-slot without the drafter 1.2 s at one request, 22.4 s at 32. The drafter adds 3–4% at
  one request.
- **VRAM peak** at `-c 131072 --parallel 32` (4,096 tokens per slot): E2B 3,845–4,061 MiB, E4B
  7,101 (shared, no drafter) to 7,741 MiB (per-slot, drafter) of 8,192. 64 slots at these lengths
  do not fit for E4B.

**Against the short mixed prompts** (`../2026-10-15-concurrency-mixed-3050oem`, shared cache,
`-c 32768 --parallel 64`): at 32 requests the drafter there gave 0.86x (E2B) and 0.92x (E4B),
here 0.96–0.99x shared and 1.02–1.05x per-slot. At 2–8 requests there E2B ran 0.93–1.17x, here
1.07–1.14x. The prompts, context per slot and slot count all differ, so the comparison is
indicative; the higher acceptance fits the gain.

## Stability check

A pass-2 repeat of E4B per-slot with the drafter ran 37 times back to back
(`e4b/perslot-q4x-n2/repeats-pass2-extra.jsonl`, kept apart from `results.jsonl`). Across those
runs each cell's median moved 0.4–1.9% at 1–16 requests and 5.4% at 32.

## Gaps

Pass 2 at 32 requests has 2 repeats instead of 6 for E4B shared without the drafter and E4B
per-slot with it; the other 4 were not recorded. Both cells agree with pass 1 (1.00x and 0.96x).

## Method

`run.sh` pass 1 2026-10-15 22:03–00:05, `run.sh` + `run-pass2.sh` pass 2 2026-10-16 00:07–11:21,
llama.cpp `77c4540ce`. One llama-server per (model, cache layout, drafter, pass) at
`-c 131072 --parallel 32` through `../2026-10-14-overnight-3050oem/lib.sh`'s `serve`, with the
rigs' flags: `-fa 1`, f16 KV, `per_layer_token_embd` on the card, `--reasoning off`; drafted arms
add the exact Q4_0 drafter at 2 tokens; shared arms go through `kvu-server.sh`, which adds `-kvu`.
`fit.sh` first checked the heaviest arm of each layout fits at 32 requests (`fit.out`).
`longmixed.py`: six prompts, each a different 1,000–3,500-token stretch of the NVIDIA driver README
(`/usr/share/doc/NVIDIA_GLX-1.0/README.txt`, SHA-256 in
`../2026-10-14-drafter-gaps-3050oem/doc.sha256`) followed by a different task (summary, option
list, bash script, explanation, quiz, FAQ), saved to `prompts.json` so every arm sends identical
text; 1,025 / 3,537 / 1,534 / 3,028 / 2,028 / 2,525 prompt tokens. c = 1, 2, 4, 8, 16, 32, six
repeats, request k of repeat r gets prompt (k + r) mod 6. 256 tokens out, greedy, prompt cache off,
thinking off, streamed, one unrecorded warm-up. Aggregate is total output tokens over the
batch's wall time; tokens, draft counts and timings come from the server's final stream chunk.
Pass 1 ran the eight arms in order, pass 2 in reverse. `summarize.py` → `summary.txt`.
