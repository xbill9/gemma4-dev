# Full sweep — `local-llamacpp-1650ti-2b-q4_0`, 2026-09-03

Concurrency at a fixed 512/128 shape, plus context × output-length at concurrency 1.
Machine-readable: `../../reports/2026-09-03-full-sweep-1650ti.json`. Harness: `sweep.py` at the rig
root, 3 repeats per cell, prompt cache defeated.

**Headline: decode is fast and flat; prefill is serial, and prefill is what you actually pay.**

## Concurrency — 512 in, 128 out

| c | aggregate tok/s | spread | TTFT ms | TPOT ms | per stream |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32.74 | 0.2% | 2135 | 14.31 | 69.89 |
| 2 | 40.59 | 0.3% | 4062 | 18.09 | 55.27 |
| 4 | 45.12 | 0.2% | 7982 | 27.13 | 36.86 |
| 8 | 44.79 | **7.7%** | 16212 | 53.35 | 18.74 |
| 16 | 45.89 | 0.5% | 32551 | 96.97 | 10.31 |
| 32 | 48.19 | 0.1% | 61825 | 184.36 | 5.42 |

**TTFT doubles exactly with every doubling of `c`.** 2135 → 4062 → 7982 → 16212 → 32551 → 61825 is
prefill running strictly one request at a time, at ~252 t/s however many clients are queued.
Aggregate output therefore saturates at **45–48 tok/s, only 1.47× from c=1**. Past c=4, more
clients buy latency and nothing else.

**This is not in conflict with `llama-batched-bench`'s 3.8×** (73 → 278 tok/s, B=1 → 64). That tool
reports `S_TG`, the decode phase in isolation, and decode genuinely does batch. But at 512 in /
128 out, prefill is 4× the token volume of decode — the phase that scales is the minority of the
work. **Do not quote the decode number without naming the phase.**

The 7.7% spread at c=8 is the only wide cell; the other five are ≤0.5%. That is the `--repeats`
fix doing its job — an earlier single-sample pass put B=32 *below* B=16 and an identical re-run did
not reproduce it.

## Context × output length, concurrency 1

| input tok | out 32 e2e | out 128 e2e | decode tok/s | implied prefill t/s |
| ---: | ---: | ---: | ---: | ---: |
| 108 | 40.87 | 59.72 | 71.15 | ~324 |
| 661 | 12.53 | 32.39 | 69.50 | ~316 |
| 1279 | 7.02 | 21.50 | 68.51 | ~305 |
| 2527 | 3.58 | 12.34 | 67.23 | ~295 |
| 5027 | 1.76 | 6.50 | 65.30 | ~284 |

**Decode barely notices context: −8.3% across a 46× increase** (71.15 → 65.23 tok/s). End-to-end
falls 23× over the same range, entirely from prefill. If you care about this rig's responsiveness,
the prompt length is the lever; the generation length is not.

## Every cell measured the model thinking

`content_chunks` is **0** and `reasoning_chunks` is 29–125 in every context cell. At 32 and 128
output tokens Gemma 4 had not finished its thinking block, so `text_head` is empty throughout.
**The tok/s figures are real; no cell completed a task.** A sweep that wants completed answers on
this model needs output budgets in the high hundreds — see `CLAUDE.md`, "Gemma 4 reasons".

## What was thrown away, and why

**The first concurrency run was discarded.** `llama-server`'s prefix cache answered **656 of 661**
prompt tokens despite the harness's leading nonce — it reuses cached KV either side of a small
mismatch rather than requiring an exact common prefix, which is a real difference from vLLM and the
assumption the harness was built on. Prefill read as 38 ms instead of 2080 ms, unevenly across
cells, so the curve was not obviously wrong — it was *incoherent*, which is worse. Kept as
`concurrency.DISCARDED-prompt-cache.json` beside this file.

Direct measurement, same prompt:

```
cache_prompt=true   prompt_tokens=661  cached=660  prefill   38 ms
cache_prompt=false  prompt_tokens=661  cached=0    prefill 2080 ms  (318 t/s)
```

Three further harness bugs were fixed to get any number at all; all were present in the sibling
copies and were propagated the same day. See `CLAUDE.md`, "`sweep.py` came from a sibling".

## Caveats

- **Two servers.** llama.cpp splits `--ctx-size` across `--parallel` slots, so concurrency ran on
  `-c 32768 --parallel 32` (1024 tok/slot) and context on `-c 8192 --parallel 1`. A single server
  cannot do both without silently queueing or truncating.
- **`stream_chunks` ≠ `usage.completion_tokens`** (29 vs 32): llama-server does not emit exactly one
  content delta per token. Decode rate comes from inter-token gaps and is unaffected, but a token
  *count* taken from chunks would read ~9% low. Recorded per cell as `chunks_match_usage: false`.
- **Max-Q thermals.** This part throttles; single samples are not safe here.
- **Not comparable to the T4 rigs** at equal compute capability 7.5 — TU104 has tensor cores, TU117
  does not.
