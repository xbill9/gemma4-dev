# 2026-09-16 — paired CPU/GPU sweep, Gemma 4 E2B q4_0

**The first controlled A/B in this family.** One GGUF, one llama.cpp commit, one
port, one harness, one prompt set. The arms were run alternately and the only
thing that differed between them is the device.

## Result

GPU decode is **4.27x** the CPU, prefill **3.63x**, end-to-end
**3.81x** (medians over 8 paired cells).

| in tok | out tok | CPU decode | GPU decode | **x** | CPU TTFT ms | GPU TTFT ms | **x** | CPU e2e | GPU e2e | **x** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 94 | 32 | 17.61 | 71.22 | **4.04** | 1143 | 385 | **2.97** | 11.67 | 41.11 | **3.52** |
| 94 | 128 | 17.22 | 71.20 | **4.13** | 1219 | 385 | **3.17** | 15.18 | 60.15 | **3.96** |
| 516 | 32 | 16.16 | 69.93 | **4.33** | 5978 | 1610 | **3.71** | 4.14 | 15.92 | **3.84** |
| 516 | 128 | 16.39 | 69.49 | **4.24** | 5995 | 1612 | **3.72** | 9.45 | 37.70 | **3.99** |
| 998 | 32 | 15.81 | 68.60 | **4.34** | 11748 | 3244 | **3.62** | 2.37 | 8.76 | **3.70** |
| 998 | 128 | 16.10 | 68.52 | **4.26** | 11532 | 3242 | **3.56** | 6.62 | 25.34 | **3.83** |
| 1959 | 32 | 15.66 | 67.61 | **4.32** | 23926 | 6522 | **3.67** | 1.24 | 4.61 | **3.71** |
| 1959 | 128 | 15.80 | 67.64 | **4.28** | 23773 | 6522 | **3.65** | 4.04 | 15.32 | **3.79** |

- **decode** — client-side inter-token rate off the SSE stream, `--decode-source
  auto -> stream` on both arms (llama-server emits no `usage.decode_tokens_per_second`).
- **TTFT ratio** is CPU/GPU, so it reads the same direction as the others: higher
  is a bigger GPU win.
- **e2e** carries prefill and the HTTP round trip, which is why it collapses with
  context on both arms while decode barely moves.

## Is the difference real?

Within-cell spread over 3 repeats: CPU max 6.29% (median 1.43%), GPU max
0.84% (median 0.41%). The decode ratio spans 4.04-4.34 across
every cell — an order of magnitude clear of that noise. The ratio is not a
single number, though: it climbs with context (4.04 at 94 tokens, 4.34 at 998),
because the CPU arm loses ground as the KV grows while the GPU arm barely does.

**Decode is flat in context on both arms** — CPU 15.66-17.61 tok/s
and GPU 67.61-71.22 tok/s across a 21x range of prompt
length — while TTFT is linear in it on both. That is the expected shape: decode
is bandwidth-bound per token, prefill is compute-bound per prompt.

## What was controlled

| | |
| --- | --- |
| checkpoint | same file, `gemma-4-E2B_q4_0-it.gguf`, byte-identical path |
| engine | llama.cpp `c6824a9`, both arms rebuilt from that commit for this run |
| flags | `-c 8192 -ctk f16 -ctv f16 -fa 1 -t 4 -tb 8 --parallel 1 --metrics`, differing ONLY in `-ngl` (0 vs 99) |
| prompts | device-neutral filler, `sweep.py` byte-identical in both rigs. All 8 cells paired on exactly equal `input_len` — 94/516/998/1959 |
| endpoint | `127.0.0.1:8080`, one arm at a time |
| arm identity | attested from `/proc` and recorded in each report, not passed in |

CPU arm exe `61432a48569c1acf`, GPU arm exe `a1b368b993e5efbc`.

## What was NOT controlled

Read a 4x as real and a 5% as nothing.

- **Order and thermals.** CPU arm ran first and saturated 12 cores; an i7-1360P
  and a Max-Q card share one thermal envelope, so the GPU arm started on a warm
  package. A fixed 120 s cooldown separated them — sized, not measured. Order
  effects would, if anything, understate the GPU arm here.
- **Page cache.** The GGUF was hot for both arms (the CPU arm had just read it).
  Not a dropped-cache measurement; it says nothing about cold-start cost, and the
  lazy-PLE claim remains untested on this host.
- **CPU affinity.** Nothing was pinned. A prior `llama-bench` sweep found
  affinity worth **1.61x on prefill** on this die, so the CPU arm's prefill here
  is NOT its best achievable number — the 3.63x prefill ratio is an upper bound on
  the true device gap, and could narrow substantially with pinning.
- **Concurrency.** Single stream only (`--parallel 1`). The GPU arm's own earlier
  work found concurrency to be its largest lever; nothing here exercises it.
