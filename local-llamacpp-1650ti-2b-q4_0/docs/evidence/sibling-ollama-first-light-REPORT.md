# First light and full sweep — `local-ollama-1650ti-2b-q4_0`, 2026-09-04

Ollama v0.33.2 serving `gemma4:e2b-it-qat` on one GTX 1650 Ti (Max-Q), 4096 MiB, driver 610.57.04.
Machine-readable: `../../reports/2026-09-04-first-light-ollama-1650ti.json`. Harness: `sweep.py` at the
rig root, 3 repeats per cell, prompt cache defeated by `--prompt-mode shuffled`.

**This run exists to be differenced against `local-llamacpp-1650ti-2b-q4_0`'s
`2026-09-03-full-sweep-1650ti`** — same card, same weights, same engine, different daemon. Where a
figure below is compared, the sibling's is quoted beside it.

**Headline: end-to-end is a tie, decode is ~3% behind, and Ollama costs 1144 MiB more VRAM for a
projector this workload never touches.**

## What was running

```
llama-server --model …/blobs/sha256-3646b4c147cd… --port 42151 -c 8192 -np 1
             --no-webui --offline --log-verbosity 4 --no-jinja --chat-template chatml
             --mmproj …/blobs/sha256-58c187648007… --flash-attn auto -b 512 -ub 512
             --context-shift --keep 4
inference compute: library=CUDA compute=7.5 libdirs=ollama,cuda_v12 driver=13.3
```

The daemon starts a `llama-server` child, so this rig and its sibling are running the same binary
family against the same tensors. Three of those flags are the daemon's choices and not ours:
`--mmproj` (unconditional), `--no-jinja --chat-template chatml` (the GGUF's own template is stripped
and replaced by Ollama's Go renderer, `renderer=gemma4 parser=gemma4`), and `--flash-attn auto`
(already on — the sibling had to be told `-fa 1`, which was worth +4.8% there).

## VRAM: 2762 MiB of 4096, and 1154 of it is the projector

| term | MiB | source |
| :--- | ---: | :--- |
| model buffer (weights) | 1341.78 | `load_tensors: CUDA0 model buffer size` |
| KV, non-SWA | 48.00 | 8192 cells × 3 layers |
| KV, SWA | 12.00 | 1024 cells × 12 layers |
| compute buffer | 122.52 | `sched_reserve: CUDA0` |
| **mmproj (worst case)** | **1154.07** | `[mtmd] estimated` |
| CUDA context + slack | ~84 | residual |
| **total** | **~2762** | nvidia-smi: **2762 MiB** |

The sibling holds **1618 MiB** for the same weights at the same context. The whole difference is the
986 MB projector blob, which carries **both** a vision and an audio encoder (`projector: gemma4v`,
then `gemma4a`; `clip_ctx: CLIP using CUDA0 backend` twice). It cannot be turned off from the API.

**`ollama ps` reports 1.7 GB.** It does not count the projector. On a card where capacity is a hard
ceiling this is the single most expensive thing to get wrong, and the daemon's own gauge is the one
that gets it wrong.

Two corrections fall out of the KV rows and are filed where they belong rather than here:
`@MODELS.md` (18 KiB/token is the geometry; `llama_kv_cache_iswa` caps the twelve sliding layers at
the 1024-cell window, so this engine allocates 60 MiB where the section predicts 144), and the
sibling's `CLAUDE.md` (its 1618 MiB derivation is right in total and wrong in both of its last two
terms — two offsetting errors landing 2 MiB from the measurement).

## Context × output length, concurrency 1

Server: `OLLAMA_CONTEXT_LENGTH=8192 OLLAMA_NUM_PARALLEL=1`, matching the sibling's
`-c 8192 --parallel 1`.

**End-to-end — a dead heat, within 1% in all ten cells:**

| in / out | llama.cpp e2e | Ollama e2e | Δ |
| :--- | ---: | ---: | ---: |
| ~100 / 32 | 40.87 | 41.53 | +1.6% |
| ~100 / 128 | 59.72 | 58.93 | −1.3% |
| ~655 / 32 | 12.53 | 12.60 | +0.6% |
| ~655 / 128 | 32.39 | 32.21 | −0.6% |
| ~1275 / 32 | 7.02 | 7.08 | +0.9% |
| ~1275 / 128 | 21.50 | 21.50 | 0.0% |
| ~2520 / 32 | 3.58 | 3.59 | +0.3% |
| ~2520 / 128 | 12.34 | 12.34 | 0.0% |
| ~5025 / 32 | 1.76 | 1.76 | 0.0% |
| ~5025 / 128 | 6.50 | 6.50 | 0.0% |

That is what "same engine" looks like. At these shapes end-to-end is prefill-dominated, and prefill
measured **~318 t/s on both rigs** at the same ~655-token prompt.

**Decode, corrected for chunk batching (see the next section), output 128:**

| input tok | llama.cpp | Ollama | Δ |
| ---: | ---: | ---: | ---: |
| ~100 | 72.55 | 70.37 | −3.0% |
| ~655 | 70.78 | 69.27 | −2.1% |
| ~1275 | 70.27 | 68.20 | −2.9% |
| ~2520 | 68.64 | 66.48 | −3.1% |
| ~5025 | 66.81 | 64.95 | −2.8% |

Consistent, small, and **an upper bound on the daemon's overhead rather than a property of its
decoder** — it is the same decoder. The extra HTTP hop through the Go frontend and the renderer are
the obvious candidates; nothing here isolates them.

Decode falls ~8% across a 50× increase in context on both rigs. Context is not the lever;
prompt length is, and it acts through prefill.

## The correction: chunks are not tokens, and the two engines batch differently

`sweep.py`'s streaming decode figure counts inter-**chunk** gaps. `chunks_match_usage` is `false` in
every cell of both sweeps, and the ratios differ:

| output | llama.cpp tok/chunk | Ollama tok/chunk |
| ---: | ---: | ---: |
| 32 | 1.103 | **1.185** |
| 128 | 1.024 | **1.067 – 1.143** |

**Taken as run, the two sweeps say 71.15 vs 61.52 tok/s at the shortest prompt — an 11% gap that is
8 points harness and 3 points hardware.** The corrected column above divides `completion_tokens` by
the measured inter-token span instead of counting chunks.

`end_to_end_tps` is immune: it is tokens over wall time and never touches the chunk count. **When the
two disagree, the end-to-end figure is the one to trust.** The `output=32` rows are the worst
affected because the series is only ~27 gaps long — quote the 128-token rows.

## Every cell measured the model thinking — identically on both sides

`content_chunks` is **0** and `reasoning_chunks` is the entire stream in every cell of both sweeps
(2156 here, 2310 in the sibling). At 32 and 128 output tokens Gemma 4 has not closed its thinking
block. **The tok/s figures are real; no cell in either rig completed a task.** That is a limitation
of both runs equally, which is what keeps the comparison fair — but a sweep that wants completed
answers on this model needs output budgets in the high hundreds.

## Concurrency at 512 in / 128 out — and it had to be run twice

**llama.cpp's concurrency server was `-c 32768 --parallel 32`: 1024 tokens per slot, all of it in
VRAM, 1818 MiB total. That configuration could not be requested while serving the stock tag.**
Ollama multiplies `OLLAMA_CONTEXT_LENGTH` by `OLLAMA_NUM_PARALLEL` — the opposite of llama.cpp,
which divides — and it applies a minimum per slot:

```
OLLAMA_CONTEXT_LENGTH=1024 NUM_PARALLEL=32  ->  llama-server -c 65536 -np 32   (2048/slot, DOUBLED)
OLLAMA_CONTEXT_LENGTH=1024 NUM_PARALLEL=8   ->  llama-server -c 16384 -np 8    (2048/slot, DOUBLED)
OLLAMA_CONTEXT_LENGTH=8192 NUM_PARALLEL=1   ->  llama-server -c 8192  -np 1    (as asked)
```

> **CORRECTED 2026-09-04, same day: that minimum applies to MULTIMODAL MODELS ONLY.** This report
> originally recorded it as a property of the daemon, uncharacterised. Four further probes settled
> it: the stock tag took 1024 → 2048 and 1536 → 2048, while the projector-free
> `gemma4:e2b-it-qat-text` honoured 1024, 1536 and 8192 exactly, including 1024 × 32 →
> `-c 32768 -np 32`. 8192 clears the floor either way, which is why the context sweep never saw it.
> **So the projector cost 1154 MiB of encoder AND a forced 192 MiB of extra KV at 32 slots**, and
> that is what pushed this configuration off the card. The like-for-like run this one could not be
> is `2026-09-04-text-only-ollama-1650ti`.

### The 32-slot attempt: silent partial offload, recorded as `status: failed`

Doubled slots plus the 1154 MiB projector do not fit 4096 MiB, and **nothing failed** — llama.cpp
moved work to the host:

```
load_tensors: offloaded 26/36 layers to GPU              <- ten transformer layers on the CPU
llama_kv_cache: CUDA0 KV 128.00 MiB   CPU KV 256.00 MiB  <- twice, one per cache
ollama ps: 4.2 GB   60%/40% CPU/GPU
```

Single-stream decode halved — **34.74 tok/s per stream against the sibling's 69.89** — and aggregate
peaked at 32.06 tok/s. Kept as `concurrency.np32-partial-offload.json` and carried in the report
with `status: failed`, because it measures a budget failure and not the runtime. **There is no
configuration knob that prevents this**; `OLLAMA_GPU_OVERHEAD` is a nudge, and the daemon owns the
offload decision. It can only be detected.

### The 8-slot control: 36/36 layers on GPU, all KV in VRAM, 2900 MiB of 4096

| c | Ollama agg | llama.cpp agg | Ollama TTFT | llama.cpp TTFT | Ollama /stream | llama.cpp /stream |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32.16 | 32.74 | 2141 | 2135 | 63.09 | 69.89 |
| 2 | **32.13** | 40.59 | 4193 | 4062 | 30.93 | 55.27 |
| 4 | 35.17 | 45.12 | 8416 | 7982 | 19.26 | 36.86 |
| 8 | 45.98 | 44.79 | 16053 | 16212 | 19.49 | 18.74 |

**TTFT tracks the sibling within 5% at every level, and doubles exactly with every doubling of `c` on
both rigs** — 2141 → 4193 → 8416 → 16053. That is the same serial prefill in both, and it is the
finding that matters: **prefill does not batch, on either runtime.** Aggregate output saturates near
**46 tok/s** here and **45–48** there.

Two caveats on the middle of the curve, and neither is resolved:

- **The c=2 cell is not a result.** Spread across three repeats is **23.7%**; every other cell in
  either run is ≤0.5%. This is the Max-Q thermal behaviour the sibling documents — it saw a c=8 cell
  at 7.7% and had a c=32 cell invert on a single-sample pass. Do not read 32.13.
- **The c=4 gap (35.17 vs 45.12) is real** — 0.4% spread — **and it is not isolated.** The two servers
  do not have the same slot geometry: 2048 tokens per slot here against 1024 there, because this
  daemon will not accept 1024. Scheduler differences and slot geometry are confounded, and this run
  cannot separate them.

## Method notes

- **Prompt cache.** The sibling used `cache_prompt: false`; Ollama drops unknown fields in its
  OpenAI translation, so that flag never reaches the engine (measured: `cached n_tokens = 660` of
  661 with the flag set). This run used `--prompt-mode shuffled`, which reorders the filler so there
  is no reusable prefix, keeping the same vocabulary so cells land on the same token counts. Verified
  cold: 2.178 s median for a 652-token prompt, ~318 t/s, five repeats, all within 1%.
  **The two runs therefore defeated the cache by different means**, both verified to `cached = 0`.
- **`metrics.prom` is empty in both runs.** Ollama serves no Prometheus endpoint — `GET /metrics`
  returns 404. `llama-server` has one. Not a harness failure.
- **`--decode-source auto` resolved to `stream`**, correctly: Ollama emits no
  `usage.decode_tokens_per_second`, which is an invention of this repo's own JAX and PyTorch servers.
- **Three harness bugs were fixed before any number existed here.** The reasoning field name (which
  cost one discarded run, kept as `sweep.DISCARDED-zero-decode.json`), the prompt-cache flag, and a
  deterministically seeded prompt generator that made a second process hit the first one's cache.
  See `CLAUDE.md`, "`sweep.py` is a copy of the sibling's and needed three more fixes".

## What this run does not establish

- **Nothing about llama.cpp vs Ollama as engines.** It is one engine. Every difference measured here
  is packaging, defaults, or resource policy.
- **Nothing transferable to the T4-based rigs.** TU117 has no tensor cores; `g4dn`/`g5g` are TU104
  and do, at the same compute capability 7.5.
- **Nothing about a completed task.** Every cell measured the thinking phase.
- **Nothing about the c=4 divergence**, for the reason given above.
