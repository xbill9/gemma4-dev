# Text-only, and the like-for-like — `local-ollama-1650ti-2b-q4_0`, 2026-09-04

Ollama v0.33.2 serving **`gemma4:e2b-it-qat-text`** — the stock tag with the projector layer
dropped — on one GTX 1650 Ti (Max-Q), 4096 MiB.
Machine-readable: `../../reports/2026-09-04-text-only-ollama-1650ti.json`.

**This is the run that `2026-09-04-first-light-ollama-1650ti` could not be.** That one asked for the
llama.cpp sibling's concurrency geometry and did not get it: the stock tag's projector forced 2048
tokens per slot, and 32 of those did not fit the card. Projector-free, the same request is honoured
and both servers run **`-c 32768`, 32 slots of 1024 tokens, 36/36 layers on GPU, all KV in VRAM.**

## Dropping the projector: what it costs and what it buys

| | resident @ 8192 ctx, 1 slot | capabilities |
| :--- | ---: | :--- |
| `gemma4:e2b-it-qat` (stock) | 2762 MiB | completion **vision audio** tools thinking |
| `gemma4:e2b-it-qat-text` | **1612 MiB** | completion tools thinking |
| llama.cpp sibling, same weights | 1618 MiB | — |

**−1150 MiB, and 6 MiB below the sibling.** The variant is `ollama show --modelfile` of the stock
tag with the second `FROM` removed and nothing else touched — same model blob **by digest**, same
`RENDERER gemma4` / `PARSER gemma4`, same three sampling parameters. It shares the blob, so it costs
no extra disk. Decode is unchanged (69.81 tok/s server-side gauge against 68.95 stock).

### SETTLED: the 2048-token per-slot floor applies to multimodal models only

The first-light run recorded a "per-sequence floor" and could not say what caused it. Four probes
settle it:

| model | asked | child got | per slot |
| :--- | :--- | :--- | ---: |
| stock | 1024 × 1 | `-c 2048 -np 1` | **2048** |
| stock | 1536 × 1 | `-c 2048 -np 1` | **2048** |
| stock | 8192 × 1 | `-c 8192 -np 1` | 8192 |
| text-only | 1024 × 1 | `-c 1024 -np 1` | 1024 |
| text-only | 1536 × 1 | `-c 1536 -np 1` | 1536 |
| text-only | 1024 × 32 | `-c 32768 -np 32` | 1024 |

**A projector-carrying model gets a 2048-token minimum per slot; a text-only one is honoured
exactly.** 8192 sails over the floor either way, which is why the context sweep never saw it. So the
projector cost 1154 MiB of encoder **plus** a forced 192 MiB of extra KV at 32 slots — and that is
what pushed the stock tag off the card into 26/36-layer partial offload.

## Concurrency, 512 in / 128 out — identical server geometry

| c | Ollama agg | llama.cpp agg | Ollama TTFT | llama.cpp TTFT | Ollama TPOT | llama.cpp TPOT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32.55 | 32.74 | 2114 | 2135 | 15.53 | 14.31 |
| 2 | 39.72 | 40.59 | 4164 | 4062 | 19.34 | 18.09 |
| 4 | 43.35 | 45.12 | 8393 | 7982 | 29.05 | 27.13 |
| 8 | 46.18 | 44.79 | 15978 | 16212 | 51.73 | 53.35 |
| 16 | 45.44 | 45.89 | 27686 | 32551 | 144.82 | 96.97 |
| 32 | 45.57 | 48.19 | 45385 | 61825 | 376.90 | 184.36 |

**Through c=8 the two runtimes are the same machine**: aggregate within 4%, TTFT within 3%, TPOT
within 9%. **TTFT doubles with every doubling of `c` on both** — 2114/4164/8393/15978 against
2135/4062/7982/16212, ratios 1.97, 2.02, 1.90 here and 1.90, 1.97, 2.03 there. Prefill runs one
request at a time on both, which is why aggregate saturates near **46 tok/s** here and **45–48**
there, and why concurrency past c=4–8 buys latency and nothing else.

**Past c=8 the two schedulers diverge, and it is a trade rather than a difference in throughput.**
Ollama's TTFT grows *sub*-linearly (ratios 1.73 and 1.64 against 2.01 and 1.90) while its TPOT
roughly doubles relative to the sibling (144.82 and 376.90 against 96.97 and 184.36). Aggregate is
unchanged — 45.44/45.57 against 45.89/48.19. First tokens arrive sooner, each stream then decodes
more slowly, and the total work per second is the same.

**That is not isolated.** One server flag still differs: Ollama forces `-b 512 -ub 512`, while the
sibling ran llama.cpp's defaults (`-b 2048 -ub 512`). Batch size governs how prefill chunks interleave
with decode, so it is the leading candidate and it is untested. Re-running the sibling with `-b 512`
would settle it and has not been done.

## Caveats

- **c=8 is the noisy cell on both sides** — 6.9% spread here, 7.7% there, against ≤1% everywhere
  else. This part is Max-Q and throttles; never read a single row.
- **Every cell measured the model thinking.** At 128 output tokens Gemma 4 has not closed its
  thinking block. The tok/s are real; no cell completed a task. Identically true of the sibling.
- **The two runs defeat the prompt cache by different means** — `cache_prompt: false` there,
  `--prompt-mode shuffled` here, because Ollama drops the flag in its OpenAI translation. Both
  verified to `cached = 0`.
- **Not transferable to the T4 rigs** at equal compute capability 7.5: TU117 has no tensor cores.
