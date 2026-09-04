# CLAUDE.md — local-ollama-1650ti-2b-q4_0

Guidance for working inside this rig. The siblings are not layers; nothing is
imported across a rig boundary, and a pattern that holds elsewhere in this tree
mostly does **not** hold here. Read this file before changing anything.

## What this rig is

The **Ollama daemon** serving `gemma4:e2b-it-qat` — Ollama's repackaging of
Google's QAT **Q4_0** GGUF — off one **GTX 1650 Ti (Max-Q)** in the machine
under the desk. Ollama v0.33.2, installed as a binary; there is no build here.

**STATUS 2026-09-04: served, and swept.** First light and a full sweep the same
day, in `benchmarks/runs/2026-09-04-first-light-ollama-1650ti/`.

It is the second `local` rig. Everything `@NAMING.md` says about `local` applies
unchanged and is not repeated here: no capacity to find, no endpoint to
discover, capacity is a hard ceiling rather than a quota conversation, nothing
billed and nothing to tear down. `local-llamacpp-1650ti-2b-q4_0/CLAUDE.md` spells
those four out.

## What it serves: the PROJECTOR-FREE variant, and that is worth 1150 MiB

`MODEL_NAME` is **`gemma4:e2b-it-qat-text`**, not the stock `gemma4:e2b-it-qat`.

The stock tag declares `vision` and `audio`, so the daemon passes `--mmproj`
unconditionally and a vision+audio encoder is resident for a pure-text workload.
`modelfiles/text-only.Modelfile` is `ollama show --modelfile` of that tag with the
second `FROM` removed and **nothing else changed** — same model blob by digest,
same `RENDERER gemma4` / `PARSER gemma4`, same three sampling parameters. Rebuild
with `make text-only`; it shares the blob, so it costs no extra disk.

| | resident @ 8192 ctx | capabilities |
| :--- | ---: | :--- |
| `gemma4:e2b-it-qat` | 2762 MiB | completion **vision audio** tools thinking |
| **`gemma4:e2b-it-qat-text`** | **1612 MiB** | completion tools thinking |
| llama.cpp sibling, same weights | 1618 MiB | — |

**−1150 MiB, and 6 MiB below the sibling.** Decode is unchanged (69.81 tok/s
server-side gauge against 68.95 stock). It buys the 32-slot configuration
outright — see the concurrency section.

`MODEL_STOCK_TAG` keeps the stock tag on disk because it is what
`gpu-ollama-g5g-2b-q4_0` serves, so it is the reference for any comparison
against that rig. **`model_info` and `verify_model_resident` read the loaded
model's declared capabilities**, not the tag name, so serving the stock tag by
accident is reported rather than silently paid for.

## Why it exists: the A/B, and what actually differs

**Two slots differ from `local-llamacpp-1650ti-2b-q4_0`: slot 2, and nothing
else.** Same card, same weights, same engine — Ollama links `libllama.so` and
the Gemma 4 graph both execute is upstream `src/models/gemma4.cpp`. The daemon
even starts a `llama-server` child; you can read its command line in
`run/ollama.log`.

So every difference below is **a choice the daemon makes and llama.cpp leaves to
you**, which is the entire content of the pair. All MEASURED 2026-09-04.

| | `local-llamacpp-…` | `local-ollama-…` |
| :--- | :--- | :--- |
| Artifact | Google's file, hashable | re-containered blob, template stripped |
| Projector (986 MB) | not loaded | loaded by the **stock tag**; dropped here |
| Resident VRAM @ 8192 ctx | 1618 MiB | **2762 MiB** |
| Chat template | the GGUF's own jinja | `--no-jinja`, Ollama's Go renderer |
| Offload | `-ngl 99`, yours to set | daemon's VRAM estimate, detect-only |
| Flash attention | `-fa 1`, worth +4.8% | `--flash-attn auto`, already on |
| Context × slots | `-c` **divided** across slots | context **multiplied** by slots |
| Build | compiles llama.cpp at a pinned commit | none |

## The artifact is what the registry manifest said — confirmed a second time

Pulled 2026-09-04 on x86_64. Both layers landed on exactly the byte counts and
digests `gpu-ollama-g5g-2b-q4_0` read off the registry manifest on 2026-09-02,
on a different architecture and a different host:

```
model      sha256:3646b4c1…   3,349,514,112 B     (Google's file: 3,349,516,256, -2,144)
projector  sha256:58c18764…     986,833,312 B     (Google's file:   986,833,664,   -352)
```

`ollama show` reports `architecture gemma4`, `parameters 4.6B`,
`quantization Q4_0`, capabilities `completion vision audio tools thinking`.

**Do not point this rig at `~/models/.../gemma-4-E2B_q4_0-it.gguf` through a
Modelfile.** It would make this a llama.cpp rig wearing an Ollama name and delete
the four differences the pair exists to measure. A test asserts `MODEL_NAME`
stays a tag.

## SETTLED 2026-09-04: the +1144 MiB is the projector, and Ollama's own gauge hides it

**This is the largest practical difference between the two rigs and it is
invisible from `ollama ps`.**

```
ollama ps    gemma4:e2b-it-qat   1.7 GB   100% GPU   8192   Forever
nvidia-smi   …/lib/ollama/llama-server                2762 MiB of 4096
```

The daemon's own `size`/`size_vram` says 1.7 GB. The driver says **2762 MiB**.
The difference is the multimodal projector, which the daemon passes
unconditionally because the tag declares `vision` and `audio`:

```
--mmproj …/blobs/sha256-58c187648007…
[mtmd] estimated worst-case memory usage of mmproj is 1154.07 MiB
clip_model_loader: has vision encoder / has audio encoder
clip_ctx: CLIP using CUDA0 backend        (twice — gemma4v and gemma4a)
```

**On a 4096 MiB card that is 28% of the budget spent on capability this rig never
exercises**, and it cannot be turned off from the API. `verify_model_resident`
reads both numbers for this reason; `make residency` does the same from the
shell. **Never size this rig from `ollama ps`.**

## The full VRAM accounting — and it corrects the sibling AND `@MODELS.md`

Read off `run/ollama.log` at `n_ctx=8192`, `--parallel 1`:

| term | MiB | source |
| :--- | ---: | :--- |
| model buffer (weights) | 1341.78 | `load_tensors: CUDA0 model buffer size` |
| KV, non-SWA | 48.00 | 8192 cells × **3** layers |
| KV, SWA | 12.00 | **1024** cells × **12** layers |
| compute buffer | 122.52 | `sched_reserve: CUDA0` |
| mmproj (worst case) | 1154.07 | `[mtmd] estimated` |
| CUDA context + slack | ~84 | residual |
| **total** | **~2762** | **matches nvidia-smi exactly** |

Two corrections fall out of the KV rows, and both are load-bearing.

**1. `@MODELS.md`'s 18 KiB/token is right as geometry and wrong as an allocation
on this engine.** Its derivation — 12 sliding layers × 1 KiB + 3 full layers ×
2 KiB — is confirmed to the byte here (6 MiB ÷ 12 ÷ 1024 = 512 B of K per cell
per sliding layer; 24 MiB ÷ 3 ÷ 8192 = 1024 B for a full one). But it assumes
**all 15 cached layers hold `n_ctx` tokens**, which is what the TPU path does and
what its two cross-checks confirm. `llama_kv_cache_iswa` **caps the sliding
layers at the window** — 1024 cells whatever `n_ctx` is. So on llama.cpp and
Ollama the cost is **6 KiB/token plus a flat 12 MiB**, not 18 KiB/token:

```
8192 ctx:   MODELS.md predicts 144 MiB      measured 60 MiB     over by 2.4x
```

The error grows with context and is a *floor* error, so it never causes an OOM —
it wastes a budget you thought you had spent.

**2. The sibling's 1618 MiB derivation is right in total and wrong in both
terms.** `local-llamacpp-1650ti-2b-q4_0/CLAUDE.md` reads
`1342 + 144 KV + ~130 CUDA context/compute = 1616, measured 1618`. The weights
term is exactly right (1341.78 measured here on the same engine). The other two
are not: KV is 60, and compute buffer plus CUDA context is ~216. **Two offsetting
errors landing 2 MiB from the measurement** is the most dangerous shape an
arithmetic can have, because it reads as confirmation. Corrected there.

## Measured performance: end-to-end is a tie, decode is ~3% behind

MEASURED 2026-09-04, `sweep.py`, 3 repeats per cell, prompt cache defeated,
concurrency 1. Full write-up in
`benchmarks/runs/2026-09-04-first-light-ollama-1650ti/REPORT.md`.

**These cells were measured on the STOCK tag, before the switch to the
projector-free variant**, and have not been repeated. The projector is not in the
text compute path — it is resident weight, not work — so the figures should
carry, and the one direct check agrees (69.81 tok/s server-side gauge text-only
against 68.95 stock). **That is one sample against a ten-cell sweep; treat the
carry-over as reasonable rather than established**, and re-run the context sweep
before quoting these as the variant's numbers.

**Decode, corrected for chunk batching (see the next section — the raw column is
not comparable), output 128:**

| input tok | llama.cpp | Ollama | Δ |
| ---: | ---: | ---: | ---: |
| ~100 | 72.55 | 70.37 | −3.0% |
| ~655 | 70.78 | 69.27 | −2.1% |
| ~1275 | 70.27 | 68.20 | −2.9% |
| ~2520 | 68.64 | 66.48 | −3.1% |
| ~5025 | 66.81 | 64.95 | −2.8% |

**End-to-end throughput is a dead heat — within 1% in all ten cells**, because
end-to-end at these shapes is dominated by prefill and prefill is the same
engine doing the same work (~318 t/s measured on both, at the same 661/652-token
prompt).

| in / out | llama.cpp e2e | Ollama e2e |
| :--- | ---: | ---: |
| ~100 / 32 | 40.87 | 41.53 |
| ~100 / 128 | 59.72 | 58.93 |
| ~655 / 32 | 12.53 | 12.60 |
| ~655 / 128 | 32.39 | 32.21 |
| ~1275 / 32 | 7.02 | 7.08 |
| ~1275 / 128 | 21.50 | 21.50 |
| ~2520 / 32 | 3.58 | 3.59 |
| ~2520 / 128 | 12.34 | 12.34 |
| ~5025 / 32 | 1.76 | 1.76 |
| ~5025 / 128 | 6.50 | 6.50 |

**Read the ~3% as an upper bound on the daemon's overhead, not as a property of
Ollama's decoder** — it is the same decoder. The plausible causes are the extra
HTTP hop through the Go frontend and the renderer, and nothing here isolates
them.

## Concurrency: the same machine through c=8, a scheduling trade past it

MEASURED 2026-09-04, 512 in / 128 out, 3 repeats, in
`benchmarks/runs/2026-09-04-text-only-ollama-1650ti/`. **Projector-free, this rig
reaches the sibling's exact geometry** — both are `llama-server -c 32768` with 32
slots of 1024 tokens, 36/36 layers on GPU, all KV in VRAM, 2106 MiB of 4096.

| c | Ollama agg | llama.cpp agg | Ollama TTFT | llama.cpp TTFT | Ollama TPOT | llama.cpp TPOT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32.55 | 32.74 | 2114 | 2135 | 15.53 | 14.31 |
| 2 | 39.72 | 40.59 | 4164 | 4062 | 19.34 | 18.09 |
| 4 | 43.35 | 45.12 | 8393 | 7982 | 29.05 | 27.13 |
| 8 | 46.18 | 44.79 | 15978 | 16212 | 51.73 | 53.35 |
| 16 | 45.44 | 45.89 | 27686 | 32551 | 144.82 | 96.97 |
| 32 | 45.57 | 48.19 | 45385 | 61825 | 376.90 | 184.36 |

**Through c=8 the two runtimes are the same machine**: aggregate within 4%, TTFT
within 3%, TPOT within 9%. **TTFT doubles with every doubling of `c` on both** —
ratios 1.97, 2.02, 1.90 here against 1.90, 1.97, 2.03 there. Prefill runs one
request at a time on both, aggregate saturates near **46 tok/s** here and
**45–48** there, and concurrency past c=4–8 buys latency and nothing else. That
is the load-bearing result and it is identical on both runtimes.

**Past c=8 the schedulers diverge, and it is a trade, not a throughput
difference.** Ollama's TTFT grows *sub*-linearly (1.73, 1.64 against 2.01, 1.90)
while its TPOT roughly doubles relative to the sibling (144.82, 376.90 against
96.97, 184.36). Aggregate is unchanged. First tokens arrive sooner, each stream
then decodes more slowly, total work per second is the same.

**One server flag still differs and is the leading candidate: Ollama forces
`-b 512 -ub 512`, the sibling ran llama.cpp's defaults (`-b 2048 -ub 512`).**
Batch size governs how prefill chunks interleave with decode. Re-running the
sibling at `-b 512` would settle it and has not been done — do not attribute the
divergence to the daemon until it has.

**c=8 is the noisy cell on both sides** — 6.9% spread here, 7.7% there, against
≤1% everywhere else. Max-Q thermals; never read a single row.

### With the STOCK tag this configuration silently fell off the card

The first attempt at that table ran on `gemma4:e2b-it-qat` and could not reach
the geometry above: the projector's 2048-token slot floor doubled the non-SWA KV,
and with the 1154 MiB encoder on top llama.cpp offloaded only 26/36 layers and
put 512 MiB of KV in host RAM. Nothing errored; per-stream decode halved (34.74
against 69.89) and aggregate peaked at 32.06. Kept as
`benchmarks/runs/2026-09-04-first-light-ollama-1650ti/concurrency.np32-partial-offload.json`
with `status: failed`, and written up under "Sizing the server for a sweep"
below. **Dropping the projector is the fix; no configuration knob is.**

## THE TRAP THAT ALMOST PUBLISHED AN 11% GAP: chunks are not tokens

**`sweep.py`'s streaming decode figure counts inter-CHUNK gaps, and neither
engine emits one chunk per token — but they batch DIFFERENTLY, so the artifact
does not cancel.**

MEASURED 2026-09-04, `completion_tokens ÷ stream_chunks`:

| output | llama.cpp | Ollama |
| ---: | ---: | ---: |
| 32 | 1.103 | **1.185** |
| 128 | 1.024 | **1.067 – 1.143** |

Taken as run, the two sweeps said 71.15 vs 61.52 tok/s at the shortest prompt —
an **11% Ollama deficit that is 8 points harness and 3 points hardware.** The
corrected figure divides `completion_tokens` by the measured inter-token span
rather than counting chunks, and lands at 72.55 vs 70.37.

Consequences:

- **Never difference two rigs' `decode_tps` without checking
  `chunks_match_usage`.** It is `false` in every cell of both sweeps. The field
  exists precisely so this is visible; it was added after the sibling noticed
  29 chunks against 32 tokens and it is doing its job.
- The `output=32` rows are the worst offenders (1.185) because the series is
  ~27 gaps long. **Quote the 128-token rows.**
- `end_to_end_tps` is immune — it is tokens over wall time and never touches the
  chunk count. It is the figure to trust when the two disagree.

## Both rigs measured the model thinking, and that part IS comparable

`content_chunks` is **0** and `reasoning_chunks` is the whole stream in every
cell of both sweeps — 2156 reasoning chunks here, 2310 in the sibling. At 32 and
128 output tokens Gemma 4 has not closed its thinking block, so no cell in either
rig completed a task. **The tok/s figures are real and the tasks are fictional**,
identically on both sides, which is what makes the comparison fair.

## Gemma 4 reasons, and Ollama has THREE different behaviours for it

The sibling documents one empty-reply path (`reasoning_content` populated,
`content` empty). Ollama has three, and they disagree with each other. MEASURED
2026-09-04, all on "Name three TPU generations":

| call | tokens | content | thinking |
| :--- | ---: | :--- | :--- |
| `/api/chat` `think:true` | 278 | 152 chars | 916 chars |
| `/api/chat` `think:false` | 490 | 2290 chars | — |
| `/api/generate` default | 278 | 152 chars | **generated and DISCARDED** |
| `/api/generate` default, `num_predict:128` | 128 | **empty** | **empty** |
| `/v1/chat/completions` streaming | — | field is **`reasoning`** | not `reasoning_content` |

**The fourth row is the trap**: 128 tokens generated, `done_reason: length`, and
**zero characters returned on either field**. The llama.cpp sibling at least
hands back the partial thought. Consequences, all applied:

- `query_model` sends `/api/chat` with `think=True` and a **1024** default
  `max_tokens`; a test asserts it stays ≥512.
- When `content` is empty it returns 📡 and says whether the thinking was
  discarded or merely truncated — never ✅ with an empty body.
- **Budget tokens accordingly in any benchmark.** A 128-token limit measures the
  thinking phase and nothing else.

## `sweep.py` is a copy of the sibling's and needed three more fixes

Copied, not imported — rigs are siblings. All three silently measure the wrong
thing rather than failing, except the first, which failed loudly and cost one
discarded run (`sweep.DISCARDED-zero-decode.json`, kept beside the good one).

1. **Ollama spells the field `reasoning`, not `reasoning_content`.** Every chunk
   is `{"content": "", "reasoning": "…"}` until the block closes, so the
   sibling's field name stamped **zero tokens in all ten cells** and the run
   reported 0.00 tok/s throughout. Note the empty *string* in `content`: falsy,
   so an `or` still works, but a check written as `"content" in d` would have
   counted every chunk and measured nothing.
2. **`cache_prompt: false` does not reach the engine.** Ollama translates the
   OpenAI body into its own schema and unknown keys do not survive; llama-server
   passes them through. An 8-token request carrying the flag still logged
   `cached n_tokens = 660` of 661. `DEFEAT_PROMPT_CACHE` is therefore `False`
   here and the cache is defeated by the prompt instead — `--prompt-mode
   shuffled`, which reorders the filler so there is no reusable prefix at all,
   while keeping the same vocabulary so cells land on the same token counts as
   the sibling's run.
3. **A fixed RNG seed re-created the previous process's prompts.** Ollama's
   prompt cache outlives the client — it holds several prompts for the life of
   the loaded model — so a seeded shuffle made the *second* sweep hit the
   *first* one's cache: a 652-token prompt returned in 0.15 s against 2.17 s
   cold, i.e. prefill "measured" at 16,000 t/s. `_RNG = random.Random()`, and a
   test pins it.

**Fixes 1 and 3 are not present in any sibling copy of `sweep.py`.** Fix 2 is
specific to this rig's engine.

### Sizing the server for a sweep: the convention is INVERTED here

**llama.cpp DIVIDES `--ctx-size` across `--parallel` slots. Ollama MULTIPLIES
`OLLAMA_CONTEXT_LENGTH` by `OLLAMA_NUM_PARALLEL`.** Same total, opposite
convention, and nothing announces it.

| sweep | llama.cpp sibling | here |
| :--- | :--- | :--- |
| context, c=1 | `-c 8192 --parallel 1` | `CONTEXT_LENGTH=8192 NUM_PARALLEL=1` |
| concurrency, c≤32 | `-c 32768 --parallel 32` | `CONTEXT_LENGTH=1024 NUM_PARALLEL=32` |

Set `CONTEXT_LENGTH=32768 NUM_PARALLEL=32` here and you have asked for a **1 M
token** allocation on a 4 GiB card.

**AND THERE IS A FLOOR — but it applies to MULTIMODAL MODELS ONLY, which is why
this rig serves a text-only variant.** Read off the child's command line in
`run/ollama.log`, four settings, both models:

| model | asked | child got | per slot |
| :--- | :--- | :--- | ---: |
| stock `gemma4:e2b-it-qat` | 1024 × 1 | `-c 2048 -np 1` | **2048** |
| stock | 1536 × 1 | `-c 2048 -np 1` | **2048** |
| stock | 8192 × 1 | `-c 8192 -np 1` | 8192 |
| `gemma4:e2b-it-qat-text` | 1024 × 1 | `-c 1024 -np 1` | 1024 |
| text-only | 1536 × 1 | `-c 1536 -np 1` | 1536 |
| text-only | 1024 × 32 | `-c 32768 -np 32` | 1024 |

A projector-carrying model gets a **2048-token minimum per slot**; a text-only
one is honoured exactly. 8192 clears the floor either way, which is why the
context sweep never saw it.

**That floor is what put the stock tag off the card.** At 32 slots it doubled the
non-SWA KV (192 → 384 MiB), and with the 1154 MiB encoder on top llama.cpp
offloaded only 26/36 layers and put 512 MiB of KV in host RAM:

```
load_tensors: offloaded 26/36 layers to GPU
llama_kv_cache: CUDA0 KV 128.00 MiB   CPU KV 256.00 MiB   (twice, one per cache)
ollama ps: 4.2 GB   60%/40% CPU/GPU
```

**Nothing errored, and per-stream decode halved** — 34.74 tok/s against the
sibling's 69.89. Kept as
`benchmarks/runs/2026-09-04-first-light-ollama-1650ti/concurrency.np32-partial-offload.json`
with `status: failed`. **No knob prevents it**; the daemon owns the offload
decision and `OLLAMA_GPU_OVERHEAD` is only a nudge. Dropping the projector does.

## Two more daemon knobs that are not optional

- **`OLLAMA_LLM_LIBRARY=cuda_v12`.** The bundle ships `cuda_v12` and `cuda_v13`
  and Ollama picks between them by driver version; 610.57.04 reports as CUDA 13.3.
  On the arm64 bundle `gpu-ollama-g5g-2b-q4_0` measured `cuda_v13` to be
  **PTX-only** — every kernel JIT-compiled at load — while `cuda_v12` carries
  native sm_75 SASS. The pin makes the pair comparable, since the llama.cpp
  sibling builds native sm_75. Confirmed from the log here:
  `libdirs=ollama,cuda_v12`. **The fatbin walk itself has not been repeated on
  the x86_64 bundle**; the pin rests on the sibling's arm64 measurement.
- **`OLLAMA_KEEP_ALIVE=-1`.** Ollama unloads after 5 minutes idle by default, so
  a sweep that pauses between cells pays a full reload and records it as
  latency. No sibling needs this — llama-server, vLLM and this repo's own JAX and
  PyTorch servers hold the weights for the life of the process.

## Hardware: sm_75 without tensor cores

Compute capability **7.5**, and do **not** read that as "same as the T4 rigs."
The T4 in `gpu-vllm-g4dn-2b` / `gpu-vllm-g5g-2b` is TU104 and has tensor cores;
the GTX 16-series is TU116/TU117 and has none. Same compute capability, different
silicon — which is why slot 3 is `1650ti` and not `turing75` (`@NAMING.md`
records the decision). A throughput number from this card is not comparable to a
T4 number at equal compute capability.

**This is a Max-Q part and it throttles.** The sibling saw a B=32 concurrency
cell come in *below* B=16 on one pass and 40% higher on an identical re-run.
Never read a single row as a result — repeat it, which is what `--repeats 3` is
for.

## Conventions

- Tests are `unittest`, never pytest: `python3 -m unittest discover -s tests -v`.
- Every subprocess call goes through `run_command(cmd: list[str])` using
  `asyncio.create_subprocess_exec`. **Never `shell=True`.**
- MCP tools are `async def` returning markdown strings with emoji status
  prefixes (`✅`, `❌`, `📡`).
- `Optional[str]`, not `X | None`.
- Use the system `python3` and install into it. **Never create a virtualenv.**
- `tpu.env` is the source of truth and is committed. Never add `*.env` to
  `.gitignore`.
- No `.claude-plugin/`, no `.codex/`, no `skills/`, and none is owed — this rig
  matches its llama.cpp sibling, which has none either and is absent from the
  root `marketplace.json` on purpose.

## Canonical root references

Read these before deriving their numbers here, and correct them **there** rather
than restating them in this rig: `@MODELS.md` (checkpoint properties, KV cost,
weight footprints), `@HARDWARE.md` (accelerator properties and native numeric
formats), `@QUANTIZATION.md` (what the serving stack actually supports),
`@NAMING.md` (how any of it is spelled), `@RIG-ANALYSIS.md` (the order to consult
them in).
