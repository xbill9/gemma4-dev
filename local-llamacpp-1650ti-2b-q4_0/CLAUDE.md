# CLAUDE.md — local-llamacpp-1650ti-2b-q4_0

Guidance for working inside this rig. The siblings are not layers; nothing is
imported across a rig boundary, and a pattern that holds elsewhere in this tree
mostly does **not** hold here. Read this file before changing anything.

## What this rig is

`llama-server` from `ggml-org/llama.cpp`, driven directly, serving
`google/gemma-4-E2B-it-qat-q4_0-gguf` off one **GTX 1650 Ti (Max-Q)** in the
machine under the desk. One process, one GGUF file named on the command line.

**STATUS 2026-09-03: nothing has been served.** llama.cpp is built at `95ef7fc`
and the model file is on disk. No token has been generated, no benchmark has
been run, and `benchmarks/runs/` is empty on purpose.

## It is the first `local` rig, and that is most of what is different

`local` was added to slot 1 of `@NAMING.md` on 2026-09-03 for this rig. It names
the case where there is no control plane at all — the accelerator is in the
machine and is there whether or not any code runs. Four consequences, and all
four are places where sibling code would be actively wrong here:

- **There is no capacity to find.** No zone scan, no queued resource, no
  `~/.cache/<rig>/tpu_zones_status.md` skip list, no flex-start wait. If this rig
  ever grows a `find_*` tool it has the wrong name.
- **The endpoint is not discovered.** `127.0.0.1:8080`, known before the process
  starts. The root `CLAUDE.md` rule "never hardcode an endpoint" is about the
  QR → node → external IP chain; there is no such chain here, so `ENDPOINT` in
  `tpu.env` is a literal on purpose.
- **Capacity is a hard ceiling, not a quota conversation.** 4096 MiB is the
  budget. There is no bigger machine type to ask for.
- **Nothing is billed and nothing needs tearing down.** The care the cloud rigs
  take not to strand capacity is dead weight here. Ctrl-C is a complete teardown.

## The memory arithmetic, and why the obvious version of it is wrong

The file is 3.35 GB on disk against 3.63 GiB free. Read those two numbers
side by side and you conclude that full offload barely fits and that you should
lower `-ngl` and cap `-c`. **That conclusion is wrong**, and it was reached and
discarded here on 2026-09-03 before anything was run.

MEASURED-STATIC (`gguf-py` over the artifact, 541 tensors, 3.334 GB of tensor
bytes):

| tensor | shape | type | bytes | resident? |
| --- | --- | --- | ---: | --- |
| `per_layer_token_embd.weight` | `[8960, 262144]` | Q6_K | 1926.8 MB | **no — lazy** |
| `token_embd.weight` | `[1536, 262144]` | Q6_K | 330.3 MB | yes |
| 35 transformer blocks | — | Q4_0 | ~1080 MB | yes |
| **must be resident** | | | **1.407 GB = 1.31 GiB** | |

`per_layer_token_embd` is **58% of the file and none of it needs to be on the
GPU.** `src/models/gemma4.cpp:53` creates it with `TENSOR_READ_LAZY`, documented
in `src/llama-model-loader.h:72` as "read rows on demand instead of loading whole
tensor; requires mmap for now". It is a `GGML_OP_GET_ROWS` lookup rather than a
matmul (`src/llama-arch.cpp:902`), so rows are pulled from the mapped file on the
host as tokens need them.

That tensor is the `E` in E2B made concrete: `@MODELS.md` gives
`vocab_size_per_layer_input=262144`, `hidden_size_per_layer_input=256` and 35
layers, and 256 × 35 = 8960 is exactly the leading dimension above. The ~5B total
against 2B effective is mostly this one tensor.

**Consequences for anyone tuning this rig:**

- **Do not lower `N_GPU_LAYERS` "to be safe."** ~2.3 GiB of headroom exists at
  full offload. Lowering it moves real matmul weights to the CPU and buys nothing.
- **`--no-mmap` breaks the arithmetic outright.** `TENSOR_READ_LAZY` "requires
  mmap for now". Disable mmap and the 1.93 GB tensor has to be materialised.
- **Do not size this rig from the file size on disk**, and do not size it from
  `@MODELS.md`'s int4 column either — that column is a floor and under-predicts
  by ~19%, for the same reason visible here.

## SETTLED 2026-09-03: `per_layer_token_embd` is NOT in VRAM

**MEASURED on this rig, first light.** `llama-server -ngl 99 -c 8192 -ctk f16 -ctv f16`:

```
nvidia-smi --query-compute-apps=pid,process_name,used_memory
37073, .../llama-server, 1618 MiB          # of 4096 MiB total
```

**1618 MiB.** The derivation predicted it almost exactly:

```
1342 MiB  resident weights (3.334 GB file - 1.927 GB lazy PLE)
+ 144 MiB  KV, 8192 tokens x 18 KiB/token f16   (MODELS.md)
+ ~130 MiB  CUDA context + compute buffers
= 1616 MiB                                        measured 1618
```

> **CORRECTED 2026-09-04: the total is right and BOTH of the last two terms are wrong.**
> `local-ollama-1650ti-2b-q4_0` runs the same engine on the same card and its daemon prints the
> allocation, so the split can be read rather than derived:
>
> ```
> 1341.78 MiB  CUDA0 model buffer          <- the weights term is exact
>    48.00 MiB  KV non-SWA, 8192 cells x 3 layers
>    12.00 MiB  KV SWA,     1024 cells x 12 layers   <- capped at the WINDOW, not n_ctx
>   122.52 MiB  CUDA0 compute buffer
>    ~94 MiB    CUDA context + slack
> = 1618 MiB
> ```
>
> KV is **60 MiB, not 144** — `llama_kv_cache_iswa` sizes the twelve sliding layers at the 1024-cell
> window whatever `n_ctx` is, so `@MODELS.md`'s 18 KiB/token (correct geometry, and correct on the TPU
> path) over-predicts by 2.4x here. Compute buffer plus CUDA context is **~216 MiB, not ~130**.
>
> **Two offsetting errors landing 2 MiB from the measurement is the most dangerous shape an arithmetic
> can have**, because agreeing with the hardware reads as confirmation of every term. The lesson is the
> one `@MODELS.md` now carries: read the engine's allocation log, do not derive KV and then check only
> the total. The qualifier is filed in `@MODELS.md` because it describes the checkpoint's KV geometry
> meeting an engine's policy, not this card.

If `per_layer_token_embd` were resident the figure would be ~3.5 GB, which **does not fit in
4096 MiB at all** — the model would have failed to load rather than served at 73 tok/s. The
`TENSOR_READ_LAZY` reading is correct.

**This corrects `gpu-llamacpp-g5g-2b-q4_0`**, whose `CLAUDE.md` table gives *Resident: 3.35 GB* for
this artifact and computes "freeing ~6.9 GB of a 14.07 GB budget" from it. The real figure is
~1.4 GB resident and ~8.8 GB freed. That rig has served nothing, so the error is derivational, not
a bad measurement — but it is load-bearing there too, because it understates how much of a T4G is
left for batching, which is the experiment that rig exists to run.

The finding describes **llama.cpp's treatment of E2B's per-layer embeddings**, not this card, so it
is filed in `@MODELS.md`. Do not re-derive it from a rig.

## The file is only 32% Q4_0

Slot 5 of the directory name is `q4_0` because that is what `MODEL_NAME` says,
per `@NAMING.md` ("named exactly as the file does"). **The dominant tensor type
in the artifact is Q6_K, not Q4_0**: both embedding tensors are Q6_K and together
they are 2.257 GB of the 3.334 GB, leaving ~1.08 GB of actually-Q4_0 transformer
body.

This is the same fact `@MODELS.md` records as the reason its int4 column
under-predicts — "`embed_tokens` stays at the storage dtype rather than being
quantized, and the per-group scales are extra" — confirmed in this artifact
rather than inferred. It matters twice:

- **Further weight quantization has little left to take, and would take it from
  the worst place.** Anything below q4_0 comes out of the Q6_K embeddings first.
  There is no headroom problem to solve, so there is no reason to pay for it.
- **A benchmark from this rig is not a clean "q4_0" datapoint.** Record the
  per-tensor split in the report, not just the slot-5 token.

## Measured performance, and what moved it

MEASURED 2026-09-03, first light. Full write-up in
`benchmarks/runs/2026-09-03-first-light-1650ti/REPORT.md`.

**Single stream: 73.75 tok/s decode, ~340 t/s prefill.** Best config is
`-fa 1 -ctk f16 -ctv f16 -t 4`, which is what `tpu.env` and `make serve` now carry.

| lever | result | adopted? |
| :--- | :--- | :--- |
| `-fa 1` flash attention | +4.8% decode (70.39 → 73.74) | **yes** |
| thread count 4/6/8 | no effect (73.75 / 73.08 / 72.97) | t=4, marginally |
| `-ctk`/`-ctv q8_0` | **−12% decode, −20-40% prefill** | **no** |
| `GGML_CUDA_FORCE_MMQ=ON` | −0.8% prefill, −1% decode | **no** |
| `--parallel` 1→64 | **+3.8x aggregate decode** | available, default 1 |

**Two of those are traps that look like optimizations**, and both would be adopted by anyone
reasoning from a sibling rig rather than measuring here:

- **KV quantization is a loss.** It is a win on the TPU rigs and it is a clear regression here.
  There are no tensor cores to hide the dequant, and E2B's KV is only ~144 MiB at 8192 context,
  so `q8_0` trades throughput for memory that was never scarce.
- **`GGML_CUDA_FORCE_MMQ` is llama.cpp's own suggestion for this card** — it prints the advice at
  init, unprompted, because the device lacks tensor cores. It does not pay. Decode at B=1 is
  bandwidth- and launch-bound rather than matmul-bound, which is also why forcing an integer
  matmul path moved prefill by 0.8%.

**Concurrency is the only large lever**: 73.33 → 223.75 tok/s at B=8, → 277.61 at B=64. Per-stream
falls correspondingly (73.33 → 27.97 → 4.34), so the right point depends entirely on whether this
is one interactive session or many. Prefill is flat at ~355 t/s from B=1 to B=64 — already
saturated. Memory does not bind: `-c 16384 --parallel 16` is 1818 MiB of 4096.

**This is a Max-Q part and it throttles.** A first concurrency pass measured B=32 at 184.09 tok/s,
*below* B=16; an identical re-run gave 258.28. Never read a single `llama-batched-bench` row as a
result — repeat it.

## Hardware: sm_75 without tensor cores

Compute capability **7.5**, MEASURED-STATIC from the driver. Do **not** read that
as "same as the T4 rigs."

The T4 in `gpu-vllm-g4dn-2b` and `gpu-vllm-g5g-2b` is TU104 and has tensor
cores. The GTX 16-series is TU116/TU117 and **has none** — Nvidia cut both the RT
and the tensor cores from that die. Same compute capability, different silicon.
This is why the hardware slot is `1650ti` and not `turing75`; `@NAMING.md`
records the decision so it is not re-derived.

Practical reading: a throughput number from this card is not comparable to a T4
number at equal compute capability, and any conclusion that depends on tensor-core
math does not transfer in either direction.

**Untested here:** whether llama.cpp's MMQ kernels take the DP4A path on TU117
and what that costs relative to a T4. Measure it; do not assume it from sm_75.

## Gemma 4 reasons, and that is the second way to get an empty reply

**MEASURED 2026-09-03.** The root `CLAUDE.md` warns that raw `/v1/completions` returns an empty
completion on `-it` models. There is a **second, unrelated** empty-reply path here, and it hits
`/v1/chat/completions` — the endpoint you were told to use instead.

Gemma 4 emits a thinking block. llama.cpp routes it to `reasoning_content` and leaves `content`
empty until the block closes:

```
max_tokens=64   -> finish_reason "length", content "",  reasoning_content 274 chars (truncated)
max_tokens=400  -> finish_reason "stop",   content "TPU v1, TPU v2, TPU v5",  reasoning 1274 chars
```

**1274 characters of reasoning to answer "name three TPU generations."** A caller that reads only
`content` with a modest `max_tokens` gets `""` and concludes the server is broken.

Consequences, all of them already applied:

- `query_model`'s default `max_tokens` is **1024**, not 256, and a test asserts it stays ≥512.
- When `content` is empty but `reasoning_content` is not, `query_model` returns 📡 and says so
  explicitly rather than ✅ with an empty body.
- **Budget tokens accordingly in any benchmark.** A 128-token generation limit on this model
  measures the thinking phase and nothing else, which makes the tok/s real but the task fictional.

## Prefill does not batch, and that caps end-to-end serving throughput at ~45 tok/s

**MEASURED 2026-09-03**, `sweep.py` concurrency mode, 512 in / 128 out, 3 repeats per level,
prompt cache defeated. Run `benchmarks/runs/2026-09-03-full-sweep-1650ti/`.

| c | aggregate tok/s | spread | TTFT ms | TPOT ms | per stream |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32.74 | 0.2% | 2135 | 14.31 | 69.89 |
| 2 | 40.59 | 0.3% | 4062 | 18.09 | 55.27 |
| 4 | 45.12 | 0.2% | 7982 | 27.13 | 36.86 |
| 8 | 44.79 | 7.7% | 16212 | 53.35 | 18.74 |
| 16 | 45.89 | 0.5% | 32551 | 96.97 | 10.31 |
| 32 | 48.19 | 0.1% | 61825 | 184.36 | 5.42 |

**TTFT doubles exactly with every doubling of `c`** — 2135, 4062, 7982, 16212, 32551, 61825. That
is prefill running strictly one request at a time: ~252 t/s of prefill however many clients are
waiting. Aggregate output therefore saturates near **45-48 tok/s and 1.47x from c=1**, and adding
clients past c=4 buys nothing but latency.

**This does not contradict the 3.8x from `llama-batched-bench`; the two measure different phases.**
`llama-batched-bench` reports `S_TG`, the decode phase in isolation, and decode genuinely does batch
(73 -> 278 tok/s). But at a 512-in/128-out shape prefill is 4x the token volume of decode, so the
phase that scales is the minority of the work. **Quote the decode figure only with the phase named**;
a bare "277 tok/s" for this rig is wrong for any real serving workload.

Two consequences for anyone planning work here:

- **A long-prompt workload will not benefit from concurrency on this card.** Short-prompt,
  long-generation is the shape that would — untested, and the obvious next sweep.
- **The `-npp 128` in the early `llama-batched-bench` run flattered it.** 128-token prompts make
  prefill only 1x the decode volume; 512-token prompts make it 4x, and the conclusion inverts.

## `sweep.py` came from a sibling and needed two fixes to measure this model at all

`sweep.py` is a copy of `gpu-pytorch-g5g-2b/sweep.py` (the newest of the four copies in this tree —
it is the only one with the concurrency dimension added 2026-09-01). Copied, not imported: rigs are
siblings. Two bugs had to be fixed before it produced a single number here, and **both are still
present in all three sibling copies**:

1. **It counted only `delta.content` off the SSE stream.** Gemma 4's tokens arrive as
   `delta.reasoning_content` until the thinking block closes, so the harness stamped **zero tokens
   for every cell**. Fixed by stamping either field — decode rate is a property of the token loop
   and does not care which field the text lands in — and reporting `content_chunks` and
   `reasoning_chunks` separately so a cell that thought until it ran out of budget is visible.
2. **The failure path itself raised.** `runs[0].get("error")` is `None` for a request that returned
   200 with no countable tokens, and `None[:200]` is a `TypeError` that took the whole run down
   instead of recording one failed cell. Fixed to coalesce first.

The mercy in (1) is that it failed loudly rather than reporting a plausible wrong number. Had the
model emitted a little content and a lot of reasoning, it would have silently measured the wrong
thing — which is the failure mode `benchmarks/README.md` exists to prevent.

> **TWO MORE FIXES LANDED IN `local-ollama-1650ti-2b-q4_0`'s COPY ON 2026-09-04 AND ARE NOT IN THIS
> ONE.** Neither changes a number measured here, but both are latent: (1) Ollama spells the reasoning
> delta `reasoning`, not `reasoning_content`, so the field list needs both — the same bug class as (1)
> above, one field name over; (2) the shuffled-prompt RNG must be seeded from entropy, because a fixed
> seed makes a second process regenerate the first one's prompts and hit a cache that outlives the
> client. Port them if this copy is ever pointed at another engine.

> **AND THE DECODE COLUMN IN THIS RIG'S OWN TABLES IS UNDERSTATED.** `decode_tps` counts inter-CHUNK
> gaps, and `chunks_match_usage` is `false` in every cell here: `completion_tokens / stream_chunks` is
> **1.103** at output 32 and **1.024** at output 128. Recomputed from this rig's own `sweep.json` as
> tokens over the measured inter-token span, the context-sweep decode column reads 78.78 / 76.94 /
> 75.85 / 74.43 / 72.30 at output 32 and 72.55 / 70.78 / 70.27 / 68.64 / 66.81 at output 128, against
> the 71.15 / 69.50 / 68.51 / 67.23 / 65.30 printed above. **`end_to_end_tps` is unaffected** — it is
> tokens over wall time and never touches the chunk count. Do not difference a `decode_tps` against
> another rig's without checking both ratios; the Ollama sibling batches at 1.185, so as-run the two
> rigs appeared 11% apart when the real gap is ~3%.

**`--decode-source auto` correctly resolves to `stream` here**, as the llamacpp sibling documents:
`llama-server` does not emit `usage.decode_tokens_per_second`, which is an invention of our own JAX
and PyTorch servers. Do not "fix" that by scraping `/metrics` — it would reintroduce a per-rig
statistic under a shared name.

### Sizing the server for a sweep is not optional

llama.cpp **splits `--ctx-size` across `--parallel` slots**. `-c 32768 --parallel 32` is 1024 tokens
per sequence, so a 2048-token context cell does not fit and the context sweep and the concurrency
sweep **need different servers**:

| sweep | server |
| :--- | :--- |
| context (concurrency 1) | `--parallel 1`, `-c` >= max context + output |
| concurrency (fixed shape) | `--parallel` >= max concurrency, `-c` >= max_c x (input + output) |

Get this wrong and the cells do not error — they queue, and the harness faithfully reports the
queue as the hardware's throughput.

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

## Canonical root references

Read these before deriving their numbers here, and correct them **there** rather
than restating them in this rig: `@MODELS.md` (checkpoint properties, KV cost,
weight footprints), `@HARDWARE.md` (accelerator properties and native numeric
formats), `@QUANTIZATION.md` (what the serving stack actually supports),
`@NAMING.md` (how any of it is spelled), `@RIG-ANALYSIS.md` (the order to consult
them in).
