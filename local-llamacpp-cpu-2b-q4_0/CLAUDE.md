# CLAUDE.md — local-llamacpp-cpu-2b-q4_0

Guidance for working inside this rig. The siblings are not layers; nothing is
imported across a rig boundary. Read this file before changing anything.

## What this rig is

`llama-server` from `ggml-org/llama.cpp`, driven directly, serving
`google/gemma-4-E2B-it-qat-q4_0-gguf` on the **CPU only** of this workstation. No
GPU, no control plane, no cloud. One process, one GGUF file named on the command
line.

**STATUS 2026-09-16: serving. One smoke test, nothing benchmarked.** llama.cpp is
built CPU-only at `~/llama.cpp/build-cpu` (commit `82324fc50`, recorded as
`LLAMA_CPP_COMMIT`), the GGUF is at `MODEL_PATH`, and `llama-server` came up on the
`tpu.env` defaults and answered correctly. One datapoint: prompt 25 tok, completion
328 tok, **24.8 tok/s end-to-end** at the guessed `-t 4`/`-tb 8`. That is a smoke
test, not a measurement. The thread and affinity levers **have** since been swept
(`benchmarks/runs/2026-09-16-thread-sweep-cpu`, 18 cells, `llama-bench`), which
kept both `tpu.env` thread values and found CPU affinity to be the real lever at
1.61x on prefill. No serving benchmark exists yet: `benchmarks/reports/` is empty
and `sweep.py` has never run here.

## This rig is one arm of a control

`local-llamacpp-1650ti-2b-q4_0` is the other arm. Same GGUF, same llama.cpp
commit, same port, same harness, same prompts — run **alternately**, so that the
only thing differing between their numbers is the device. Sharing port 8080 is
deliberate and load-bearing: the endpoint, the harness and the prompt set stay
fixed while the device changes underneath them.

**The arm is measured, never asserted.** Nothing in an HTTP response says which
binary produced it, and `sweep.py` used to take the rig name from its own `--rig`
argument. So restarting into the other arm and forgetting was enough to label a
whole run with the wrong device, with nothing anywhere disagreeing. `attest.py`
reads it off the live process instead:

| source | answers |
| --- | --- |
| `/proc/<pid>/exe` | which binary is executing — not what `LLAMA_SERVER_BIN` says |
| `/proc/<pid>/maps` | which ggml backends it **has loaded** |
| `/proc/<pid>/cmdline` | the real `-ngl`, whatever any env file claims |
| `/proc/<pid>/environ` | `CUDA_VISIBLE_DEVICES` as the child actually got it |

`maps` rather than `ldd` on purpose: llama.cpp **dlopen's** its backends, so a
CUDA backend can be absent from `ldd` output and present in the running process.
It is also evidence about the process that ran rather than about a file on disk
that may since have been rebuilt.

The verdict needs two signals to **agree** — a GPU backend mapped in, *and*
layers assigned to it. Disagreement is reported as `mixed`, never rounded to
either: a CUDA build with `-ngl 0` computes on the CPU but is not a clean CPU
arm, because the device is initialised and llama.cpp can still move large prefill
batches onto it. That is why this rig hides the device from the child rather than
trusting `-ngl 0` alone.

- `model_server_status` leads with ❌ when a healthy server is the **other** arm.
  Until 2026-09-16 it answered ✅ and merely annotated "not started through this
  server" — in a control that is the entire failure mode.
- `query_model` refuses before spending up to 900 s producing a mislabelled number.
- `sweep.py` attests before it measures, aborts on a mismatch (`--expect-device`,
  defaulting to this rig's own arm), and **stamps the attestation into every
  report** as `attestation` / `device`.
- `attest_arm` reports the whole picture, including `exe_sha256` — which, not a
  commit string, is what makes two arms pairable. A commit is what you meant to
  build; the hash is what ran.

### What must match between the arms, and what did not

| | status |
| --- | --- |
| llama.cpp commit | **fixed 2026-09-16** — was 82324fc50 here vs 95ef7fc there; both now `c6824a9` |
| `-tb` | **fixed 2026-09-16** — this arm passed it, the GPU arm had no `THREADS_BATCH` key at all |
| `sweep.py` FILLER | **fixed 2026-09-16** — each arm described its own hardware, so one `--contexts 512` built two differently-tokenizing prompts. Now device-neutral and byte-identical; `sweep.py` is byte-identical in both arms |
| `-c`, `-ctk`/`-ctv`, `-fa`, `--parallel`, `--metrics`, the GGUF, the port | already matched |

### Running the pair

Order effects are real on this host and are **not** yet controlled for: an
i7-1360P and a Max-Q card share one thermal envelope, so the CPU arm saturating
12 cores leaves the package hot and downclocked for whatever runs next. Page
cache is the same kind of problem from the other side — this arm reads the Q4_0
body every token across a 3.35 GB mmap, while the GPU arm loads to VRAM once.
Neither is measured. Until they are: fix an order, cool down between arms, keep
the cache state the same rule for both, and **record what you did in the report**
rather than assuming it did not matter. A-B-B-A detects the drift if you can
afford four runs.

Do not start an arm while anything else is loading the machine. A model load was
lost to an OOM on 2026-09-16 because a 12-way CUDA build was running beside it.

## Read this first: what this directory was until 2026-09-15

**A byte-identical copy of `local-llamacpp-1650ti-2b-q4_0`** (`diff -rq` empty):
a GTX 1650 Ti rig with `-ngl 99`, a `gpu_status` tool that shelled out to
`nvidia-smi`, a test asserting the 1650ti rig name, and three runs of GPU
measurements plus the published 1650 Ti articles under `docs/`.

**If `benchmarks/runs/*-1650ti`, `benchmarks/reports/*-1650ti.json` or `docs/` are
still present, they are those inherited copies, not results from this rig — delete
them.** `benchmarks/rollup.py` globs `*/benchmarks/` and would otherwise count
every GPU report twice, the second time under a CPU rig's name. The originals live
in the 1650ti sibling.

## CPU only is enforced, not configured

A CPU number from a process that could have offloaded to a GPU is not a CPU
number, so none of the three guards is a `tpu.env` value:

- **`-ngl 0` is hardcoded** in `server.py` and `make serve`. There is no
  `N_GPU_LAYERS` key; a test fails if one is added.
- **The child gets `CUDA_VISIBLE_DEVICES=`** (empty). With `-ngl 0` alone a CUDA
  build still initialises the device and llama.cpp can offload large prompt
  batches to it, so hiding the device is what actually keeps prefill on the CPU.
- **`start_model_server` refuses a GPU build.** It looks for `ggml-cuda`,
  `ggml-vulkan` etc. both next to the binary (dynamically loaded backends that
  `ldd` cannot see) and in `ldd` output (static links).

`make build` makes a `GGML_CUDA=OFF` build in its own `build-cpu/` directory so it
cannot be confused with a GPU build in the same checkout.

## Host — MEASURED-STATIC 2026-09-15

| | |
| :--- | :--- |
| CPU | 13th Gen Intel Core i7-1360P, 12 cores / 16 threads |
| Topology | hybrid: 4 P-cores with SMT (logical 0-7), 8 E-cores (logical 8-15) |
| SIMD | `avx avx2 avx_vnni` — **no AVX-512** |
| RAM | 15 GiB total, ~8 GiB available when checked |
| GPU | **GTX 1650 Ti, 4096 MiB, driver 615.71.09** — present, and the other arm |

The `cpu_status` tool re-reads all of this from `/proc` and `/sys`.

**The GPU row was wrong until 2026-09-16** — it read "none (`/dev/nvidia*`
absent)". `/dev/nvidia0` is there and `nvidia-smi` reports the card. That error
pointed the wrong way: it made the CPU-only guards look like belt-and-braces on a
machine with no GPU to hit, when they are in fact the only thing keeping this arm
honest. They do work — `start_model_server` refuses the GPU build, verified.

`local-pytorch-cpu-2b/CLAUDE.md` says this machine has 6 physical cores and six SMT
siblings. `lscpu` and `/sys/devices/cpu_{core,atom}` disagree (12 cores, only the
4 P-cores have SMT). Trust the kernel, and do not size threads from that sentence.

## Nothing tuned on the GPU transfers

Every lever the 1650ti sibling measured was measured against a CUDA device. Two
have since been measured here and are marked inline below; **`KV q8_0` and `-fa 1`
remain UNMEASURED on this host.** The sweep also found a lever the sibling never
had — CPU affinity — which matters more than either.

- **Threads are NOT the main lever — MEASURED 2026-09-16, this file predicted the
  opposite.** Decode spans only 22.97-25.58 t/s across `-t` 4/8/12/16,
  non-monotonic, best cell 0.9% over the `-t 4` default and inside the error bars.
  Decode is memory-bandwidth bound and more cores add no bandwidth, so threads are
  nearly as much of a no-op as on the GPU sibling — for a different reason.
  `THREADS=4` and `THREADS_BATCH=8` both survive the sweep: 4 is within noise of
  the best decode cell, and 8 **pinned to the P-core logical threads** is the
  fastest prefill cell at 139.77 t/s, vindicating the `tpu.env` rationale.
  Run: `benchmarks/runs/2026-09-16-thread-sweep-cpu/`.
- **CPU affinity is the real lever, and nothing here sets it.** Prefill spans
  86.61-139.77 t/s — a **1.61x swing** decided by *which* cores run it, not how
  many threads. **E-cores are stragglers**: 4 P-cores alone prefill at 135.52, and
  adding all 8 E-cores DROPS that to 116.77 (-13.8% for 8 extra cores), because
  llama.cpp splits work evenly and every barrier waits on the slowest thread.
  E-cores alone are 86.61 pp / 19.17 tg. SMT, by contrast, is neutral for prefill
  (+3.1%, error bars overlap, same 4 physical cores) and costs decode 5.6%.
  **A mask must never become a `tpu.env` key.** `0x55` is a fact about this die,
  not a setting — it means "one thread per P-core" only on a host with 4 SMT
  P-cores at logical 0-7. Elsewhere it silently pins to the wrong cores. Derive it
  from sysfs the way `cpu_status` already does: the prefill set is
  `/sys/devices/cpu_core/cpus` (absent ⇒ homogeneous ⇒ online CPUs), the decode set
  is the lowest of each distinct `thread_siblings_list` within it, mask is
  `sum(1 << cpu)`. On this host that derives `0xFF` and `0x55`, the two masks
  measured, and it degrades correctly when there is no hybrid split and no SMT.
  `THREADS`/`THREADS_BATCH` share the defect in milder form: 4 and 8 are the sizes
  of those two derived sets, frozen as integers. `server.py` passes no affinity
  flags, still gated on evidence rather than design — **engine-level only**
  (`llama-bench`, no HTTP path), and the prefill win is invisible to a short
  prompt, so it is unconfirmed end-to-end.
- **KV `q8_0` was a 12% decode loss on CUDA.** Whether AVX2 dequant costs the same
  is open. `f16` is kept because the KV is ~60 MiB and memory is not scarce.
- **`-fa 1` was +4.8% on CUDA.** Unknown on CPU. Every cell of the 2026-09-16
  sweep ran `-fa on`, so it is held constant there, not measured.

- **The `CPU_REPACK` buffer is real — MEASURED 2026-09-16.** `RssAnon` =
  1,202,152 kB with the model loaded and idle, against a predicted ~1.08 GB Q4_0
  body plus ~60 MiB of KV and the compute buffers. On x86 llama.cpp repacks Q4_0
  weights into an interleaved layout for its AVX2 kernels, copying the body out of
  the mmap into anonymous memory. **Do not look for this in the allocation log**:
  commit `82324fc50` prints no per-buffer table even at verbosity 3 — the whole
  load is 19 lines. `grep RssAnon /proc/<pid>/status` is the route that works, and
  it gives one total rather than the sibling's per-term breakdown, so it cannot
  tell you which term is wrong when the total is right.

## The memory arithmetic still holds, and means something different

MEASURED-STATIC on the artifact (541 tensors, 3.334 GB of tensor bytes):

| tensor | shape | type | bytes | touched per token |
| --- | --- | --- | ---: | --- |
| `per_layer_token_embd.weight` | `[8960, 262144]` | Q6_K | 1926.8 MB | **a few rows — lazy** |
| `token_embd.weight` | `[1536, 262144]` | Q6_K | 330.3 MB | a few rows |
| 35 transformer blocks | — | Q4_0 | ~1080 MB | all of it |

`per_layer_token_embd` is created with `TENSOR_READ_LAZY` (`src/models/gemma4.cpp`)
and served by `GGML_OP_GET_ROWS` out of the mmap. On the GPU that decided whether
the model fit a 4 GiB card. **On a CPU it decides what stays hot in the page
cache**: the Q4_0 body is read every token, the PLE a few rows at a time. With 15
GiB of host RAM, fit is not the question it was.

- **On this host the lazy-PLE claim is still untested.** At first light `RssFile`
  was 3.216 GB of the 3.349 GB file — essentially the whole mmap resident — but the
  page cache was hot from the download minutes earlier, so that number says nothing
  about what inference actually touches. Testing it needs a dropped cache.
- **Never pass `--no-mmap`.** `TENSOR_READ_LAZY` "requires mmap for now". A test
  asserts it.
- **The file is only 32% Q4_0.** Both embeddings are Q6_K (2.257 of 3.334 GB).
  Slot 5 says `q4_0` because `MODEL_NAME` does. Record the per-tensor split in any
  report.

These facts describe the checkpoint and llama.cpp, not a device, and are filed in
`@MODELS.md`. Correct them there.

## Gemma 4 reasons, and that is the second way to get an empty reply

MEASURED on the 1650ti sibling, and an engine property rather than a device one.
Gemma 4 emits a thinking block; llama.cpp routes it to `reasoning_content` and
leaves `content` empty until the block closes. "Name three TPU generations" spent
1274 characters reasoning. `query_model` defaults to `max_tokens=1024` (a test
keeps it ≥512) and returns 📡 rather than ✅ when only reasoning came back.

**On a CPU this costs minutes, not seconds.** `query_model`'s HTTP timeout is 900 s
and `make query` passes `--max-time 900`; both are sized, not measured.

## `sweep.py`

Copied from the 1650ti sibling with its two fixes (count `delta.reasoning_content`
as well as `delta.content`; coalesce `None` before slicing an error). Two further
fixes in `local-ollama-1650ti-2b-q4_0`'s copy are **not** in this one: the Ollama
`reasoning` field name, and seeding the shuffled-prompt RNG from entropy. The
filler text was changed from a GPU description to a CPU one on retargeting, so
prompt token counts will not match the sibling's exactly.

- **`decode_tps` counts inter-chunk gaps.** Check `chunks_match_usage` before
  differencing it against another rig; `end_to_end_tps` is unaffected.
- **`--decode-source auto` must resolve to `stream`.** `llama-server` does not emit
  `usage.decode_tokens_per_second`. Do not fill it from `/metrics`.
- **Verify the prompt cache is defeated** via `/metrics`
  (`llamacpp:prompt_tokens_cached_total`), which is on (`METRICS=1`).
- **llama.cpp splits `--ctx-size` across `--parallel` slots.** Context and
  concurrency sweeps need differently-sized servers; get it wrong and cells queue
  instead of erroring.

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

`@MODELS.md`, `@HARDWARE.md`, `@QUANTIZATION.md`, `@NAMING.md`,
`@RIG-ANALYSIS.md`. Read them before deriving their numbers here, and correct them
there.
