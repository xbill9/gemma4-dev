# CLAUDE.md — local-llamacpp-cpu-2b-q4_0

Guidance for working inside this rig. The siblings are not layers; nothing is
imported across a rig boundary. Read this file before changing anything.

## What this rig is

`llama-server` from `ggml-org/llama.cpp`, driven directly, serving
`google/gemma-4-E2B-it-qat-q4_0-gguf` on the **CPU only** of this workstation. No
GPU, no control plane, no cloud. One process, one GGUF file named on the command
line.

**STATUS 2026-09-22: serving on `f95b0d9`. HOST HARDWARE CORRECTED, SWEEP
QUARANTINED, NO VALID ABSOLUTE NUMBERS.** llama.cpp is built CPU-only at
`~/llama.cpp/build-cpu`, rebuilt from clean on 2026-09-22 after the host moved to
Debian sid (gcc 16.2, CUDA 13.4); both arms are on `f95b0d9`, 100 commits on from
the `c6824a9` they were paired at. Verified: this arm lists **no devices**, the
GPU arm lists `CUDA0`.

**The CPU in this rig's own documentation was not the CPU in this machine.** It
claimed an i7-1360P, 12 cores / 16 threads, hybrid 4 P + 8 E. It is an
**i7-10750H, 6 cores / 12 threads, homogeneous** — see the Host table. That
invalidates the interpretation of the 18-cell thread/affinity sweep, whose
headline (E-core stragglers, a 1.61x affinity swing) describes a die that is not
here; the run is kept and quarantined. `THREADS`/`THREADS_BATCH` are re-derived to
`6`/`12` and spot-checked, not swept.

`benchmarks/runs/2026-09-16-paired-sweep-cpu` (8/8 cells, the CPU arm of the first
controlled A/B here) still stands as a **paired ratio** — decode flat at
15.66-17.61 tok/s across a 21x range of prompt length, TTFT linear in it, 1.1 s at
94 tokens to 23.9 s at 1959, with the GPU arm 4.27x on decode and 3.63x on
prefill. The ratio is what survives; its **absolute** CPU figures carry the same
thermal problem as everything else here, and the old note that affinity made 3.63x
an "upper bound" is withdrawn with the 1.61x it rested on.

**Nothing measured in this rig came off the binary now installed.** Re-run before
quoting any figure.

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

Order effects are real on this host and are **not** yet controlled for — and as
of 2026-09-22 they are known to be *larger than any lever measured here*. A
6-core i7-10750H and a Max-Q card share one thermal envelope, so the CPU arm
saturating all 12 threads leaves the package hot and downclocked for whatever runs
next. Measured: 28,271 package throttle events, 74 °C before a benchmark and 92 °C
after, and an identical config re-run cold moved **19% on decode**. Page
cache is the same kind of problem from the other side — this arm reads the Q4_0
body every token across a 3.35 GB mmap, while the GPU arm loads to VRAM once.
Neither is measured. Until they are: fix an order, cool down between arms, keep
the cache state the same rule for both, and **record what you did in the report**
rather than assuming it did not matter. A-B-B-A detects the drift if you can
afford four runs.

Do not start an arm while anything else is loading the machine. A model load was
lost to an OOM on 2026-09-16 because a 12-way CUDA build was running beside it,
and the CUDA build OOMed again on 2026-09-22 for a related reason — see
"Building llama.cpp here" below.

### `start_model_server` could not leave a server running (fixed 2026-09-16)

It spawned through `asyncio.create_subprocess_exec` with `start_new_session=True`.
Asyncio's subprocess transport **kills a live child when it is torn down** —
`BaseSubprocessTransport.__del__` calls `close()`, which calls `_proc.kill()` on
a child that has not exited — and a new session does not prevent it. MEASURED
2026-09-16: a spawned `sleep 60` is dead within a second of the interpreter
exiting. `make serve` (a plain exec) was never affected.

That is very likely the real reason "no pid file" is documented here as the
NORMAL case: the only path that writes one could not leave a server behind. The
spawn is now `subprocess.Popen`, which merely warns when collected with a live
child. Still no shell — the rule is against `shell=True`, not against the module.

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

## Building llama.cpp here

**`-j` WITHOUT A NUMBER IS WHAT OOMs THIS MACHINE.** `cmake --build … -j` with no
value passes an unbounded `-j` to make, which will start **all 188 CUDA
translation units at once**; each `nvcc` forks `cicc` and `cudafe++`, and 15 GiB
of RAM does not survive it. That is what killed the build on 2026-09-22 — the
`dmesg` victims are `cicc`, not the linker — and it is the same shape as the
2026-09-16 model-load OOM. Both Makefiles now pass an explicit `-j`; do not
"simplify" it back. MEASURED 2026-09-22: the full CUDA rebuild at `-j6` peaks at
**4.0 GiB and never touches swap**, so bounding it is not a tradeoff.

Swap was raised to 47 GiB the same day (15.7 G partition + 32 G `/swapfile`,
in `/etc/fstab`). It is headroom, **not the fix** — at `-j6` none of it is used.

**Toolchain, 2026-09-22 (Debian sid / forky, kernel 7.2.6):** gcc **16.2.0**,
CUDA **13.4**. Two things to know before touching either:

- **CUDA 13.4's host-compiler ceiling is gcc 16** (`crt/host_config.h`: "later
  than 16 are not supported"). sid is inside it by exactly one version. The next
  gcc will need `-allow-unsupported-compiler` or a pinned `gcc-14`, which is
  installed.
- **`sm_75` is now the LOWEST arch CUDA ships.** `nvcc --list-gpu-arch` starts at
  `compute_75`, and the 1650 Ti is 7.5 — right on the floor. The next major CUDA
  is the one to expect to strand this card.

**Reconfigure from clean after a toolchain change.** Both build directories' stale
`CMakeCache.txt` still recorded gcc 14 and CUDA 13.3; `/usr/local/cuda` is a
symlink, so the cached path looked current while the detected compiler behind it
was not. Both arms were wiped and re-configured rather than rebuilt in place.

## Host — RE-MEASURED 2026-09-22

| | |
| :--- | :--- |
| Machine | Lenovo **Yoga 9 15IMH5** (DMI 82DE) |
| CPU | Intel Core **i7-10750H** (Comet Lake), **6 cores / 12 threads** |
| Topology | **homogeneous, no P/E split.** SMT siblings pair `(0,6) (1,7) … (5,11)` |
| SIMD | `avx avx2` — **no AVX-512 and no `avx_vnni`** |
| RAM | 15 GiB total, ~13 GiB available when checked |
| Swap | 15.7 G partition + 32 G `/swapfile` = **47 GiB** (added 2026-09-22) |
| GPU | **GTX 1650 Ti Max-Q, 4096 MiB, driver 615.71.09** — present, and the other arm |
| Thermals | throttles hard: 28,271 package throttle events, 74 → 92 °C under load |

The `cpu_status` tool re-reads all of this from `/proc` and `/sys`.

**Every CPU row above was wrong until 2026-09-22.** The table read "i7-1360P, 12
cores / 16 threads, hybrid 4 P + 8 E at logical 0-15, `avx_vnni`". The kernel says
a homogeneous 6-core i7-10750H with 12 logical CPUs and no VNNI;
`/sys/devices/cpu_core` and `/sys/devices/cpu_atom` **do not exist here**, and that
absence is the P/E test. DMI confirms a Yoga 9 15IMH5, which is an i7-10750H
machine — no 1360P laptop ships a GTX 1650 Ti Max-Q, so the two rows never fitted
each other.

**This is the second time this table pointed the wrong way, and the expensive
one.** The GPU row was wrong until 2026-09-16 — it read "none (`/dev/nvidia*`
absent)", which made the CPU-only guards look like belt-and-braces on a machine
with no GPU to hit, when they are the only thing keeping this arm honest. They do
work — `start_model_server` refuses the GPU build, verified. The CPU rows were
worse: a whole sweep was interpreted through a die that is not here.

**Where the wrong CPU came from — established 2026-09-22.** It is not invented.
`local-jax-cpu-2b`'s host block records "i7-1360P, 16 logical cores", `MemTotal`
14,682,148 kB and **btrfs on `/dev/vdb`**. `/dev/vdb` is a *virtio* disk: that rig
was measured **inside a VM**, and its numbers are probably correct for where they
were taken. They are not correct here — this box is bare metal
(`systemd-detect-virt: none`), nvme + ext4, `MemTotal` 16,035,492 kB. The facts
crossed a rig boundary and were never re-measured. That is precisely what the
root rule — *"the rigs are siblings, not layers; read the rig you are in"* —
exists to stop, and it cost a sweep.

It also means the quarantined sweep's `topology.json` (16 CPUs, hybrid, 4 P-cores)
is probably a faithful record of **that VM**, not a corrupt file. It is the label
that is wrong, not the measurement.

**`local-pytorch-cpu-2b/CLAUDE.md` was right about this host all along** and this
file overruled it. It says 6 physical cores and six SMT siblings; that is exactly
what `lscpu` reports here. The instruction here used to be "trust the kernel"
while quoting numbers that were not the kernel's. Trust the kernel — and check
you actually read it, rather than a sibling's inherited copy.

## Nothing tuned on the GPU transfers — and nothing swept here survived either

Every lever the 1650ti sibling measured was measured against a CUDA device.
**`KV q8_0` and `-fa 1` remain UNMEASURED on this host.** The two that were
briefly believed measured here were interpreted through the wrong CPU and are
quarantined; what replaces them is weaker, and honestly so.

- **The 2026-09-16 thread/affinity sweep is QUARANTINED (2026-09-22).** Its 18
  cells describe a 4 P-core + 8 E-core die. This host is a homogeneous 6-core
  i7-10750H. The run's own `results/topology.json` records 16 online CPUs,
  `hybrid: true`, 4 perf cores and an `0xFF00` "E-cores only" mask; its
  `topology.py`, re-run here, emits 12 CPUs, `hybrid: 0`, 6 cores and an **empty**
  efficiency set — that cell cannot exist on this machine. Everything downstream
  goes with it: "E-cores are stragglers", the **1.61x affinity swing**, and the
  claim that `THREADS=4`/`THREADS_BATCH=8` were vindicated. `0x55` on this die is
  cpus 0,2,4,6 — and cpu6 is the SMT **sibling** of cpu0, so it pins to 3 physical
  cores with one doubled. Run kept, not deleted:
  `benchmarks/runs/2026-09-16-thread-sweep-cpu/` with a quarantine header.
- **Threads: direction only, and the direction is the ordinary one.** Spot-check
  2026-09-22 (llama-bench, f95b0d9, pp512/tg128, r=2, one invocation so the cells
  share a thermal state): prefill 78.74 → 89.32 → 93.42 t/s at `-t` 4/6/12, decode
  17.72 → 17.53 → 15.54. Prefill scales with threads (+18.6%), decode does not and
  SMT costs it (-12.3% at 12). That is the expected bandwidth-bound decode story,
  and it is why `THREADS=6` (physical cores) and `THREADS_BATCH=12` (logical) are
  now the derived values. It is **not** a sweep.
- **Affinity is now an open question, not the answer.** `-t 6 -C 0x3F` measured
  pp512 94.32 against 89.32 unpinned — then the *same cell* re-run cold measured
  83.98. See the thermal bullet; the drift is bigger than the effect.
- **THE REAL FINDING OF 2026-09-22: this host cannot support absolute numbers as
  it stands.** It logs **28,271 package throttle events**, sits at 74 °C before a
  benchmark and 92 °C after, and an identical config re-run moved **19% on decode**
  (tg128 17.70 → 14.29). Run-to-run drift exceeds every lever anyone has tried to
  measure here. `CLAUDE.md` already warned that order effects "are real on this
  host and are **not** yet controlled for" — that warning is now the headline, not
  a caveat. Before any number from this rig is quoted: fixed cell order, cooldown
  to a stated temperature between cells, interleaved A-B-B-A, and the thermal log
  recorded in the report.
- **The derivation rule survives; every mask in it was wrong.** A hex mask is a
  fact about one die, never a `tpu.env` key — that part was always right, and is
  exactly why the frozen `0xFF`/`0x55` did damage. From sysfs, as `cpu_status`
  already does: prefill set is `/sys/devices/cpu_core/cpus` (absent ⇒ homogeneous
  ⇒ online CPUs), decode set is the lowest of each distinct
  `thread_siblings_list` within it, mask is `sum(1 << cpu)`. On **this** host that
  derives `0xFFF` and `0x3F`. `server.py` still passes no affinity flags.

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
