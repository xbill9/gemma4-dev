# CLAUDE.md — local-llamacpp-sc7180p-2b-q4_0

Guidance for working inside this rig. The siblings are not layers; nothing is
imported across a rig boundary. Read this file before changing anything.

## What this rig is

`llama-server` from `ggml-org/llama.cpp`, driven directly, serving the exact
Q4_0 GGUF of Gemma 4 E2B-it (`xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf`, v2,
SHA-256 `419db9a6…`) on the **CPU of a Samsung Galaxy Book Go** — a Snapdragon
7c Gen 2, aarch64, fanless, 4 GB. No GPU backend, no control plane, no cloud.

**Forked 2026-10-11 from `local-llamacpp-sdm850-2b-q4_0`** (Snapdragon 850,
Lenovo Yoga C630). Slot 3 is the only slot that differs: same runtime, same GGUF,
same serving flags apart from threads. The pair is the tree's **with-vs-without
dot product** comparison on identical weights — but **not a controlled A/B**:
different machines, different RAM, different llama.cpp commits (`f2cc7282c`
here, `23b0202a1` there). Quote a ratio between them as indicative only.

**Slot 3 is `sc7180p`**, the vendor part number, per `NAMING.md`'s local-CPU
rule. The device tree says `qcom,sc7180` because mainline Linux describes the 7c
Gen 1 (SC7180) and Gen 2 (SC7180P, the higher-clocked bin) with one tree, and
lscpu says only "Kryo-4XX". Samsung ships the Galaxy Book Go with the 7c Gen 2.
Do not "correct" the name from either string.

The x86 grandparent's machinery came along unchanged and mostly does not apply:
`attest.py`'s CUDA checks and the `-ngl 0` / `CUDA_VISIBLE_DEVICES=` guards are
inert (harmless — the build has no GPU backend). `cpu_status` reads the
Intel-hybrid path `/sys/devices/cpu_core`, which does not exist on ARM, so **it
reports this die as homogeneous, which is wrong**; the truth is in
`cpu_capacity` (below).

## Host — MEASURED 2026-10-11

| | |
| :--- | :--- |
| Machine | Samsung **Galaxy Book Go** (Wi-Fi), bare metal, **fanless** |
| SoC | Qualcomm **Snapdragon 7c Gen 2**, SC7180P (DT: `qcom,sc7180`) |
| Cores | **cpu0-5 Kryo 468 Silver** (A55-class, capacity 293, ≤1805 MHz); **cpu6-7 Kryo 468 Gold** (A76-class, capacity 1024, ≤2554 MHz). No SMT |
| SIMD | `asimd fphp asimdhp asimdrdm asimddp` — **dot product yes**, no i8mm, no SVE |
| RAM | **3.4 GiB**, shared with a GNOME session; 3.3 GiB swap on `/dev/sdb4` |
| OS | Debian forky/sid arm64, kernel 7.2.9, gcc 16.2.0 |
| Thermals | idle ~36 °C; a 2-thread llama-bench cell peaks ~51-56 °C. CPU sensors are `thermal_zone{1-6,9-12}` |

**Only two big cores, and they are the two highest-numbered.** A mask or
`taskset` that takes "the first N cpus" picks the slow cluster. Gold-only is
`-C 0xC0`.

**Dot product changes the memory picture, not just the speed.** With `asimddp`
llama.cpp repacks the Q4_0 body into SDOT layout in anonymous memory:
`RssAnon` 1.44 GB here against 192 MB on the SDM850, on a machine with half the
RAM. **Memory is this host's binding constraint**; every bench cell swapped.

## Build

`make build` works as is (`GGML_NATIVE=ON` detects
`-mcpu=generic-armv8-a+lse+dotprod+rdma+crc+crypto+fp16+rcpc`). The 2026-10-11
build used ninja at `-j6` alongside the model download and took ~25 minutes with
2.7 GiB in swap — **pass `BUILD_JOBS=4` or lower**; `nproc` = 8 is the Makefile
default and too many for 3.4 GiB. cmake, ninja, `time` and `hf` came from apt
(`cmake ninja-build time python3-huggingface-hub`) — this host's system python
has no pip.

## Measurements

`benchmarks/runs/2026-10-11-first-light-sc7180p/` — llama-bench pp512/tg128,
r=3, `-fa 1`, each cell gated on the hottest CPU sensor cooling to ≤45 °C first,
with temperature, swap-in and peak RSS recorded. Headlines:

| | pp512 | tg128 |
| --- | ---: | ---: |
| 2 Gold (`-t 2`, the serving value) | 36.36 | **8.33** |
| 1 Gold | 18.42 | 7.64 |
| all 8 | **46.68** | 5.17 |
| 6 Silver | 19.92 | 5.53 |

End to end through `make serve`: decode 7.78 tok/s on one short chat request.

- **Never put decode on the Silver cores.** All 8 threads decode at 62% of 2
  Gold. `THREADS=2` / `THREADS_BATCH=8` follow from that.
- **Decode barely scales past one core** (1 → 2 Gold: 1.09x, against the
  SDM850's 1.53x from 2 to 4). With SDOT, compute stops being the limit.
- **Every cell swapped 0.3-1.4 GiB**, varying 4.6x, so treat run-to-run
  differences under ~10% as noise until a run is made with the desktop stopped.

## Inherited and still true here

- **Never pass `--no-mmap`.** `per_layer_token_embd` is `TENSOR_READ_LAZY`, which
  requires mmap. A test asserts it. On 3.4 GiB this is not optional: the lazy
  table and the page cache are the only reason a 2.62 GB file plus a 1.44 GB
  repack serve at all.
- Gemma 4 reasons, and llama.cpp routes the thinking block to
  `reasoning_content`; `REASONING=off` is the server default. At ~8 tok/s a
  1,000-token reasoning block is two minutes.
- `sweep.py`: `decode_tps` counts inter-chunk gaps; verify the prompt cache is
  defeated via `/metrics`; llama.cpp splits `--ctx-size` across `--parallel`
  slots.

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
