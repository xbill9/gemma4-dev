# CLAUDE.md — local-llamacpp-sdm850-2b-q4_0

Guidance for working inside this rig. The siblings are not layers; nothing is
imported across a rig boundary. Read this file before changing anything.

## What this rig is

`llama-server` from `ggml-org/llama.cpp`, driven directly, serving the exact
Q4_0 GGUF of Gemma 4 E2B-it (`xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf`, v2,
SHA-256 `419db9a6…`) on the **CPU of a Lenovo Yoga C630** — a Snapdragon 850,
aarch64, fanless, 8 GB. No GPU backend, no control plane, no cloud.

**Forked 2026-10-11 from `local-llamacpp-cpu-2b-q4_0`** (i7-10750H, x86_64).
Slot 3 is the only slot that differs: same runtime, same GGUF, same serving flags
apart from threads. That makes the pair an x86_64-vs-aarch64 comparison on
identical weights — but **not a controlled A/B**: different machines, different
llama.cpp commits (`23b0202a1` here, `f95b0d9` there), different days. Quote a
ratio between them as indicative only.

**Slot 3 is `sdm850`**, the vendor part number with its punctuation dropped, per
`NAMING.md`'s local-CPU rule. The device tree says `qcom,sdm845` because mainline
Linux describes the 850 with the 845's tree; the 850 is a higher-binned 845, and
Lenovo ships the C630 with the 850. Do not "correct" the name from that string.

The x86 parent's machinery came along unchanged and mostly does not apply: there
is no GPU arm here, so `attest.py`'s CUDA checks and the `-ngl 0` /
`CUDA_VISIBLE_DEVICES=` guards are inert (harmless — the build has no GPU
backend). `cpu_status` reads the Intel-hybrid path `/sys/devices/cpu_core`, which
does not exist on ARM, so **it reports this die as homogeneous, which is wrong**;
the truth is in `cpu_capacity` (below).

## Host — MEASURED 2026-10-11

| | |
| :--- | :--- |
| Machine | Lenovo **Yoga C630** (DMI 81JL), bare metal, **fanless** |
| SoC | Qualcomm **Snapdragon 850** (DT: `qcom,sdm845`) |
| Cores | **cpu0-3 Kryo 385 Silver** (A55-class, capacity 379, ≤1766 MHz); **cpu4-7 Kryo 385 Gold** (A75-class, capacity 1024, ≤2842 MHz). No SMT |
| SIMD | `asimd fphp asimdhp asimdrdm` — **no `asimddp` (dot product), no i8mm, no SVE** |
| RAM | 7.3 GiB, ~3-4 GiB of it held by the GNOME session; 9 GiB swap |
| OS | Debian sid arm64, kernel 7.2.9, gcc 16.2.0 |
| Thermals | idle ~49 °C; one 4-thread llama-bench cell takes it 55 → ~78 °C |

**The big cores are the high-numbered ones.** A mask or `taskset` that takes
"the first N cpus" picks the slow cluster. Gold-only is `-C 0xF0`.

**No dot product matters more than the clock.** llama.cpp's cmake prints
`-U__ARM_FEATURE_DOTPROD`, so Q4_0 runs on plain NEON multiply-accumulate rather
than SDOT. That is a property of this core generation, not a build mistake.

## Build

`make build` works as is (`GGML_NATIVE=ON` detects
`-mcpu=generic-armv8-a+lse+rdma+crc+crypto+fp16+rcpc`). `BUILD_JOBS` defaults to
`nproc` = 8; the 2026-10-11 build ran at `-j4` alongside the model download and
took ~20 minutes. cmake and `hf` came from apt (`cmake`,
`python3-huggingface-hub`) — this host's system python had no pip.

## Measurements

`benchmarks/runs/2026-10-11-first-light-sdm850/` — llama-bench pp512/tg128,
r=3, `-fa 1`, each cell gated on the hottest CPU sensor cooling to ≤55 °C first,
start/end temperature recorded. Headlines:

| | pp512 | tg128 |
| --- | ---: | ---: |
| 4 Gold (`-t 4`, the serving value) | 7.14 | **5.00** |
| all 8 | **9.22** | 2.18 |
| 4 Silver | 3.22 | 2.28 |

End to end through `make serve`: decode 5.49 tok/s on one short chat request.

- **Never put decode on the Silver cores.** All 8 threads decode at less than
  half of 4 Gold. `THREADS=4` / `THREADS_BATCH=8` follow from that.
- **Decode is compute-bound here** (2 → 4 Gold: 1.53x), unlike the
  bandwidth-bound x86 parent. No dot product means no Q4_0 repack and the generic
  NEON path; `RssAnon` is 192 MB against the parent's 1.2 GB.
- Fanless: 55 → ~78 °C inside one cell. One pass, fixed order — no A-B-B-A yet.

## Inherited and still true here

- **Never pass `--no-mmap`.** `per_layer_token_embd` is `TENSOR_READ_LAZY`, which
  requires mmap. A test asserts it. On 7 GiB this matters more than it did on the
  parent: the lazy table is what lets a 2.62 GB file serve beside a desktop.
- Gemma 4 reasons, and llama.cpp routes the thinking block to
  `reasoning_content`; `REASONING=off` is the server default. At ~5 tok/s a
  1,000-token reasoning block is over three minutes.
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
