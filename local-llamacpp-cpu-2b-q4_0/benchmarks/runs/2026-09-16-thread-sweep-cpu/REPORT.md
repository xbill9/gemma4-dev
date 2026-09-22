> ## ⛔ QUARANTINED 2026-09-22 — MEASURED THROUGH THE WRONG CPU
>
> **This run's host description is not this machine, and its headline conclusions
> do not survive that.** Kept as a record, not as evidence. Do not cite it.
>
> The report claims an i7-1360P with 4 P-cores + 8 E-cores. The host is a
> **homogeneous 6-core / 12-thread i7-10750H** (Lenovo Yoga 9 15IMH5, DMI 82DE) —
> `lscpu`, `/proc/cpuinfo` and DMI all agree, and `/sys/devices/cpu_core` and
> `/sys/devices/cpu_atom` **do not exist here**, which is the P/E test.
>
> The run's own `results/topology.json` records 16 online CPUs, `hybrid: true`,
> 4 perf cores and an `0xFF00` "E-cores only" mask. Its own `topology.py`, re-run
> on this host on 2026-09-22, emits 12 CPUs, `hybrid: 0`, 6 physical cores and an
> **empty** efficiency set — the `0xFF00` cell cannot exist on this machine.
>
> **What this invalidates:**
> - "E-cores are stragglers" — there are no E-cores here.
> - The **1.61x affinity swing**, and affinity being "the largest measured lever".
> - The claim that `THREADS=4` / `THREADS_BATCH=8` were vindicated. They were the
>   set sizes of the wrong die; the same rule gives **6 and 12** here.
> - The mask labels. `0x55` on this die is cpus 0,2,4,6 — and cpu6 is the SMT
>   **sibling** of cpu0, so it pins to 3 physical cores with one doubled, not to
>   "one thread per P-core".
>
> **What may still be true:** the cells are real `llama-bench` output and the
> *relative* shape (prefill scales with threads, decode does not) was reproduced
> on 2026-09-22. The absolute t/s are not reproducible here in any case — this
> host throttles hard, and an identical config re-run cold moved **19% on decode**.
>
> **To replace this run:** re-derive masks with `topology.py` on the real host,
> build on `f95b0d9`, and use a cooldown protocol with interleaved cell order.

# 2026-09-16 — thread and affinity lever sweep, CPU only

**Rig:** `local-llamacpp-cpu-2b-q4_0` · **Instrument:** `llama-bench` ·
**Engine:** llama.cpp `82324fc50`, `build-cpu` (`GGML_CUDA=OFF`) ·
**Model:** `gemma-4-E2B_q4_0-it.gguf` ·
**Host:** i7-1360P, 4 P-cores +SMT (logical 0-7), 8 E-cores (logical 8-15), avx2+avx_vnni

**18 measured cells, 0 failed, 0 infeasible** — 9 configs × {`pp512`, `tg128`}.
Fixed across every cell: `-ngl 0 -ctk f16 -ctv f16 -fa on -p 512 -n 128`.
`-r 3` unpinned, `-r 5` pinned.

## This is not a serving report

`llama-bench` drives the engine directly with no HTTP path, so nothing here is
comparable to a `sweep.py` figure or to `serving-report.schema.json`, and no file
was written to `benchmarks/reports/`. The reason for using it: `-t`, `-tb` and the
CPU masks are server **launch** flags, so sweeping them through the serving path
needs a server restart per cell.

One cross-check that did land: engine `tg128` at `-t 4` is 25.36 t/s against the
24.8 tok/s measured end-to-end through `llama-server` the same day, so the serving
path costs about 2%.

## Headline

| config | mask | pp512 t/s | tg128 t/s |
| --- | --- | ---: | ---: |
| `-t 4` | unpinned | 135.72 ± 2.64 | 25.36 ± 0.13 |
| `-t 8` | unpinned | 131.86 ± 14.45 | 22.97 ± 0.42 |
| `-t 12` | unpinned | 118.17 ± 0.73 | **25.58 ± 0.27** |
| `-t 16` | unpinned | 132.65 ± 1.55 | 23.10 ± 0.15 |
| `-t 4` 4 distinct P-cores | `0x55` | 135.52 ± 2.69 | 24.89 ± 0.59 |
| `-t 8` 4 P-cores, SMT doubled | `0xFF` | **139.77 ± 5.71** | 23.49 ± 0.39 |
| `-t 12` 4 distinct P + 8 E | `0xFF55` | 116.77 ± 2.18 | 24.96 ± 1.37 |
| `-t 8` E-cores only | `0xFF00` | 86.61 ± 3.75 | 19.17 ± 0.63 |
| `-t 4` 2 P-cores, SMT-paired | `0x0F` | 73.50 ± 2.89 | 21.66 ± 0.83 |

## Findings

**1. Thread count is a weak lever — the rig's prediction was wrong.** `CLAUDE.md`
said "threads are the main lever now, not a no-op". Decode spans 22.97-25.58 t/s
across 4/8/12/16, non-monotonic, and the best cell beats the `-t 4` default by
0.9%, inside the error bars. Decode is memory-bandwidth bound; more cores do not
add bandwidth. Threads are nearly as much of a no-op as on the 1650ti sibling,
for a different reason — that rig was GPU-bound.

**2. Placement is the real lever.** Prefill spans 86.61-139.77 t/s, a **1.61×
swing**, decided by *which* cores run it rather than by thread count.

**3. E-cores are stragglers, and net harmful to prefill.** 4 P-cores alone prefill
at 135.52; adding all 8 E-cores **drops** it to 116.77, -13.8% for 8 extra cores.
llama.cpp splits work evenly across threads, so every barrier waits on the
slowest. E-cores alone manage 86.61 pp / 19.17 tg — 64% and 76% of the P-cores.

**4. SMT is not the prefill mechanism.** The first pass suggested one ("4 and 12
good, 8 and 16 bad"), and it was wrong. Clean control on the *same* 4 physical
P-cores: 8 threads (`0xFF`) 139.77 vs 4 threads (`0x55`) 135.52 — +3.1% with
overlapping error bars. SMT is neutral-to-slightly-good for prefill. It does cost
decode 5.6% (23.49 vs 24.89), error bars just clear of each other.

## One cell is confounded — do not cite it for SMT

`-t 4 -C 0x0F` puts 4 threads on **2** physical cores: half the P-core compute
*and* SMT doubling, mixed together. Its 73.50 pp is roughly what halving the cores
alone predicts (131.86/1.8), so it is evidence about core count, not about SMT.
It is kept in `results/` because it was run, with `-t 8 -C 0xFF` added afterwards
as the clean control. Cite `0xFF` vs `0x55` for SMT, never `0x0F`.

## Practical guidance

- **`THREADS=4` — keep.** Within noise of the best decode cell, and SMT costs
  decode 5.6%, so 4 threads on 4 distinct P-cores is right.
- **`THREADS_BATCH=8` — keep. The `tpu.env` rationale was correct.** Pinned to the
  P-core logical threads it is the fastest prefill cell measured. Unpinned it reads
  131.86 because the scheduler leaks threads onto E-cores.
- **Affinity is unset, and it is where the 1.61× lives — but the mask is not
  portable.** Every hex mask in this report is a fact about an i7-1360P (4 SMT
  P-cores at logical 0-7, 8 E-cores at 8-15), not a setting. `0x55` means "one
  thread per P-core" *here*; on a homogeneous CPU it pins to a quarter of the
  machine. So it must be derived, not configured: prefill set =
  `/sys/devices/cpu_core/cpus` (absent ⇒ online CPUs), decode set = lowest of each
  distinct `thread_siblings_list` within it, mask = `sum(1 << cpu)`. That derives
  `0xFF`/`0x55` on this host and degrades correctly without a hybrid split or SMT.
  `llama-server` spells the split `-C`/`--cpu-strict` and `-Cb`/`--cpu-strict-batch`.
  **Engine-level only so far** — the prefill win is invisible to a short prompt, so
  confirming it end-to-end needs `sweep.py`'s context sweep, which has not been run.
- **`-t 4`/`-tb 8` are the same kind of frozen fact**, in milder form: they are the
  sizes of those two derived sets on this host.

## Reproducing

`./run.sh` — stop `llama-server` first so it is not holding ~4.2 GB and 4 threads
beside the benchmark. Raw output in `results/llama-bench-threads.md` and
`results/llama-bench-affinity.md`; machine-readable cells in `results/summary.json`.

## Re-running on another host

`./run.sh` is portable: `topology.py` derives every thread count and every mask
from sysfs, so nothing in it is specific to this die. On an i7-1360P it
regenerates this run's exact configuration — thread list `4,8,12,16` and masks
`0x55`/`0xFF`/`0xFF00`/`0xFF55`. On a homogeneous non-SMT CPU the prefill and
decode masks collapse to the same value and the two E-core cells are **recorded
as infeasible** rather than dropped. `results/topology.json` records what was
derived, so two hosts' runs can be told apart after the fact.

Three things to get right when re-running elsewhere:

- **Match the engine commit.** These numbers are llama.cpp `82324fc50`. A CPU
  figure from a different commit differenced against a GPU figure from
  `95ef7fc` mixes an engine delta into the device delta.
- **`<hw-short>` cannot distinguish two CPUs.** Both hosts' runs would be
  `<date>-thread-sweep-cpu`, since the hardware slot is just `cpu`. Put the CPU
  in the free-form `<what>` — `thread-sweep-i7-1360p` — or the two runs are
  told apart only by date.
- **A different box is not a device-only A/B.** Prefill, sampling and memory
  bandwidth all move with the host, and decode here was bandwidth bound. A CPU
  number from another machine is not this machine's 25 t/s, so the CPU leg has
  to be re-measured on whichever box hosts the GPU.

- **`CUDA_VISIBLE_DEVICES=` stops being belt-and-braces on a box with a GPU.**
  With `-ngl 0` alone a CUDA build still initialises the device and can offload
  large prompt batches; `run.sh` sets it, and the binary must still come from
  `build-cpu` (`GGML_CUDA=OFF`).
