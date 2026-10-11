# 2026-10-11 — first light, Snapdragon 850 (Lenovo Yoga C630)

Gemma 4 E2B-it, exact Q4_0 GGUF v2 (`xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf`,
SHA-256 `419db9a6…`, verified after download), llama.cpp `23b0202a1`, CPU-only
build (`GGML_NATIVE=ON`, no dot product detected), Debian sid arm64, gcc 16.2.0.

## Method

`bench.sh`: `llama-bench -p 512 -n 128 -r 3 -fa 1 -ngl 0`, one cell per
thread/affinity setting, run in the order below. Before each cell the script waits
until the hottest of the eight per-core sensors reads ≤55 °C (idle is ~49 °C);
`results/cells.txt` records the hottest sensor at start and end. The machine is
fanless and a GNOME session was running. Nothing else was started during the run.
Cpus 0-3 are Silver (A55-class), 4-7 Gold (A75-class).

## Results (tok/s, mean ± stddev of 3)

| cell | flags | pp512 | tg128 | °C start → end |
| --- | --- | ---: | ---: | --- |
| 4 Gold | `-t 4 -C 0xF0` | 7.14 ± 0.07 | **5.00 ± 0.03** | 55.4 → 77.7 |
| all 8 | `-t 8` | **9.22 ± 0.03** | 2.18 ± 0.04 | 54.4 → 79.7 |
| 4, unpinned | `-t 4` | 6.97 ± 0.08 | 5.01 ± 0.03 | 56.1 → 77.0 |
| 2 Gold | `-t 2 -C 0x30` | 4.39 ± 0.00 | 3.26 ± 0.01 | 55.1 → 74.3 |
| 4 Gold + 2 Silver | `-t 6 -C 0xF3` | 8.26 ± 0.02 | 4.59 ± 0.11 | 54.1 → 79.0 |
| 4 Silver | `-t 4 -C 0x0F` | 3.22 ± 0.00 | 2.28 ± 0.00 | 55.4 → 53.4 |

End-to-end through `make serve` (`-t 4 -tb 8`, one chat request, 22-token
prompt, 29 tokens out): prefill 8.6 tok/s, **decode 5.49 tok/s**, from the
server's own `timings`. A short prompt and a short answer, so it is a smoke test,
not a sweep cell.

## Findings

1. **Decode belongs on the Gold cores only.** Adding Silver cores to decode
   *hurts*: all 8 threads decode at 2.18, under half of 4 Gold (5.00), and even
   below 4 Silver alone (2.28) — every token waits on the slowest thread. Two
   Silver threads beside four Gold already cost 8% (4.59).
2. **Prefill wants every core.** 9.22 at 8 threads against 7.14 at 4 Gold
   (+29%); the Silver cluster adds throughput when the work is a big batch.
   Hence `THREADS=4`, `THREADS_BATCH=8`.
3. **Decode is compute-bound here, unlike on the x86 parent.** 2 → 4 Gold
   threads is 3.26 → 5.00 (1.53x). On the i7-10750H decode was flat from 4 to 6
   threads (bandwidth-bound). The Kryo 385 has no `asimddp` and no i8mm, so
   llama.cpp skips its Q4_0 repack (`ggml-cpu/repack.cpp` gates the ARM layouts
   on i8mm or dot product) and runs the generic NEON path.
4. **No repacked copy of the weights.** With the server loaded and idle after one
   request: `RssAnon` 191,696 kB, `RssFile` 2,557,880 kB. The x86 parent measured
   `RssAnon` 1,202,152 kB because AVX2 repacks the Q4_0 body into anonymous
   memory. Here the weights stay in the page cache, which is what makes a 2.62 GB
   model workable beside a desktop on 7.3 GiB.
5. **The scheduler already does the right thing with 4 threads.** Unpinned `-t 4`
   matches pinned Gold within noise on decode (5.01 vs 5.00) and 2% on prefill.
   With 8 threads there is nowhere for it to hide the stragglers.

## Caveats

- **Thermals.** Every Gold-including cell went from 55 °C to 74-80 °C within
  one cell, on a fanless chassis. The cooldown gate gives each cell the same
  starting temperature, but not the same thermal path. The Silver-only cell ran
  *cooler* than it started, so it never reached the throttled regime the others did.
- One pass, fixed order, no A-B-B-A. The stddevs are within-cell; between-pass
  drift on this host is unmeasured.
- **Against `local-llamacpp-cpu-2b-q4_0`** (i7-10750H, `f95b0d9`, spot-check
  2026-09-22: pp512 89.32, tg128 17.53 at `-t 6`): roughly 12x on prefill and
  3.5x on decode. That compares different commits, days and protocols, so it is
  indicative, not a paired result.
- llama-bench output, not a `serving-report.schema.json` report, so
  `benchmarks/rollup.py` does not count it.
