# 2026-10-11 — first light, Snapdragon 7c Gen 2 (Samsung Galaxy Book Go)

Gemma 4 E2B-it, exact Q4_0 GGUF v2 (`xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf`,
SHA-256 `419db9a6…`, verified after download), llama.cpp `f2cc7282c`, CPU-only
build (`GGML_NATIVE=ON`, **dot product detected**, no i8mm), Debian forky/sid
arm64, kernel 7.2.9, gcc 16.2.0.

## Method

`bench.sh`: `llama-bench -p 512 -n 128 -r 3 -fa 1 -ngl 0`, one cell per
thread/affinity setting, run in the order below. Before each cell the script waits
until the hottest of the ten `cpuN-thermal` sensors reads ≤45 °C (idle is ~36 °C);
`results/cells.txt` records the hottest sensor at start and end, the pages swapped
in during the cell (`pswpin` from `/proc/vmstat`), and llama-bench's peak RSS and
major faults (`/usr/bin/time`). The machine is fanless and a GNOME session was
running. Cpus 0-5 are Silver (A55-class), 6-7 Gold (A76-class).

## Results (tok/s, mean ± stddev of 3)

| cell | flags | pp512 | tg128 | °C start → end | swapped in |
| --- | --- | ---: | ---: | --- | ---: |
| 2 Gold | `-t 2 -C 0xC0` | 36.36 ± 2.41 | **8.33 ± 0.04** | 38.4 → 51.2 | 1.20 GiB |
| all 8 | `-t 8` | **46.68 ± 0.83** | 5.17 ± 0.28 | 41.3 → 50.9 | 0.92 GiB |
| 2, unpinned | `-t 2` | 39.18 ± 0.24 | 8.28 ± 0.05 | 42.2 → 55.4 | 0.57 GiB |
| 1 Gold | `-t 1 -C 0x40` | 18.42 ± 2.73 | 7.64 ± 0.08 | 43.5 → 47.7 | 1.41 GiB |
| 2 Gold + 2 Silver | `-t 4 -C 0xCC` | 39.79 ± 0.07 | 6.40 ± 0.10 | 40.0 → 55.7 | 0.30 GiB |
| 6 Silver | `-t 6 -C 0x3F` | 19.92 ± 0.08 | 5.53 ± 0.08 | 42.6 → 42.9 | 0.77 GiB |

Peak RSS was 2.40-2.61 GB in every cell.

End-to-end through `make serve` (`-t 2 -tb 8`, one chat request, 18-token
prompt, 29 tokens out): prefill 18.4 tok/s, **decode 7.78 tok/s**, from the
server's own `timings`. Server memory after that request: `RssAnon` 1,440,836 kB,
`RssFile` 729,252 kB, `VmHWM` 2,452,612 kB. A smoke test, not a sweep cell.

## Findings

1. **Decode belongs on the two Gold cores only.** All 8 threads decode at 5.17,
   62% of 2 Gold (8.33); adding just two Silver threads costs 23% (6.40). Same
   straggler effect the SDM850 sibling measured.
2. **Prefill wants every core.** 46.68 on all 8 against 36.36-39.18 on 2 Gold
   (+19-28%). Hence `THREADS=2`, `THREADS_BATCH=8`.
3. **Dot product is the headline, and it is visible in memory.** The Kryo 468 has
   `asimddp`, so llama.cpp repacks the Q4_0 body for SDOT: `RssAnon` 1.44 GB
   against the SDM850's 192 MB. Two A76 cores at 2.55 GHz decode 1.67x the
   SDM850's four A75 cores (8.33 vs 5.00) and prefill 5.1x (all 8: 46.68 vs 9.22).
4. **Decode is close to saturated at one core.** 1 → 2 Gold is 7.64 → 8.33
   (1.09x), where the SDM850 scaled 1.53x from 2 to 4. With SDOT the compute per
   token is cheap enough that the second core mostly waits on memory — but see
   the swap caveat before calling it bandwidth.
5. **Memory, not compute, is this host's ceiling.** 3.4 GiB of RAM, shared with
   a desktop session, holds a 1.44 GB repacked copy of the weights beside a 2.62 GB
   mmap. Every cell swapped in 0.3-1.4 GiB.

## Caveats

- **Swap contaminates every cell.** Swap-in varied 4.6x between cells and does
  not track the throughput ranking, so it is noise of unknown size rather than
  a correctable offset. The 2.41 and 2.73 prefill stddevs in the pinned Gold
  cells are probably this. A cleaner run needs the desktop session stopped.
- Thermals stayed mild (max 55.7 °C, against the SDM850's ~80); the 45 °C gate
  held each start within 5 °C. No throttling is evident, but none was tested for.
- One pass, fixed order, no A-B-B-A.
- **Against `local-llamacpp-sdm850-2b-q4_0`** the commits differ (`f2cc7282c`
  here, `23b0202a1` there), as do the RAM and the cooldown threshold (both rigs ran the same day),
  so the ratios in finding 3 are indicative, not a paired result.
- llama-bench output, not a `serving-report.schema.json` report, so
  `benchmarks/rollup.py` does not count it.
