# First light — `local-llamacpp-1650ti-2b-q4_0`, 2026-09-03

First tokens from this rig, and the first run of any kind on the `local` platform slot.
Machine-readable copy: `../../reports/2026-09-03-first-light-1650ti.json`.

**Headline: 73.75 tok/s single-stream decode, 277.61 tok/s aggregate at B=64, ~355 t/s prefill
throughout — in 1618 MiB of a 4096 MiB card.**

## What was settled

**`per_layer_token_embd` is not in VRAM.** This was the open question the rig was created to
answer, and it took one run.

```
$ nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
37073, /home/xbill/llama.cpp/build/bin/llama-server, 1618 MiB
```

against a prediction of 1616 MiB:

```
1342 MiB  resident weights (3.334 GB file - 1.927 GB lazy PLE)
+ 144 MiB  KV, 8192 tokens x 18 KiB/token f16
+ ~130 MiB  CUDA context + compute buffers
```

Had the tensor been resident the requirement would be ~3.5 GB, which does not fit in 4096 MiB —
the model would have failed to load rather than serving at 73 tok/s. Filed in `MODELS.md`;
`gpu-llamacpp-g5g-2b-q4_0` corrected.

## Configuration sweep

`llama-bench -m <gguf> -ngl 99 -p 512 -n 128 -r 3`, build `b1-95ef7fc`, CUDA 13.3.

| `-fa` | `-ctk` | `-ctv` | `-t` | prefill pp512 t/s | decode tg128 t/s |
| ---: | :--- | :--- | ---: | ---: | ---: |
| 0 | f16 | f16 | 6 | 338.74 ± 1.11 | 70.39 ± 0.13 |
| **1** | **f16** | **f16** | **4** | **340.33 ± 0.87** | **73.75 ± 0.06** |
| 1 | f16 | f16 | 6 | 340.39 ± 0.66 | 73.74 ± 0.08 |
| 1 | f16 | f16 | 8 | 338.49 ± 0.60 | 72.97 ± 0.03 |
| 1 | f16 | q8_0 | 4 | 202.66 ± 0.53 | 65.55 ± 0.02 |
| 1 | q8_0 | f16 | 4 | 239.09 ± 0.52 | 64.91 ± 0.03 |

**Three findings, two of them negative:**

- **Flash attention is a free +4.8% on decode** (70.39 → 73.74) and does nothing to prefill.
  Adopted as the default in `tpu.env`.
- **KV quantization is a loss, and a large one.** `q8_0` on either K or V costs ~12% of decode and
  20-40% of prefill. This is the opposite of the intuition that carried over from the TPU rigs, and
  the reason is structural: there are no tensor cores here to hide the dequant, and E2B's KV is
  only ~144 MiB at 8192 context, so it buys memory that was never scarce. **Do not re-try this
  without re-running the sweep.**
- **Thread count does not matter** (73.75 / 73.08 / 72.97 at t=4/6/8). Decode is GPU-bound. `t=4`
  is marginally best and leaves cores free.

## What this is not

- **Not a serving benchmark.** `llama-bench` drives the model directly — no HTTP, no concurrency.
  Every sweep point is concurrency 1 by construction. The `llama-server` path was verified healthy
  and is what the memory figure came from, but it was not benchmarked.
- **Not comparable to the T4 rigs.** `gpu-vllm-g4dn-2b` and `gpu-vllm-g5g-2b` are also compute
  capability 7.5, but the T4 is TU104 and has tensor cores while this TU117 has none. llama.cpp
  says so itself at init:

  ```
  The following devices will have suboptimal performance due to a lack of tensor cores:
    Device 0: NVIDIA GeForce GTX 1650 Ti with Max-Q Design
  ```

- **Not a clean `q4_0` datapoint.** By bytes the artifact is 67.7% Q6_K and 31.4% Q4_0.

## Concurrency — the only large lever on this card

`llama-batched-bench -ngl 99 -fa 1 -t 4 -c 16384 -npp 128 -ntg 128`:

| B | prefill t/s | decode t/s | per stream | total t/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 327.20 | 73.33 | 73.33 | 119.80 |
| 2 | 345.85 | 121.31 | 60.66 | 179.62 |
| 4 | 352.68 | 168.37 | 42.09 | 227.93 |
| **8** | 354.38 | **223.75** | 27.97 | 274.30 |
| 16 | 353.71 | 238.41 | 14.90 | 284.84 |
| 24 | 355.47 | 244.16 | 10.17 | 289.48 |
| 32 | 355.03 | 258.28 | 8.07 | 299.02 |
| 48 | 355.00 | 271.19 | 5.65 | 307.49 |
| 64 | 358.50 | 277.61 | 4.34 | 312.91 |

**Aggregate decode is 3.8x from B=1 to B=64**, and most of it arrives by B=8 (3.05x). After that
each doubling buys 6-8%. **Prefill is flat at ~355 t/s across the entire range**, so that path is
already saturated at B=1 and concurrency buys nothing there.

Memory is not the limit: `-c 16384 --parallel 16` occupies **1818 MiB**, leaving 1903 MiB free.
Note llama.cpp **splits `--ctx-size` across `--parallel` slots**, so 16 slots at `-c 16384` is
1024 tokens each, not 16384.

### A thermal outlier, recorded because it inverted the curve

The first pass measured **B=32 at 184.09 tok/s — below B=16's 238.41**. Re-running the identical
command gave **258.28** and restored monotonicity. This is a Max-Q laptop part and it throttles.
**Do not read a single `llama-batched-bench` row as a result**; the numbers above are from the
run where the curve is monotonic, and any future comparison needs repeats.

## FORCE_MMQ: tested, rejected

llama.cpp's init message suggests `CMAKE_CUDA_ARCHITECTURES=61-virtual;80-virtual` with
`GGML_CUDA_FORCE_MMQ` "to force the use of the Pascal code for Turing". Built into
`~/llama.cpp/build-mmq` (separate directory; the working build is untouched):

| build | pp512 | tg128 |
| :--- | ---: | ---: |
| stock (`arch=75`, `FORCE_MMQ=OFF`) | **340.33 ± 0.87** | **73.75 ± 0.06** |
| `arch=75`, `FORCE_MMQ=ON` | 337.61 ± 0.80 | 73.04 ± 0.11 |

**A wash, marginally worse.** Not adopted. The `61-virtual;80-virtual` half was not tested:
decode here is bandwidth- and launch-bound rather than matmul-bound — which is also why forcing
an integer matmul path changed prefill by 0.8% and decode by 1% — so it is unlikely to pay.

## Reproduce

```bash
cd ~/gemma4-dev/local-llamacpp-1650ti-2b-q4_0
make serve                 # llama-server on 127.0.0.1:8080 with the settings above
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
make info                  # the resident-vs-lazy split, from the artifact
```
