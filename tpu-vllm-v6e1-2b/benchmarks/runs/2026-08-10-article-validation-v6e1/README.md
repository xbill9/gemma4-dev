# 2026-08-10 — article validation on v6e-1

Independent re-measurement of every cell published in `devto-vllm-gemma4-e2b-v6e1.md`, on a
**second, freshly provisioned v6e-1 node** in a different zone under a different provisioning
model from the `2026-08-10-config-validation-v6e1` sweep it checks.

`v6e-1`, `europe-west4-a`, **flex-start**, `vllm/vllm-tpu:nightly`, vLLM
`0.26.1rc1.dev256+gf5bb701fa`, `google/gemma-4-E2B-it`, TP=1, `max_model_len` 32768,
`OUTPUT_LEN` 128. Node ACTIVE 22:25:56Z, vLLM serving 22:37Z.

## Allocation — 5/5 PASS, identical to the first node

Every figure reproduced to the digit, on hardware provisioned 12 hours later in another region:

| | value |
| :--- | ---: |
| `total_hbm_limit_gb` | 31.24 GiB |
| `total_hbm_limit_cap_gb` | 28.74 GiB |
| `total_hbm_used_gb` (weights) | 8.97 GiB |
| `total_hbm_avail_gb` (KV pool) | 19.77 GiB |
| `GPU KV cache size` | 1,151,744 tokens |
| `block_size` | 64 |
| `num_kv_cache_groups` / `tensors` | 1 / 15 |
| `num_blocks` | 17,996 (× 64 = 1,151,744 exactly) |
| `regular_attn_dtype` | **bfloat16** |
| `Automatically using fp8_e5m2 …` | logged **20×**, allocated bf16 |

The fp8 result is now reproduced on a second node: the prose fires 20 times on the default path
with no flag passed, and the pool matches the bf16 model to 0.01% while the fp8 model is 50% off.

## Throughput — 10 cells, two passes

| cell | published | re-measured | delta |
| :--- | ---: | ---: | ---: |
| 128×1 | 202.8 | 202.9 | **+0.1%** |
| 128×8 | 1,124.4 | 1,195.1 | +6.3% |
| 1,024×16 | 1,544.3 | 1,508.0 | −2.4% |
| **4,096×64** | 1,643.6 | **1,360.0** | **−17.3%** |
| **8,192×32** | 938.6 | **758.4** | **−19.2%** |
| **8,192×64** | 989.5 | **870.0** | **−12.1%** |
| 16,000×32 | 444.0 | 432.6 | −2.6% |
| 16,000×64 | 469.2 | 446.0 | −4.9% |
| 16,000×112 | 473.9 | 465.5 | −1.8% |
| 32,000×16 | 249.1 | 242.6 | −2.6% |
| 32,000×32 | 232.8 | 229.0 | −1.6% |

**Seven of eleven agree within 6.3%. Three do not, and they are contiguous** — every cell in the
4,096–8,192 context band at concurrency 32–64, each 12–19% low, with median TPOT 26–44% *higher*
than published (4,096×64: 29.79 → 42.95 ms).

Two candidate causes, neither isolated by this run:

1. **Seed coupling in the source sweep.** All 10 cells there ran at `--seed 0` and recorded no
   prefix-cache metrics. Its `8192×32` cell ran directly after `8192×64` at the same seed and
   `input_len` — the identical pattern that run's own REPORT identified for `16000×32`, one cell
   later and not caught. This run used a distinct seed per cell and measured **0.0% prefix hits in
   every cell** (hit tokens 0 against up to 11.6 M queried).
2. **Node-to-node variation.** Different zone, different provisioning model, different physical
   chip. Uncontrolled.

Seed coupling does not explain it alone: `16000×32`, which that REPORT *demonstrated* was ~100%
cache-served, reproduces here at −2.6% on throughput — consistent with its finding that the
contamination moved TTFT, not tokens/sec. Treat the band as the least reproducible region of the
matrix and quote the clean-seed values.

## The TTFT line is the most reproducible result here

| point | predicted by `TTFT = −8542 + 265 × c` | measured | delta | preemptions |
| :--- | ---: | ---: | ---: | ---: |
| c=64 (90% of pool) | 8,444 ms | 8,459 ms | **+0.18%** | **0** |
| c=112 (157% of pool) | 21,183 ms | 21,229 ms | **+0.22%** | **0** |

A line fitted on one node predicts a different node to within a quarter of a percent, across the
pool boundary, with eviction never engaging. The no-knee finding stands unmodified.

## Files

| | |
| :--- | :--- |
| `validate.py` | 7 anchor cells, distinct seed each, per-cell preemption and prefix-cache deltas |
| `results/allocation.json` | the five allocation checks, all PASS |
| `results/validation.json` | pass 1 — 7 cells |
| `results/validation2.json` | pass 2 — the 4 remaining published cells |
| `logs/boot.log` | boot evidence for the allocation table above |

`verify_allocation.py` is reused unmodified from `../2026-08-10-config-validation-v6e1/`.

## Not established

- **No within-node repetitions.** Each cell is one run on each of two nodes. A sub-6% difference
  between cells is not resolvable.
- **The 4,096–8,192 divergence is not attributed.** Separating seed coupling from node variation
  needs the source sweep re-run with distinct seeds on its own node.
- **Nothing above 157% of pool**, and no cell forced eviction.
- **Cold-boot timing was not instrumented** — the node served ~11 min after ACTIVE, but the phases
  were not separated.
