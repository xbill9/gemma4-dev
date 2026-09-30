# jev-tpu-v5e1

Exploration: the Jev-style label read of Gemma 4 on one TPU v5e chip. Sibling of `../jev-tpu` (unpatched read on one v6e chip) and `../jev-tpu-31b` (the patched W4A16 read on v6e). The read, data, scoring and suite code are copied from `../jev-tpu-31b`, and so are `tpu/serve.sh`, `tpu/run_quant.sh`, `tpu/startup_quant.sh`, `tpu/w4a16_client.py` and `tpu/w4a16_matmul_bench.py`. Runs in: E2B bf16, which served, and the W4A16 arms, none of which served (both 2026-09-27); the 26B W4A16 repack, which serves on one v5e chip with `patches/lowmem.diff` (2026-09-28); and the 12B QAT W4A16 repack, which serves with a smaller patch and scores level with bf16 (2026-09-29). The E2B and E4B QAT repacks (2026-09-29) are level with bf16 too, 1.3–2.4 points over Google's exports. **For one user or a few agents on this chip, use the 12B repack.** See the sections below.

**Where the results live (2026-09-30).** This tree is a repack sweep harness. Each scored arm is filed in the rig named for its chip and exact checkpoint, under `<rig>/benchmarks/runs/`; `results/` keeps a relative symlink at every old arm path, so the scripts and comparison files here still resolve, and `../benchmarks/sweep-moves.json` maps each path. This tree's cells: `../tpu-vllm-v5e1-12b-q4w4a16`, `../tpu-vllm-v5e1-12b-q4w4a16emb4`, `../tpu-vllm-v5e1-12b-w4a16`, `../tpu-vllm-v5e1-12b-w8a8`, `../tpu-vllm-v5e1-12b-w8a8emb4`, `../tpu-vllm-v5e1-26b-q4w4a16`, `../tpu-vllm-v5e1-2b`, `../tpu-vllm-v5e1-2b-fp8`, `../tpu-vllm-v5e1-2b-fp8emb4`, `../tpu-vllm-v5e1-2b-q4_0`, `../tpu-vllm-v5e1-2b-q4w4a16`, `../tpu-vllm-v5e1-2b-q4w4a16emb4`, `../tpu-vllm-v5e1-2b-q4w4a16ple4`, `../tpu-vllm-v5e1-2b-w4a16`, `../tpu-vllm-v5e1-2b-w8a8`, `../tpu-vllm-v5e1-2b-w8a8emb4`, `../tpu-vllm-v5e1-2b-w8a8rtn`, `../tpu-vllm-v5e1-4b-fp8`, `../tpu-vllm-v5e1-4b-q4w4a16`, `../tpu-vllm-v5e1-4b-q4w4a16emb4`, `../tpu-vllm-v5e1-4b-w4a16`, `../tpu-vllm-v5e1-4b-w8a8`, `../tpu-vllm-v5e1-4b-w8a8emb4`. Runs with no suite are filed the same way, by checkpoint. Run-wide logs of runs that span several cells, and the paired comparisons, stay here.

## Status and next steps (2026-09-28)

**Done: `2026-09-27-v5e1-e2btest`**, E2B bf16 only (`jev-arms`), bundle `e2btest-baf42672.tgz` (this tree plus `patches/` from `../jev-tpu-31b`'s `tp4fix3-54532225.tgz`). Boot 360 s. Paired with `../jev-tpu-31b/results/2026-09-25-followup-e2b-bf16` in `results/2026-09-27-v5e1-e2btest-QUANT.md`: suite 0.685 on both (−0.001, 95% −0.004 to +0.003), every task within a point. `quant_compare.py` heads the columns "bf16" / "4-bit"; there "bf16" is v6e and "4-bit" is v5e. `score.py` writes an empty summary for this run because it only scores autoregressive-plus-diffusion pairs.

Answered from the boot log: **the KV cache is bf16 on v5e** (`kv_cache_dtype=auto`, 323,456 tokens in 5.55 GiB, 18.0 KiB/token). The v6e E2B run is also bf16 (1,153,792 tokens in 19.81 GiB), so for E2B the pairing changes only the chip; "v6e picks fp8 by itself" below does not hold for E2B. Weights resident 8.94 GiB. Concurrency-1 latency 12.8 ms median (v6e: 8.7).

Fixed after that run: the self-delete called `gcloud compute tpus queued-resources`, which the TPU VM's gcloud has only under `alpha`, so the node stayed up and was deleted by hand. `run_quant.sh` now calls `gcloud alpha`.

**Done, no arm served: `2026-09-27-v5e1-w4a16`**, bundle `w4a16-352c9e6a.tgz`, `jev-bench=1`, arms `e2b-w4a16`, `e4b-w4a16`, `12b-w4a16` (override). The QR ran 20:51–22:06 UTC and deleted itself (the `gcloud alpha` fix works). All three arms loaded their weights, sized a KV pool, and then failed during warm-up compilation, so there are no records and nothing for `quant_compare.py`. Logs are in `results/2026-09-27-v5e1-w4a16-logs/`.

| Arm | Weights resident | KV pool | Failure |
|---|---|---|---|
| E2B W4A16 | 7.17 GiB | 426,112 tokens | HBM: program needs 1.15G, 1.14G free |
| E4B W4A16 | 10.15 GiB | 81,152 tokens | HBM: program needs 1.77G, 1.22G free |
| 12B W4A16 | 9.46 GiB | 15,616 tokens (7.62x at 2,048) | VMEM: 117.62M scoped against 115.20M, `gmm_v2` tile `tm=128, tk=15360, tn=3840` |

- **E2B and E4B: the KV pool leaves too little HBM for the W4A16 program.** vLLM sizes the pool to the 14.49 GiB cap before warm-up, and the W4A16 `jit_run_model_impl` then needs more than what is left. E2B bf16 booted under the same flags, so the larger program is the W4A16 path's. The untested fix is to shrink the pool: pass `--gpu-memory-utilization` below the default, or `--kv-cache-memory-bytes`, through `run_quant.sh`'s extra args.
- **12B: the `wna16` kernel's tile exceeds v5e's scoped VMEM**, the same failure class as the 31B `down` bench case, here on a 12B shape at 128 tokens. A smaller HBM pool does not help this one; it needs a smaller `tk`/`tn` for v5e in the kernel's tile choice, or `--max-num-batched-tokens` low enough that no 128-token step compiles, which is unverified.

The matmul bench (`logs/matmul-bench.txt`), from the same run: on the 31B shapes that ran (`gate_up` at 1, 16, 64 tokens; `down` at 1, 16) the kernel matched XLA (relative error ≤ 0.0024, no non-finite values) and ran 1.32–1.44x faster than bf16 and about 6x faster than the XLA fallback. 31B `down` at 64 tokens failed to compile: 115.92M scoped VMEM against 115.20M, tile `tk=21504, tn=2688`.

Next:

1. E2B and E4B: relaunch with a smaller KV pool. It needs at least 0.01 GiB freed for E2B and 0.55 GiB for E4B; both leave room for well over 16 × 2,048 tokens.
2. 12B: needs a v5e tile for `gmm_v2` at `k=15360, n=3840`; a flag alone will not fix it.
3. Once an arm serves: `python3 quant_compare.py --prefix <run> --pair e2b-w4a16=results/2026-09-27-v5e1-e2btest-e2b-bf16 --solo e4b-w4a16 --solo 12b-w4a16`. The same arms on v6e are in `../jev-tpu-31b/results/2026-09-25-w4a16-*` for a chip-only pairing.

Rebuilding a bundle: tar this tree without `tests/`, `results/`, `__pycache__` or this README, add `patches/` from `gs://aisprint-491218-bucket/jev-tpu-31b/inputs/tp4fix3-54532225.tgz`, name it by content hash, upload to `gs://aisprint-491218-bucket/jev-tpu-v5e1/inputs/`. `tests/` stays out because `run_quant.sh` runs pytest from a `/tests` mount where the scoring tests cannot import `score.py`.

## Every QAT build of E2B, E4B and 12B on one v5e chip (2026-09-29)

Suite (3,880 records) paired record for record against bf16 (E2B on v5e; E4B and 12B on v6e,
where bf16 fits); output tok/s at 1 / 4 / 16 requests, same flags throughout. Runs:
`results/2026-09-29-{e2c,e4a,e4b-v5e1b,12b,speed,mtp*}-v5e1*` and `results/2026-09-30-{12bgap,e4bemb4}-v5e1*`;
paired comparisons for every arm in `results/2026-09-29-{e2c,e4a,e4b-v5e1b,12b}-v5e1*-QUANT.md`.

| Model | Build | Weights on chip | Suite (vs bf16, 95% range) | tok/s |
|---|---|---:|---|---|
| 12B | emb4 (int4 linears and vocabulary tables) | 6.86 GiB | 0.762 (+0.2, −0.4 to +0.8) | 35 / 127 / 407 |
| 12B | **W8A8 + int4 embeddings** | 11.31 GiB | **0.761 (+0.1, −0.6 to +0.8)** | **57 / 219 / 675** |
| 12B | W8A8 (bf16 tables) | 12.03 GiB | 0.757 (−0.2, −1.0 to +0.5) | 53 / 201 / 624 |
| E4B | W8A8 | — | 0.727 (−0.4, −1.1 to +0.3) | 119 / 459 / 1,590 |
| E4B | fp8 | — | 0.733 (+0.2, −0.4 to +1.0) | 99 / 382 / 1,349 |
| E4B | **W8A8 + int4 embeddings** | — | **0.728 (−0.3, −1.0 to +0.5)** | **133 / 509 / 1,747** |
| E4B | emb4 | — | 0.730 (−0.1, −0.7 to +0.6) | 79 / 308 / 1,064 |
| E2B | QAT bf16 (`-qat-q4_0-unquantized`) | — | 0.681 (−0.2) | 144 / 560 / 2,007 |
| E2B | ple4 (int4 per-layer table only) | — | 0.680 (−0.2) | 137 / 533 / 1,910 |
| E2B | emb4 | 2.66 GiB | 0.681 (−0.2) | 150 / 583 / 2,067 |
| E2B | **W8A8 + int4 embeddings** | 3.43 GiB | 0.677 (−0.6, −1.5 to +0.4) | **243 / 923 / 3,086** |
| E2B | fp8 | — | **0.670 (−1.3, −2.2 to −0.3)** | 182 / 703 / 2,453 |
| E2B | fp8 + int4 embeddings | — | 0.677 (−0.6) | 203 / 781 / 2,644 |

- **W8A8 linears with int4 vocabulary tables is the best build at every size**: the fastest, level
  with bf16 on the suite, and half the size of W8A8 with bf16 tables. At 12B it is the biggest
  model this chip serves: 11.31 GiB of weights, compiled with 2.08 GiB free, ready in 781 s. Both
  12B builds ran with the 40-block KV cap of the earlier 12B run.
- **fp8 is storage-only on v5e** and is the one build with a measurable accuracy loss (E2B).
- **The int4 tables must be 128 columns wide.** Gathering more than one row of an int32 table whose
  rows are not a multiple of 128 reads the whole table: 4.2 ms per step for the E2B per-layer table,
  flat in the number of ids, against 0.06 ms padded (`tpu/lookup_bench.py`). Unpadded, E2B emb4
  decoded at 85 tok/s and E4B emb4 ran out of HBM compiling.
- **int4 tables are worth 8% of throughput at 12B** (W8A8: 57 / 219 / 675 against 53 / 201 / 624 with
  bf16 tables) and 0.72 GiB of weights.
- **The int4 `lm_head` runs through gmm_v2 at 1.7-3.2x the bf16 head** (`tpu/lm_head_bench.py`).

### 12B context on one v5e chip (2026-09-30)

12B KV costs 336 KiB per token: the boot logs bound it at 333–338 (5.11 GiB held 62 256-token blocks,
4.48 GiB held 54), which is the config geometry with a V cache stored for the eight `attention_k_eq_v` layers. Every 12B run
before this held 5,120 KV tokens (`--num-gpu-blocks-override 40`). Runs `results/2026-09-30-12bctx{1..7}-v5e1*`,
`--max-model-len 8192` (vLLM then uses 256-token blocks):

| Build | Setting | KV tokens | tok/s at 1 / 4 / 16 | Ready after |
|---|---|---:|---|---:|
| **W8A8 + int4 embeddings** | `--gpu-memory-utilization 0.92`, no block override | **9,728** | **57 / 217 / 723** | 421 s |
| emb4 | `--gpu-memory-utilization 0.72`, no block override | **13,824** | 35 / 105 / 433 | 2,507 s |

- **W8A8 + int4 tables is bound by its weights** (11.31 GiB): at 0.92, vLLM's own 38 blocks leave
  enough outside the cap for its compiled program.
- **emb4 is bound by its compiled program**, which needs a 3.92 GB allocation on top of weights and KV,
  the same figure as the 12B W4A16 repack. At 0.76 (15,872 tokens) it found 3.73 GB free and failed;
  0.72 leaves 0.46 GiB to spare.
- **fp8 KV cache is untested here without a block override.** Both fp8 runs (`12bctx2`, `12bctx4`)
  set `--num-gpu-blocks-override` to 128 and 176 blocks of 16 tokens, below the 16,384-token
  `--max-model-len` they started with, so vLLM refused them (2.75 GiB needed against 0.34 and 0.47 GiB).
  Before the override it had sized 1,185 and 2,840 blocks.

### Speculative decoding (Gemma 4 MTP, 4 draft tokens), E2B

| Target + drafter | 1 req | 4 req | 16 req | Mean acceptance length |
|---|---:|---:|---:|---:|
| W8A8, no drafter | 220 | 841 | 2,872 | — |
| **W8A8 + `google/gemma-4-E2B-it-assistant`** | **395** | 887 | 1,614 | 3.21 |
| bf16 + `E2B-it-assistant` | 346 | 800 | 1,345 | — |
| W4A16 repack + `E2B-it-assistant` | 282 | — | 1,275 | 2.49 |
| bf16 + `-qat-q4_0-unquantized-assistant` | 81 | 320 | 1,149 | 1.97 |
| W8A8 + `-qat-q4_0-unquantized-assistant` | 99 | 390 | 1,343 | 1.82 |

The drafter gains at one request and loses at 16, where the chip is no longer waiting on memory.
The QAT assistant is slower than no drafter on every target. The drafter copies the target's bf16
`embed_tokens`, so it cannot pair with the int4-table builds, and it needs the `gemma4_mtp.py`
change in `lowmem.diff` to load unquantized beside a quantized target. Load prompts run with
`ignore_eos`, which may favour acceptance.

## 12B QAT W4A16 repack on one v5e chip: the model to use here (2026-09-29)

**`2026-09-28-12b3-v5e1`** served [`xbill9/gemma-4-12B-it-qat-q4_0-w4a16-ct`](https://huggingface.co/xbill9/gemma-4-12B-it-qat-q4_0-w4a16-ct), the 12B repacked from `google/gemma-4-12B-it-qat-q4_0-unquantized` with `../jev-tpu-31b/repack_q4_0.py` (340,623,360 groups, none off the source grid; card and upload script in `../jev-tpu-31b/hf/12b/`), and `google/gemma-4-12B-it-qat-w4a16-ct` after it on the same chip with the same flags. Suite, paired record for record (`results/2026-09-28-12b3-v5e1-VS-*.md`):

| repack on v5e-1 against | reference | repack | difference | 95% range |
|---|---:|---:|---:|---|
| Google's `-qat-w4a16-ct`, **same v5e chip** | 0.752 | 0.758 | +0.006 | +0.000 to +0.012 |
| Google's `-qat-w4a16-ct`, v6e-1 | 0.751 | 0.758 | +0.007 | +0.001 to +0.013 |
| `-qat-q4_0-unquantized` (its source, bf16), v6e-1 | 0.757 | 0.758 | +0.001 | −0.002 to +0.004 |
| `google/gemma-4-12B-it` bf16, v6e-1 | 0.760 | 0.758 | −0.001 | −0.007 to +0.005 |

The 26B on this chip scores 0.754 (section below).

| one v5e chip | 12B repack | 12B Google | 26B repack (`lowmem.diff`) |
|---|---:|---:|---:|
| weights on the chip | 7.59 GiB | 9.46 GiB | 13.58 GiB |
| compiled model | 3.98 GiB | — | 1.26 GiB |
| KV room left (arithmetic from the boot logs) | ~3.9 GiB | ~2.0 GiB | ~0.65 GiB |
| first-token latency, concurrency 1, median | 62.9 ms | 63.1 ms | 68.2 ms |
| output tok/s at concurrency 1 / 4 / 16 | 33.0 / 127.4 / 387.9 | — | — |
| boot, first / with the compile cache | 2,852 s / 436 s | 436 s (cache from the repack) | 1,806 s / — |

Settings (`jev-env`, `jev-serve-args`): `GMM_V2_TILE_VMEM_FRACTION=0.85` (in `patches/lowmem.diff`; without it the 12B's gmm_v2 tile exceeds v5e's scoped VMEM), `MIN_TOKEN_BUCKET=64` and `--max-num-batched-tokens 512` (four backbone buckets, 64–512, instead of eight), `--gpu-memory-utilization 0.85 --num-gpu-blocks-override 40` (5,120 tokens), and `jev-swap-gb=40`, `jev-boot-timeout=7200`, `--hf-overrides` to `Gemma4ForCausalLM` (arm flag `override`). The compile cache is at `gs://aisprint-491218-bucket/jev-tpu-v5e1/xla-cache/v5e-12b-repack` (`jev-xla-seed=v5e-12b-repack`).

What set those, from the two earlier runs:

- **The compiled model is 3.92 GiB of "overlays"** for the dense 12B (1.15 for the 26B). With the KV cache filled to the 0.85 cap (5.78 GiB, 18,048 tokens) the compile needed 17.60 GiB (`2026-09-28-12b-v5e1`), hence the block cap. 40 blocks is what the suite and the load test need; about 90 would fit.
- **Compiling eight buckets did not finish in an hour**: host memory grew about 7 GB per 5 minutes, filled RAM and 21 GB of swap (`2026-09-28-12b2-v5e1`). Four buckets peaked at about 40 GB plus 9 GB of swap.

## E2B int8 W8A8 on one v5e chip: from the QAT weights it is the best E2B build (2026-09-29)

int8 W8A8 (int8 weights with a per-channel scale, activations quantized to int8 per token) runs v5e's native int8 x int8 -> int32 path. `patches/lowmem.diff` adds it to the JAX compressed-tensors path, which only knew fp8 W8A8. Two checkpoints, both text-only through `Gemma4ForCausalLM`, same flags as the bf16 baseline (`GMM_V2_TILE_VMEM_FRACTION=0.85`, `MIN_TOKEN_BUCKET=64`, `--max-num-batched-tokens 512`, `--gpu-memory-utilization 0.80`):

- `glenic/gemma-4-E2B-it-W8A8-INT8`: llm-compressor round-to-nearest from the original bf16 model.
- `gemma-4-E2B-it-qat-w8a8-int8` (local, bucket `jev-tpu-31b/models/`): made by `../jev-tpu-31b/w8a8_from_qat.py` from `google/gemma-4-E2B-it-qat-q4_0-unquantized`; the int8 values sit within 0.6-1.7% (mean ~0.8%) of the QAT values.

| E2B, one v5e chip | suite | tok/s at 1 / 4 / 16 | first-token latency |
|---|---:|---|---:|
| bf16 (`2026-09-29-w8a8-v5e1`) | 0.683 | 144 / 560 / 2,008 | 12.6 ms |
| W4A16 QAT repack (`2026-09-29-e2br-v5e1`) | 0.678 | 136 / 532 / 1,906 | 16.2 ms |
| W8A8, glenic (`2026-09-29-w8a8b-v5e1`) | 0.670 | 220 / 842 / 2,876 | 10.7 ms |
| **W8A8 from QAT** (`2026-09-29-qatw8-v5e1`) | **0.686** | **220 / 841 / 2,872** | **10.4 ms** |

Paired record for record (`results/2026-09-29-qatw8-v5e1-VS-*.md`), W8A8 from QAT against bf16 +0.003 (−0.007 to +0.013), against glenic's W8A8 +0.015 (+0.005 to +0.026), against the W4A16 repack +0.007 (+0.000 to +0.015). glenic's W8A8 against bf16: −0.012 (−0.020 to −0.004). So the int8 activations cost nothing measurable here; glenic's loss is its rounding of the original bf16 weights. On E2B the W4A16 repack is slightly slower than bf16: its 4-bit linears are a small part of a model whose embeddings stay bf16, and the unpacking costs more than the bytes it saves.

## E2B and E4B QAT W4A16 repacks on one v5e chip (2026-09-29)

[`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct) and [`xbill9/gemma-4-E4B-it-qat-q4_0-w4a16-ct`](https://huggingface.co/xbill9/gemma-4-E4B-it-qat-q4_0-w4a16-ct), repacked from the `-qat-q4_0-unquantized` exports with `../jev-tpu-31b/repack_q4_0.py`, which now also quantizes the per-layer-embedding projections (E2B 58,650,624 groups, E4B 124,149,760, none off the source grid). Each served alone (`2026-09-29-e2br-v5e1`, `-e4br-v5e1`); Google's `-qat-w4a16-ct` served in `2026-09-29-e2b-v5e1` and `-e4b-v5e1` with the same flags. Suite, paired record for record (`results/2026-09-29-e*r-v5e1-VS-*.md`):

| | bf16 | Google `-w4a16-ct`, v5e | repack, v5e | repack − bf16 | repack − Google |
|---|---:|---:|---:|---|---|
| E2B | 0.685 (same v5e chip) | 0.655 | **0.678** | −0.006 (−0.016 to +0.003) | **+0.024 (+0.014 to +0.034)** |
| E4B | 0.731 (v6e; does not fit v5e) | 0.716 | **0.729** | −0.002 (−0.008 to +0.005) | **+0.013 (+0.007 to +0.020)** |

| one v5e chip | E2B repack | E2B Google | E4B repack | E4B Google |
|---|---:|---:|---:|---:|
| weights resident | 6.42 GiB | 7.17 GiB | 8.90 GiB | 10.15 GiB |
| first-token latency, median | 16.2 ms | 16.0 ms | 30.9 ms | 31.4 ms |
| output tok/s at 1 / 4 / 16 | 136 / 532 / 1,906 | — | 75 / 291 / 1,012 | — |
| boot, uncached | 872 s | 917 s | 1,187 s | 1,218 s |

Google's E2B and E4B exports also store a bf16 `lm_head.weight` byte-identical to `embed_tokens` although the config ties them: 0.75 and 1.25 GiB, which vLLM loads, and which is exactly the resident difference. Settings: as the 12B (`GMM_V2_TILE_VMEM_FRACTION=0.85`, `MIN_TOKEN_BUCKET=64`, `--max-num-batched-tokens 512`, 40 GB swap) but `--gpu-memory-utilization 0.80` and no block cap; that is what the 2026-09-27 E2B and E4B W4A16 arms lacked. The first repacks (`2026-09-29-e2b-v5e1`, `-e4b-v5e1`, repack arms) failed to load because the PLE projections were bf16 but not on the ignore list.

## 26B W4A16 on one v5e chip: serves (2026-09-28)

**`2026-09-28-lowmem7-v5e1`** served [`xbill9/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct`](https://huggingface.co/xbill9/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct) on one v5e chip and ran the full read. Paired record for record with the same checkpoint on one v6e chip (`../jev-tpu-31b/results/2026-09-26-moe2-26b-q4w4`) in `results/2026-09-28-lowmem7-v5e1-QUANT.md` (its "bf16" column is the v6e run, "4-bit" the v5e one):

| group | n | v6e-1 | v5e-1 | difference | 95% range |
|---|---:|---:|---:|---:|---|
| sst2 | 300 | 0.950 | 0.950 | +0.000 | +0.000 to +0.000 |
| ag_news | 300 | 0.860 | 0.863 | +0.003 | −0.007 to +0.013 |
| emotion | 300 | 0.600 | 0.603 | +0.003 | −0.007 to +0.017 |
| irony | 300 | 0.907 | 0.903 | −0.003 | −0.010 to +0.000 |
| reversed options | 900 | 0.798 | 0.797 | −0.001 | −0.004 to +0.002 |
| suite | 3,880 | 0.753 | 0.754 | +0.001 | −0.004 to +0.005 |

Concurrency-1 latency is 68.2 ms median against 27.1 ms on v6e. Boot 1,806 s, 30 minutes, most of it compiling.

**On the chip** (15.75 GiB total): 13.58 GiB of weights, a 17-block KV cache (2,176 tokens, 0.46 GiB), and a compiled model of about 1.26 GiB (1.15 GiB of it "overlays"). **On the host** (48 GB): compiling the eight backbone buckets peaked at 45.7 GB used plus 2.75 GB of swap.

Serve settings, in addition to the siblings' flags and `--max-model-len 2048` (the suite's longest record is about 1,930 tokens):

| setting | what it does | measured effect |
|---|---|---|
| `--hf-overrides` to `Gemma4ForCausalLM` (arm flag `override`) | text-only class, no vision tower | −1.07 GiB |
| `W4A16_MOE_NO_PAD=1` | experts' intermediate dim stays 704 (not 768); gmm1 runs unfused and the activation runs in JAX with gate/up in f32 | −0.81 GiB (16.36 → 15.55) |
| `W4A16_MOE_BF16_SCALES=1` | expert scales stored bf16 as `[E, 1, groups, N]`, widened to f32 per layer in the forward | −1.33 GiB |
| `EMBED_INT8_GROUP=32` | tied embedding stored int8 with a bf16 scale per 32 columns; the LM head dequantizes 8,192 rows at a time | −0.64 GiB |
| `SKIP_IDENTITY_MODEL_JIT=1` | skips the post-load `create_jit_model` pass, an identity without Qwix | avoids a 16.00 GiB temporary |
| `--gpu-memory-utilization 0.97 --num-gpu-blocks-override 17` | the smallest KV cache that holds 2,048 tokens | |
| metadata `jev-swap-gb=24` | a swap file on the host | keeps compilation from being OOM-killed |

All four environment switches are in `patches/lowmem.diff` (branch `gemma4-w4a16-moe-lowmem` in `~/tpu-inference`, on top of `moe.diff`), each off unless set. Patch list: `kvshare.diff wna16.diff moe.diff lowmem.diff`.

What each attempt found, all in the logs under `gs://aisprint-491218-bucket/jev-tpu-v5e1/2026-09-28-lowmem*`:

- **gmm_v2 cannot read bf16 scales on v5e**: `Strided load with non 32-bit data` (Mosaic), from the zero-stride sublane broadcast of the scale. `tokamax-bf16-scale.diff` is kept in `patches/` but not applied; scales are widened per call instead, which costs 190–235 µs per MoE layer on v5e (`tpu/moe_lowmem_probe.py`: identical error to the served layout, worst 2.6e-3 against an XLA reference).
- **A bf16 scale shaped `[E, groups, 1, N]` takes as much HBM as f32**: the size-1 dim second from minor is padded to bf16's two-row sublane packing. 15.55 GiB resident with the switch on, the f32 figure. Moving the axis gave the saving.
- **`create_jit_model` asked for 16.00 GiB of HLO temporaries** after the weights loaded; skipping it changes nothing without Qwix.
- **With weights at 14.22 GiB (bf16 embeddings) the compile was 199 MB over**: 15.94 of 15.75 GiB used.
- **The host ran out of memory compiling**: without swap the engine was SIGKILLed 17 minutes into the backbone compiles.

The int8 embedding is the one change that alters the served weights; the paired read above is its check. The v6e check of these switches (`2026-09-28-lowmem*-v6e1`) never got flex-start capacity in `europe-west4-a`.

## What differs from the v6e trees

**Provisioning.** v5e has no Compute Engine path: `instances create` refuses `ct5lp-hightpu-1t` (`../HARDWARE.md`, "Can v5e use the Compute Engine path?"). The `gcloud compute instances create` launches in `../jev-tpu` and `../jev-tpu-31b` therefore have no v5e equivalent. This tree runs on a Cloud TPU queued resource, and flex-start `v5litepod-1` is accepted only in `us-west4-a` (`../CLAUDE.md`, Cloud gotchas). Because the node is a queued resource, `run_quant.sh` deletes itself through metadata `jev-qr` rather than `compute instances delete "$(hostname)"`. A TPU VM's hostname is not its node name. Without `jev-qr` the node is left up.

**Memory.** 14.49 GiB usable against about 28.74 on v6e-1 (`../HARDWARE.md` §v5e-1). From `../MODELS.md` §Weight footprints:

| Checkpoint | Weights | Fits one v5e chip |
|---|---|---|
| `google/gemma-4-E2B-it` bf16 | 9.5 GiB (8.97 resident, measured) | yes |
| `google/gemma-4-E2B-it-qat-w4a16-ct` | 8.3 GB (PLE stays bf16) | yes |
| `google/gemma-4-E4B-it` bf16 | 14.9 GiB | no |
| `google/gemma-4-E4B-it-qat-w4a16-ct` | not measured | expected yes |
| `google/gemma-4-12B-it` bf16 | 22.4 GiB | no |
| `google/gemma-4-12B-it-qat-w4a16-ct` | 9.46 GiB resident, measured | yes (above) |
| `xbill9/gemma-4-12B-it-qat-q4_0-w4a16-ct` (repack) | 7.59 GiB resident, measured | **yes, recommended** (above) |
| 26B A4B W4A16 (repacked, `xbill9/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct`) | 15.29 GiB on disk, 17.43 resident on v6e | **yes, with `lowmem.diff`**: 13.58 GiB resident, measured (below) |
| 31B, any build | ≥ 14.4 GiB before anything else | no |

The default arms in `run_quant.sh` are therefore E2B bf16, E2B W4A16, E4B W4A16, and 12B W4A16 served with `--hf-overrides` to `Gemma4ForCausalLM` (the route `../jev-tpu-31b` found needs no `unified.diff`). The default patch list is `kvshare.diff wna16.diff`. As in `../jev-tpu-31b`, the diffs are not in the tree: they come from the upstream PR branches (vllm-project/tpu-inference#3299, #3653) and go into the code bundle under `patches/`.

## Open items

- **`wna16.diff` on v5e.** Correct and faster than bf16 on the 31B bench shapes that compiled. The kernel's tile choice exceeds v5e's 115.20M scoped VMEM on 31B `down` at 64 tokens and on a 12B serving shape at 128 tokens (Status above).
- **W4A16 HBM headroom.** E2B and E4B W4A16 fail warm-up because the default KV pool leaves 1.14–1.22G free. Whether a smaller pool is enough is untested.
- **KV cache dtype.** Answered for E2B bf16: `auto` resolves to bf16 on v5e (Status above).
- **12B W4A16 KV pool.** Answered: 15,616 tokens with 9.46 GiB resident, 7.62x concurrency at 2,048 tokens, before the warm-up failure.

## Pairing with the v6e results

The image digest (`vllm/vllm-tpu@sha256:19a1a052…`), serve flags and records match `../jev-tpu-31b`, so each arm here pairs record for record with the same checkpoint there (`quant_compare.py`). That pairing changes only the chip, and is a hardware comparison only if the KV cache dtype matches (see above). E4B and 12B have no bf16 arm on v5e, so their W4A16 penalty can only be read against the v6e bf16 runs, which is a cross-chip difference.

## Launch

Write `PREREGISTRATION.md` first, as in the siblings. Then:

```shell
RUN=<yyyy-mm-dd>-v5e1
gcloud alpha compute tpus queued-resources create jev-tpu-v5e1-$RUN --zone us-west4-a \
  --accelerator-type v5litepod-1 --runtime-version v2-alpha-tpuv5-lite --node-id jev-tpu-v5e1-$RUN-node \
  --provisioning-model flex-start --max-run-duration 4h --valid-until-duration 2h \
  --metadata jev-code=<code-tarball>,jev-run=$RUN,jev-qr=jev-tpu-v5e1-$RUN,jev-bench=1 \
  --metadata-from-file startup-script=tpu/startup_quant.sh
```

The code tarball goes to `gs://aisprint-491218-bucket/jev-tpu-v5e1/inputs/`. Results and logs land under `gs://aisprint-491218-bucket/jev-tpu-v5e1/<run>/`. The suite is read from `../jev-tpu`'s shared `jev-tpu/inputs/suite.tgz`. The flags follow `../tpu-vllm-v5e1-2b/server.py`'s `_create_queued_resource`, except for `--metadata`, which that rig does not pass. Every flag above appears in `gcloud alpha compute tpus queued-resources create --help` (checked 2026-09-27), but no launch has used this command yet.
