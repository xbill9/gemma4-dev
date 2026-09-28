# jev-tpu-v5e1

Exploration: the Jev-style label read of Gemma 4 on one TPU v5e chip. Sibling of `../jev-tpu` (unpatched read on one v6e chip) and `../jev-tpu-31b` (the patched W4A16 read on v6e). The read, data, scoring and suite code are copied from `../jev-tpu-31b`, and so are `tpu/serve.sh`, `tpu/run_quant.sh`, `tpu/startup_quant.sh`, `tpu/w4a16_client.py` and `tpu/w4a16_matmul_bench.py`. Two runs are in: E2B bf16, which served, and the W4A16 arms, none of which served (both 2026-09-27); see Status below.

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
| `google/gemma-4-12B-it-qat-w4a16-ct` | not measured; ~5.6 GiB body plus bf16 embeddings | expected yes, with a small KV pool |
| 26B A4B W4A16 (repacked, `xbill9/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct`) | 15.29 GiB on disk, 17.43 resident on v6e | no as it loads today (0.80 GiB over on disk); 13.96 GiB after three unbuilt loader changes, leaving room for about one request (`../MODELS.md`, "The 26B W4A16 repack on a v5e-1") |
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
