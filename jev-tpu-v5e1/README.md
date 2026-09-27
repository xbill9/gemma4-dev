# jev-tpu-v5e1

Exploration: the Jev-style label read of Gemma 4 on one TPU v5e chip. Sibling of `../jev-tpu` (unpatched read on one v6e chip) and `../jev-tpu-31b` (the patched W4A16 read on v6e). The read, data, scoring and suite code are copied from `../jev-tpu-31b`, and so are `tpu/serve.sh`, `tpu/run_quant.sh`, `tpu/startup_quant.sh`, `tpu/w4a16_client.py` and `tpu/w4a16_matmul_bench.py`. One test run is in (E2B bf16, 2026-09-27); see Status below.

## Status and next steps (2026-09-27)

**Done: `2026-09-27-v5e1-e2btest`**, E2B bf16 only (`jev-arms`), bundle `e2btest-baf42672.tgz` (this tree plus `patches/` from `../jev-tpu-31b`'s `tp4fix3-54532225.tgz`). Boot 360 s. Paired with `../jev-tpu-31b/results/2026-09-25-followup-e2b-bf16` in `results/2026-09-27-v5e1-e2btest-QUANT.md`: suite 0.685 on both (−0.001, 95% −0.004 to +0.003), every task within a point. `quant_compare.py` heads the columns "bf16" / "4-bit"; there "bf16" is v6e and "4-bit" is v5e. `score.py` writes an empty summary for this run because it only scores autoregressive-plus-diffusion pairs.

Answered from the boot log: **the KV cache is bf16 on v5e** (`kv_cache_dtype=auto`, 323,456 tokens in 5.55 GiB, 18.0 KiB/token). The v6e E2B run is also bf16 (1,153,792 tokens in 19.81 GiB), so for E2B the pairing changes only the chip; "v6e picks fp8 by itself" below does not hold for E2B. Weights resident 8.94 GiB. Concurrency-1 latency 12.8 ms median (v6e: 8.7).

Fixed after that run: the self-delete called `gcloud compute tpus queued-resources`, which the TPU VM's gcloud has only under `alpha`, so the node stayed up and was deleted by hand. `run_quant.sh` now calls `gcloud alpha`.

**Running: `2026-09-27-v5e1-w4a16`**, queued resource `jev-tpu-v5e1-2026-09-27-v5e1-w4a16` in `us-west4-a` (project `aisprint-491218`), launched about 20:55 UTC, bundle `w4a16-352c9e6a.tgz` (has the delete fix), `jev-bench=1`, arms `e2b-w4a16`, `e4b-w4a16`, `12b-w4a16` (override). Flex-start: it lapses if not granted within 2 h and is capped at 4 h once up. It deletes itself at the end; check that it did.

Next, on any machine:

1. Check progress: `gcloud storage cat gs://aisprint-491218-bucket/jev-tpu-v5e1/2026-09-27-v5e1-w4a16/logs/run.log | grep -E 'jev-quant|READY|FAILED'`, and `gcloud alpha compute tpus queued-resources list --project aisprint-491218 --zone us-west4-a`. If it shows `DONE` and the QR is still listed, delete it with `gcloud alpha compute tpus queued-resources delete jev-tpu-v5e1-2026-09-27-v5e1-w4a16 --project aisprint-491218 --zone us-west4-a --force --quiet`. Leave `jax-gemma4-qr` and `torchtpu-v5e1-qr` alone; they belong to other rigs.
2. Read `logs/matmul-bench.txt` first: it settles whether `wna16.diff`'s bf16-activation branch holds on v5e's 128x128 MXUs (open item below).
3. Read the KV pool for 12B W4A16 from `logs/google_gemma-4-12B-it-qat-w4a16-ct.boot.log` (`TPU KV cache size`) — the third open item.
4. Pull results: `gcloud storage rsync -r gs://aisprint-491218-bucket/jev-tpu-v5e1/2026-09-27-v5e1-w4a16/results results`.
5. Compare: `python3 quant_compare.py --prefix 2026-09-27-v5e1-w4a16 --pair e2b-w4a16=results/2026-09-27-v5e1-e2btest-e2b-bf16 --solo e4b-w4a16 --solo 12b-w4a16`. The E2B pair is same-chip; E4B and 12B have no bf16 arm on v5e, so only cross-chip reads against the v6e bf16 runs are possible for them. The same arms on v6e are in `../jev-tpu-31b/results/2026-09-25-w4a16-*` for a chip-only pairing.
6. Commit `results/` and a note here.

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
| 26B A4B W4A16 (repacked) | 15.27 GiB measured | no |
| 31B, any build | ≥ 14.4 GiB before anything else | no |

The default arms in `run_quant.sh` are therefore E2B bf16, E2B W4A16, E4B W4A16, and 12B W4A16 served with `--hf-overrides` to `Gemma4ForCausalLM` (the route `../jev-tpu-31b` found needs no `unified.diff`). The default patch list is `kvshare.diff wna16.diff`. As in `../jev-tpu-31b`, the diffs are not in the tree: they come from the upstream PR branches (vllm-project/tpu-inference#3299, #3653) and go into the code bundle under `patches/`.

## Open before the first run

- **`wna16.diff` has only run on v6e.** Its `gmm_v2` path keeps activations in bf16 when the quantization group is narrower than the matrix unit, which is 256 columns on v6e. v5e's MXUs are 128x128 (`../HARDWARE.md`). A group of 32 is narrower than either, so the same branch should be taken, but that is inferred, not measured. Run `jev-bench=1` (per-layer `w4a16_matmul_bench.py`, kernel checked against XLA) on the first boot.
- **KV cache dtype.** v6e picks an fp8 KV cache on its own. v5e has no native fp8 (`../HARDWARE.md`), and what vLLM picks there is unrecorded. Read it from the first boot log before pairing any arm against a v6e result.
- **12B W4A16 headroom.** The KV pool left over at `--max-model-len 2048 --max-num-seqs 16` is arithmetic until a boot log shows it.

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
