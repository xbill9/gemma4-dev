# 2026-10-02-suite-tp4-v5e4

Accuracy of `xbill9/gemma-4-E2B-it-qat-w8a8-int8` served at `--tensor-parallel-size 4` on one v5e-4 (`v5litepod-4`, flex-start, `us-west4-a`), paired record for record with the same checkpoint at TP=1 on one v5e chip. Run by `../../../../jev-tpu-v5e1/tpu/run_quant.sh` with metadata `jev-tp=4` on Queued Resource `jev-v5e4-suite`, which deleted itself when the run finished.

**The suite is level at TP=4** (−0.4 points, −1.2 to +0.5). GSM8K and BFCL come out 1.4 and 1.2 points higher than at TP=1; the 12B moves the other way on the same tasks (−0.9 and −0.8), so splitting the matmuls four ways changes which long greedy answers come out right without a direction. No repeat of a TP=1 generation run exists to size that spread at one chip. With the sweep's serve flags (`MIN_TOKEN_BUCKET=64`, `--max-num-batched-tokens 512`) E2B at TP=4 runs at one chip's speed at 16 requests (1.00x); the rig's own boot, with its default flags, measured 0.51x.

#### Settings

Everything but the chip count matches the v5e-1 arms: image `vllm/vllm-tpu@sha256:19a1a052…` patched with `kvshare.diff wna16.diff moe.diff lowmem.diff textonly.diff` (lowmem.diff `27e40c1b`; the v5e-1 suite arms used earlier versions of it whose differences touch only the int4 embedding method's decode path and the MTP drafter), `GMM_V2_TILE_VMEM_FRACTION=0.85 MIN_TOKEN_BUCKET=64`, `--max-num-batched-tokens 512`, the checkpoint copied from `gs://aisprint-491218-bucket/jev-tpu-31b/models/gemma-4-E2B-it-qat-w8a8-int8` (the same files the v5e-1 arms served). Suite arm `e2b-qat-w8a8`: `--gpu-memory-utilization 0.80`, `--max-model-len 2048`. GSM8K/BFCL arm `e2b-qat-w8a8-tools`: `--gpu-memory-utilization 0.80 --max-model-len 4096`, `--enable-auto-tool-choice --tool-call-parser gemma4`, GSM8K answer limit 2,048 tokens. Code bundle `suite-tp4-e2a8358a.tgz`.

#### Suite and the four tasks against TP=1 on v5e-1

`quant_compare.py`, 2,000 bootstrap resamples. Reference: `2026-09-29-qatw8-v5e1-e2b-qat-w8a8` (`../../../../tpu-vllm-v5e1-2b-w8a8/benchmarks/runs/2026-09-29-qatw8-e2b-qat-w8a8-v5e1/`).

| group | n | v5e-1, TP=1 | v5e-4, TP=4 | difference (points) | 95% range | v5e-1 right only | v5e-4 right only |
|---|---:|---:|---:|---:|---|---:|---:|
| sst2 | 300 | 0.880 | 0.893 | +1.3 | +0.0 to +3.0 | 1 | 5 |
| ag_news | 300 | 0.557 | 0.563 | +0.7 | -3.0 to +4.3 | 14 | 16 |
| emotion | 300 | 0.450 | 0.460 | +1.0 | -2.0 to +4.0 | 10 | 13 |
| irony | 300 | 0.803 | 0.793 | -1.0 | -3.7 to +1.7 | 10 | 7 |
| reversed options (3 choice tasks) | 900 | 0.606 | 0.603 | -0.2 | -1.6 to +1.2 | 24 | 22 |
| suite (all subsets) | 3,880 | 0.686 | 0.682 | -0.4 | -1.2 to +0.5 | 132 | 118 |

#### GSM8K and BFCL simple against TP=1 on v5e-1

`gen_compare.py`, 10,000 bootstrap resamples. Reference: `2026-09-30-gen2048a-v5e1-e2b-qat-w8a8-gen` (`../../../../tpu-vllm-v5e1-2b-w8a8/benchmarks/runs/2026-09-30-gen2048a-e2b-qat-w8a8-v5e1/`).

| task | n | v5e-1, TP=1 | v5e-4, TP=4 | difference (points) | 95% range | v5e-1 right only | v5e-4 right only |
|---|---:|---:|---:|---:|---|---:|---:|
| gsm8k | 1,319 | 0.889 | 0.903 | +1.4 | +0.2 to +2.5 | 22 | 40 |
| bfcl_simple | 400 | 0.915 | 0.927 | +1.2 | +0.0 to +2.5 | 1 | 6 |

#### Throughput of the suite arm

`../../../../jev-tpu-v5e1/tpu/w4a16_client.py load` on the VM, output tok/s, median of three passes. Same flags as the v5e-1 suite arm, whose load JSONs are in its logs.

| Requests | v5e-1, TP=1 | v5e-4, TP=4 | v5e-4 / v5e-1 |
|---:|---:|---:|---:|
| 1 | 219.6 | 198.8 | 0.91x |
| 4 | 841.2 | 776.2 | 0.92x |
| 16 | 2,872.0 | 2,866.9 | 1.00x |

#### Files

| File | What |
|---|---|
| `2026-10-02-suite-tp4-v5e4-e2b-qat-w8a8{,-latency,-smoke}/` | the four tasks (both option orders), latency, smoke records |
| `2026-10-02-suite-tp4-v5e4-e2b-qat-w8a8-suite/` | the 3,880 suite records |
| `2026-10-02-suite-tp4-v5e4-e2b-qat-w8a8-tools-gen/` | GSM8K and BFCL simple records |
| `logs/2026-10-02-suite-tp4-v5e4-logs/` | this rig's arms' boot logs, host memory, load JSONs, gen summaries, vLLM version |
| `../../../../jev-tpu-v5e1/results/2026-10-02-suite-tp4-v5e4-logs/` | run-wide logs: `run.log`, patch logs, suite checksums |
| `../../../../jev-tpu-v5e1/results/2026-10-02-suite-tp4-v5e4-QUANT.md` | suite pairing for both rigs (its columns are headed "bf16" for v5e-1 and "4-bit" for v5e-4) |
| `../../../../jev-tpu-v5e1/results/2026-10-02-suite-tp4-v5e4-GEN-VS-V5E1.md` | GSM8K/BFCL pairing for both rigs |
