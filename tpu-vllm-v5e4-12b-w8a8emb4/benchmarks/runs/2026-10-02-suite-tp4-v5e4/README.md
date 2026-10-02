# 2026-10-02-suite-tp4-v5e4

Accuracy of `xbill9/gemma-4-12B-it-qat-w8a8-int8-emb4` served at `--tensor-parallel-size 4` on one v5e-4 (`v5litepod-4`, flex-start, `us-west4-a`), paired record for record with the same checkpoint at TP=1 on one v5e chip. Run by `../../../../jev-tpu-v5e1/tpu/run_quant.sh` with metadata `jev-tp=4` on Queued Resource `jev-v5e4-suite`, which deleted itself when the run finished.

**The suite is level at TP=4** (−0.1 points, −0.6 to +0.5). GSM8K is 0.9 points lower (−1.7 to −0.2) and BFCL 0.8 lower; E2B moves the other way on the same tasks (+1.4 and +1.2), so splitting the matmuls four ways changes which long greedy answers come out right without a direction. No repeat of a TP=1 generation run exists to size that spread at one chip. With the sweep's serve flags the suite arm runs 2.4x one chip at 1, 4 and 16 requests.

#### Settings

Everything but the chip count matches the v5e-1 arms: image `vllm/vllm-tpu@sha256:19a1a052…` patched with `kvshare.diff wna16.diff moe.diff lowmem.diff textonly.diff` (lowmem.diff `27e40c1b`; the v5e-1 suite arms used earlier versions of it whose differences touch only the int4 embedding method's decode path and the MTP drafter), `GMM_V2_TILE_VMEM_FRACTION=0.85 MIN_TOKEN_BUCKET=64`, `--max-num-batched-tokens 512`, the checkpoint copied from `gs://aisprint-491218-bucket/jev-tpu-31b/models/gemma-4-12B-it-qat-w8a8-int8-emb4` (the same files the v5e-1 arms served). Suite arm `12b-w8a8-emb4`: `--gpu-memory-utilization 0.85 --num-gpu-blocks-override 40`, `--max-model-len 2048`. GSM8K/BFCL arm `12b-w8a8-emb4-tools`: `--gpu-memory-utilization 0.92 --max-model-len 4096`, `--enable-auto-tool-choice --tool-call-parser gemma4`, GSM8K answer limit 2,048 tokens. Code bundle `suite-tp4-e2a8358a.tgz`.

#### Suite and the four tasks against TP=1 on v5e-1

`quant_compare.py`, 2,000 bootstrap resamples. Reference: `2026-09-29-12b-v5e1-12b-w8a8-emb4` (`../../../../tpu-vllm-v5e1-12b-w8a8emb4/benchmarks/runs/2026-09-29-12b-12b-w8a8-emb4-v5e1/`).

| group | n | v5e-1, TP=1 | v5e-4, TP=4 | difference (points) | 95% range | v5e-1 right only | v5e-4 right only |
|---|---:|---:|---:|---:|---|---:|---:|
| sst2 | 300 | 0.950 | 0.947 | -0.3 | -1.3 to +0.7 | 2 | 1 |
| ag_news | 300 | 0.860 | 0.857 | -0.3 | -2.0 to +1.3 | 4 | 3 |
| emotion | 300 | 0.587 | 0.583 | -0.3 | -2.0 to +1.3 | 4 | 3 |
| irony | 300 | 0.857 | 0.863 | +0.7 | -1.3 to +2.7 | 4 | 6 |
| reversed options (3 choice tasks) | 900 | 0.799 | 0.802 | +0.3 | -0.7 to +1.3 | 9 | 12 |
| suite (all subsets) | 3,880 | 0.761 | 0.760 | -0.1 | -0.6 to +0.5 | 61 | 59 |

#### GSM8K and BFCL simple against TP=1 on v5e-1

`gen_compare.py`, 10,000 bootstrap resamples. Reference: `2026-09-30-gen2048c-v5e1-12b-w8a8-emb4-gen` (`../../../../tpu-vllm-v5e1-12b-w8a8emb4/benchmarks/runs/2026-09-30-gen2048c-12b-w8a8-emb4-v5e1/`).

| task | n | v5e-1, TP=1 | v5e-4, TP=4 | difference (points) | 95% range | v5e-1 right only | v5e-4 right only |
|---|---:|---:|---:|---:|---|---:|---:|
| gsm8k | 1,319 | 0.964 | 0.955 | -0.9 | -1.7 to -0.2 | 20 | 8 |
| bfcl_simple | 400 | 0.955 | 0.948 | -0.8 | -1.8 to +0.0 | 3 | 0 |

#### Throughput of the suite arm

`../../../../jev-tpu-v5e1/tpu/w4a16_client.py load` on the VM, output tok/s, median of three passes. Same flags as the v5e-1 suite arm, whose load JSONs are in its logs.

| Requests | v5e-1, TP=1 | v5e-4, TP=4 | v5e-4 / v5e-1 |
|---:|---:|---:|---:|
| 1 | 57.4 | 140.0 | 2.44x |
| 4 | 219.4 | 534.7 | 2.44x |
| 16 | 675.2 | 1,629.2 | 2.41x |

#### Files

| File | What |
|---|---|
| `2026-10-02-suite-tp4-v5e4-12b-w8a8-emb4{,-latency,-smoke}/` | the four tasks (both option orders), latency, smoke records |
| `2026-10-02-suite-tp4-v5e4-12b-w8a8-emb4-suite/` | the 3,880 suite records |
| `2026-10-02-suite-tp4-v5e4-12b-w8a8-emb4-tools-gen/` | GSM8K and BFCL simple records |
| `logs/2026-10-02-suite-tp4-v5e4-logs/` | this rig's arms' boot logs, host memory, load JSONs, gen summaries, vLLM version |
| `../../../../jev-tpu-v5e1/results/2026-10-02-suite-tp4-v5e4-logs/` | run-wide logs: `run.log`, patch logs, suite checksums |
| `../../../../jev-tpu-v5e1/results/2026-10-02-suite-tp4-v5e4-QUANT.md` | suite pairing for both rigs (its columns are headed "bf16" for v5e-1 and "4-bit" for v5e-4) |
| `../../../../jev-tpu-v5e1/results/2026-10-02-suite-tp4-v5e4-GEN-VS-V5E1.md` | GSM8K/BFCL pairing for both rigs |
