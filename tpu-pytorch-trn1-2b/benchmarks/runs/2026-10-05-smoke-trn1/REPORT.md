# Inf2-compiled Gemma 4 images on trn1.2xlarge — smoke test

- **Date:** 2026-10-05 (UTC), us-east-2c, one on-demand `trn1.2xlarge` (`i-0ae4b5e09a1d24e21`)
- **Host:** 8 vCPU, 32 GiB RAM, 1 Trainium device (2 NeuronCore-v2, 32 GiB), Ubuntu 24.04.4
- **AMI:** Deep Learning AMI Neuron PyTorch Inference vLLM 0.24.0.1.1.0 (Ubuntu 24.04) 20260818 (`ami-0cb0faefcc95d3e64`); `aws-neuronx-dkms 2.30.2.0`, `aws-neuronx-runtime-lib 2.34.10.0`
- **Question:** do the Gemma 4 images compiled for Inferentia2 load and serve on Trainium without a recompile?
- **Answer:** yes, for both images tried. No image was rebuilt and no flag was changed.

## Images

| Image | Digest | Built for |
|---|---|---|
| `xbill9/gemma4-optb:slim` (E2B) | `sha256:6cf0ae7e…` | inf2.xlarge, 512 total / 128 prompt tokens |
| `xbill9/gemma4-optb-26b:xlarge` (26B A4B MoE, int8 squeeze) | `sha256:32c10084…` | inf2.xlarge, MAX=512 BUCKET=128, ModelBuilder TP=2 |

Both ran as `docker run --device /dev/neuron0 --ipc=host -p 8080:8080 <image>` (the 26B with `-v gemma26b-data:/data`). A 32 GB swapfile was present and unused (`Swap: 0 used` at every check).

## Results

| | E2B | 26B A4B |
|---|---|---|
| Image pull | 365 s | 195 s |
| Pull to healthy | 171 s (server: `READY in 121.2s`) | 518 s (server: `READY in 220.1s`, includes fetching 27 GB of artifacts from Hugging Face) |
| "What is the capital of France?" (greedy) | `Paris`, 3 of 3 | `Paris`, 3 of 3 |
| Decode, server-reported | 36.2, 38.1, 36.2, 36.0, 36.6, 37.1 tok/s (six answers of 94–123 tokens) | 5.5 tok/s (three 101-token answers, `finish=length`) |
| Prefill, server-reported | — (slim server does not report it) | 0.31–0.34 s at 21–25 prompt tokens |
| Peak host RSS | 19.53 GB (server log) | not reported by the server |

Raw lines: `e2b-server.log`, `26b-server.log`. Stage times are differences of `date +%s` stamps taken on the instance (`t0` before `docker pull`, `tpull` after it, `tready` at the first `200` from `/health`, polled every 5 s).

## What this does and does not show

- It shows the inf2 NEFFs execute on Trainium's NeuronCore-v2 and produce the expected greedy answer and coherent paragraphs.
- It does **not** compare trn1 with inf2: there is no same-day inf2 run with the same requests here. The inf2 figures in `~/gemma4-tips-aws/gpu-4B-inf-devops-agent` (E2B slim ~24 tok/s; 26B ~6 tok/s, prefill 250 ms) came from different requests on different days, so no ratio is drawn.
- E2B's 19.53 GB peak exceeds inf2.xlarge's 16 GiB of RAM (which is why that box needs swap) and fits in trn1.2xlarge's 32 GiB.
