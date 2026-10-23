# local-pytorch-1650ti-2b-q4w4a16

Gemma 4 E2B with Google's QAT 4-bit weights, in plain PyTorch on a 4 GB GTX 1650 Ti.

- Checkpoint: [`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text), every int4 code equal to Google's QAT release.
- Packed int4 layers on the GPU, per-layer embeddings in CPU RAM, a Triton int4 matrix-vector kernel and a captured CUDA graph per decode step.
- 18.1 tok/s at 1,842 MiB peak, greedy reply token-identical to bf16, which does not fit the card.

Details and reproduction: [`benchmarks/runs/2026-10-23-first-light-1650ti/REPORT.md`](benchmarks/runs/2026-10-23-first-light-1650ti/REPORT.md).
