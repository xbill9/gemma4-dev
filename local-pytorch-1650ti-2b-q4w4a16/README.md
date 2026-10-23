# local-pytorch-1650ti-2b-q4w4a16

Gemma 4 E2B with Google's QAT 4-bit weights, in plain PyTorch on a 4 GB GTX 1650 Ti.

- Checkpoint: [`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text), every int4 code equal to Google's QAT release.
- Every linear, `embed_tokens`, the per-layer table and `lm_head` int4 on the GPU; a Triton int4 matrix-vector kernel at 83% of the card's measured bandwidth; the decode step compiled with Inductor and replayed as one CUDA graph.
- 57.3–57.6 tok/s decode at 2,769 MiB peak, greedy replies token-identical to the fp16-table build on 8 of 8 prompts. bf16 does not fit the card.

Details and reproduction: [`benchmarks/runs/2026-10-23-lmhead-kernel-1650ti/REPORT.md`](benchmarks/runs/2026-10-23-lmhead-kernel-1650ti/REPORT.md); first light: [`benchmarks/runs/2026-10-23-first-light-1650ti/REPORT.md`](benchmarks/runs/2026-10-23-first-light-1650ti/REPORT.md).
