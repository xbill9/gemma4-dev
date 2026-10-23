# local-pytorch-1650ti-2b-q4w4a16

Gemma 4 E2B with Google's QAT 4-bit weights, in plain PyTorch on a 4 GB GTX 1650 Ti.

- Checkpoint: [`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text), every int4 code equal to Google's QAT release.
- Every linear, `embed_tokens`, the per-layer table and `lm_head` int4 on the GPU; a Triton int4 matrix-vector kernel at 83% of the card's measured bandwidth; the decode step compiled with Inductor and replayed as one CUDA graph.
- 87.8 tok/s decode at short context (1.07x llama.cpp on the same card the same day), and the full 131,072-token context in 3,471 MiB peak (decode 21.8 tok/s there, 0.72x), greedy replies token-identical to the fp16-table build on 8 of 8 prompts. bf16 does not fit the card.

Details and reproduction: [`benchmarks/runs/2026-10-23-close-gap-1650ti/REPORT.md`](benchmarks/runs/2026-10-23-close-gap-1650ti/REPORT.md); before it: [`2026-10-23-lmhead-kernel-1650ti`](benchmarks/runs/2026-10-23-lmhead-kernel-1650ti/REPORT.md); first light: [`benchmarks/runs/2026-10-23-first-light-1650ti/REPORT.md`](benchmarks/runs/2026-10-23-first-light-1650ti/REPORT.md).

#### Run it

```
pip install transformers==5.16.1 compressed-tensors==0.17.0 accelerate==1.14.0   # system python3, torch with CUDA
python3 openai_server.py --port 8100 --max-ctx 32768
curl localhost:8100/v1/chat/completions -d '{"messages":[{"role":"user","content":"Hi"}]}'
```

Or from Python: `from e2b_engine import Engine; eng = Engine(); eng.generate(eng.render(messages))`.
A Colab T4 notebook that runs it inside a 4 GB memory budget is in [`colab/`](colab/).
