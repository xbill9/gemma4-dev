# gpu-llamacpp-g6-4b-q4_0

Gemma 4 E4B with Google's QAT 4-bit weights in llama.cpp on an NVIDIA L4 (AWS EC2 G6), with Google's QAT drafter for speculative decoding.

- Target: [`xbill9/gemma-4-E4B-it-qat-q4_0-exact-gguf`](https://huggingface.co/xbill9/gemma-4-E4B-it-qat-q4_0-exact-gguf), every Q4_0 block on Google's QAT grid.
- Drafter: [`xbill9/gemma-4-E4B-it-qat-assistant-q4_0-exact-gguf`](https://huggingface.co/xbill9/gemma-4-E4B-it-qat-assistant-q4_0-exact-gguf).
- 74.0 tok/s without the drafter, **129.7 tok/s (1.75x) at 3 drafted tokens**, GSM8K and BFCL level with and without it ([`REPORT.md`](benchmarks/runs/2026-10-26-drafter-g6/REPORT.md)).

On a G6 instance: `make toolkit build download draft serve`.
