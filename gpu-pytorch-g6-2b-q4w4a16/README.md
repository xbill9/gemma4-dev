# gpu-pytorch-g6-2b-q4w4a16

Gemma 4 E2B with Google's QAT 4-bit weights, in plain PyTorch on an NVIDIA L4 (AWS EC2 G6).

- Checkpoint: [`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text), every int4 code equal to Google's QAT release; vocabulary tables from [`-text-emb4`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4).
- Engine: the one from [`local-pytorch-1650ti-2b-q4w4a16`](../local-pytorch-1650ti-2b-q4w4a16/) — int4 weights packed on the GPU, a Triton int4 GEMV, split decode attention, and the decode step compiled with Inductor and replayed as a CUDA graph.
- Provisioning: an MCP server (`server.py`) that launches a G6 spot instance on the PyTorch DLAMI, installs the pinned runtime, ships the engine over SSM and runs it under systemd. Forked from [`gpu-pytorch-g6-2b`](../gpu-pytorch-g6-2b/).

First light 2026-10-26: 141.35 tok/s decode at short context and 51.41 at 131k on the L4, at the card's memory limit for one request ([`REPORT.md`](benchmarks/runs/2026-10-26-first-light-g6/REPORT.md)).

#### Run it

Through the MCP server: `create_g6_instance` → `get_install_progress` → `verify_gpu_arch` → `deploy_torch_server` → `get_torch_logs` → `verify_model_health`.

On any CUDA machine with torch and Triton:

```
pip install transformers==5.16.1 compressed-tensors==0.17.0 accelerate==1.14.0
python3 openai_server.py --host 0.0.0.0 --port 8000 --max-ctx 131072
curl localhost:8000/v1/chat/completions -d '{"messages":[{"role":"user","content":"Hi"}]}'
```
