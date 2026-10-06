# Baked Docker image (vLLM/OpenAI-compatible) — Gemma-4-E2B on inf2

**ECR:** `106059658660.dkr.ecr.us-east-2.amazonaws.com/gemma4-optb:256-64` (also `:latest`)
~16.9 GB compressed / 40.4 GB uncompressed (CPU-torch build). Built + verified 2026-07-06.

Self-contained: neffs (256/64) + real model + transformers 5.13 + torch_neuronx 2.8.0.2.12
+ the box's `/opt/aws/neuron` runtime (libnrt) all baked in. Version-pinned so the neffs load.

## Run (on any inf2 host with the Neuron driver)
```bash
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 106059658660.dkr.ecr.us-east-2.amazonaws.com
docker run -d --name gemma-optb --device /dev/neuron0 --ipc=host -p 8080:8080 \
  106059658660.dkr.ecr.us-east-2.amazonaws.com/gemma4-optb:256-64
# ~82s to READY (warm), then:
curl -s -X POST localhost:8080/v1/chat/completions -d '{"messages":[{"role":"user","content":"hi"}]}'
```

Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/generate`
(OpenAI-compatible, sampling via temperature/top_p/top_k, no auth). Container name `gemma-optb`,
port 8080, `--device /dev/neuron*` → matches the gpu-devops-agent MCP `check_vllm`/`query_vllm` contract.

## Notes
- Needs Neuron HARDWARE (inf2) + host driver; not runnable on a laptop.
- Uses CPU torch (installed from a local wheel — the build net can't TLS-handshake PyTorch's R2 CDN; `curl` the wheel on the host, COPY it in). Saved ~9 GB vs the CUDA build. Remaining bulk is the model (10 GB) + neffs (7 GB), irreducible.
- Rebuild: see Dockerfile; build context = /workspace/imgbuild (hardlinked artifacts + opt_neuron).
