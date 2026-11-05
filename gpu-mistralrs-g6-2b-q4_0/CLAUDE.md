# CLAUDE.md — `gpu-mistralrs-g6-2b-q4_0`

mistral.rs v0.9.5 (built with CUDA and flash-attn for sm_89) serving the exact Q4_0 E2B GGUF that
`gpu-llamacpp-g6-2b-q4_0` serves, same path and hash, on one NVIDIA L4 in an EC2 G6, so the pair is an engine A/B on
identical bytes. Forked 2026-11-05 from `local-mistralrs-3050oem-2b-q4_0`; slots 1 and 3 differ.

**No `server.py`, MCP server or skill.** The instance is `gpu-pytorch-g6-2b-q4w4a16`'s `create_g6_instance` (g6.2xlarge, us-east-1, the PyTorch 2.13 / Ubuntu 26.04 DLAMI, SSM access); this rig is `tpu.env`, a `Makefile` run on that instance, and its measurements. `tpu.env` decides every value. **Nothing is measured on the L4 yet**: every value is the 3050 rig's, and the unknowns are listed in `gpu-vllm-g6-2b-q4w4a16emb4/flight/FLIGHT.md`. The flight driver is `gpu-vllm-g6-2b-q4w4a16emb4/flight/engines.py`; `flight/setup.sh` installs every engine in order. One engine serves at a time on the card.

- **Build order**: `gpu-llamacpp-g6-2b-q4_0`'s `make toolkit download` (CUDA 13.1, the GGUF), then here
  `make deps rust mmproj install`. `CUDA_COMPUTE_CAP=89`; the flash-attn kernels are most of the build time.
- **The serving limits in `tpu.env` are the 3050's** (4 requests, fixed 8,192-token cache, 1,024 tokens per step),
  forced there by 8 GB. They stay the default so the like-for-like arm matches; the flight's `mistralrs-card` arm
  tries 64 requests, a 65,536-token cache and 4,096 per step, untested.
- The projector (`MMPROJ_*`) is Google's file at a pinned revision: mistral.rs loads one for the gemma4 GGUF
  architecture whatever the request.
- The drafter is Google's QAT assistant (bf16 safetensors) at `MTP_HF_REVISION`, fetched by the flight driver.
