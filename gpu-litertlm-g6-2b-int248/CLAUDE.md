# CLAUDE.md — `gpu-litertlm-g6-2b-int248`

LiteRT-LM 0.19.0 serving Google's `gemma-4-E2B-it.litertlm` unmodified (int2/int4/int8, one scale per row, with its
built-in drafter) on one NVIDIA L4 in an EC2 G6. Forked 2026-11-05 from `local-litertlm-3050oem-2b-int248`; slots 1
and 3 differ.

**No `server.py`, MCP server or skill.** The instance is `gpu-pytorch-g6-2b-q4w4a16`'s `create_g6_instance` (g6.2xlarge, us-east-1, the PyTorch 2.13 / Ubuntu 26.04 DLAMI, SSM access); this rig is `tpu.env`, a `Makefile` run on that instance, and its measurements. `tpu.env` decides every value. **Nothing is measured on the L4 yet**: every value is the 3050 rig's, and the unknowns are listed in `gpu-vllm-g6-2b-q4w4a16emb4/flight/FLIGHT.md`. The flight driver is `gpu-vllm-g6-2b-q4w4a16emb4/flight/engines.py`; `flight/setup.sh` installs every engine in order. One engine serves at a time on the card.

- **The GPU path is WebGPU (Dawn) on Vulkan, not CUDA** (measured on the 3050). Whether the DLAMI ships the NVIDIA
  Vulkan ICD and a headless L4 enumerates is unverified; `make vulkan` fails unless `vulkaninfo` lists the L4.
- System python3 user site with protobuf ≥ 6.31.1, as on the 3050.
- Per-rig state in `LITERT_LM_DIR` (`/opt/cache/gpu-litertlm-g6-2b-int248`).
- The server ignores `max_tokens`; only `max_completion_tokens` caps a reply.
