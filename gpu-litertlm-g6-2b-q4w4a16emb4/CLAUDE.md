# CLAUDE.md — `gpu-litertlm-g6-2b-q4w4a16emb4`

LiteRT-LM 0.19.0 serving the `.litertlm` that `local-litertlm-3050oem-2b-q4w4a16emb4` built, holding the QAT Q4_0
grid exactly, on one NVIDIA L4 in an EC2 G6. The sister of `gpu-litertlm-g6-2b-int248`; slot 5 is the only slot that
differs. Forked 2026-11-05.

**No `server.py`, MCP server or skill.** The instance is `gpu-pytorch-g6-2b-q4w4a16`'s `create_g6_instance` (g6.2xlarge, us-east-1, the PyTorch 2.13 / Ubuntu 26.04 DLAMI, SSM access); this rig is `tpu.env`, a `Makefile` run on that instance, and its measurements. `tpu.env` decides every value. **Nothing is measured on the L4 yet**: every value is the 3050 rig's, and the unknowns are listed in `gpu-vllm-g6-2b-q4w4a16emb4/flight/FLIGHT.md`. The flight driver is `gpu-vllm-g6-2b-q4w4a16emb4/flight/engines.py`; `flight/setup.sh` installs every engine in order. One engine serves at a time on the card.

- **The file is not rebuilt here.** Its export peaked near 45 GB of RAM on the 3050 host; g6.2xlarge has 32,768 MiB.
  `make stage` on the 3050 host uploads it to `MODEL_S3_URI`, `make fetch` on the instance copies it back, both
  checking `MODEL_SHA256`.
- The build has no drafter section, so `SPECULATIVE_DECODING=off`.
- Vulkan: see `gpu-litertlm-g6-2b-int248` (`make vulkan` there).
