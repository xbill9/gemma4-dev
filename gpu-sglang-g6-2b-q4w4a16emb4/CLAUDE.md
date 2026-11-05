# CLAUDE.md — `gpu-sglang-g6-2b-q4w4a16emb4`

SGLang 0.5.23 with the 3050 rig's two patches (`patches/`) serving the vLLM G6 rig's checkpoint and revision on one
NVIDIA L4 in an EC2 G6. Forked 2026-11-05 from `local-sglang-3050oem-2b-q4w4a16emb4`; slots 1 and 3 differ.

**No `server.py`, MCP server or skill.** The instance is `gpu-pytorch-g6-2b-q4w4a16`'s `create_g6_instance` (g6.2xlarge, us-east-1, the PyTorch 2.13 / Ubuntu 26.04 DLAMI, SSM access); this rig is `tpu.env`, a `Makefile` run on that instance, and its measurements. `tpu.env` decides every value. **Nothing is measured on the L4 yet**: every value is the 3050 rig's, and the unknowns are listed in `gpu-vllm-g6-2b-q4w4a16emb4/flight/FLIGHT.md`. The flight driver is `gpu-vllm-g6-2b-q4w4a16emb4/flight/engines.py`; `flight/setup.sh` installs every engine in order. One engine serves at a time on the card.

- **Install after `gpu-vllm-g6-2b-q4w4a16emb4`'s `make pip install`**: same system python3 and user site, and that
  install brings torch. `make install` retags the cp313 wheel for cp314 and installs it `--no-deps`, then the
  requirements the 3050 found missing, then applies `patches/` to the user site (`python3 -m site --user-site`, root's
  on the instance, not the 3050's `~/.local/lib/python3.14`).
- **Both patches are still required**: they fix SGLang's handling of a text-only Gemma 4 config and its int4
  `embed_tokens`/`embed_tokens_per_layer` loading, properties of SGLang and the checkpoint rather than the card.
- Prefill CUDA graphs stay off (they refuse the KV-shared layers); decode graphs to 64.
- The drafter arm uses `gpu-vllm-g6-2b-q4w4a16emb4`'s `DRAFTER_DIR`.
