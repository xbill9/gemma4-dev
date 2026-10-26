# GEMINI.md — `gpu-pytorch-g6-2b-q4w4a16`

Guidance for coding agents. **`CLAUDE.md` is authoritative where these disagree**; `CLAUDE.md`, `AGENTS.md` and
`GEMINI.md` are kept in step by hand.

Gemma 4 E2B, Google's QAT 4-bit weights, served on one NVIDIA L4 (AWS EC2 G6) by the PyTorch int4 engine of
`local-pytorch-1650ti-2b-q4w4a16`. Provisioning (boto3, SSM, DLAMI, systemd) is forked from `gpu-pytorch-g6-2b`.
Nothing has been measured on this rig yet; the 1650 Ti, T4 and bf16-L4 numbers belong to other rigs.

- Keep `e2b_engine.py`, `w4gemv.py`, `w4attn.py`, `gemma4_tools.py` byte-identical to the 1650 Ti rig's.
- `openai_server.py` flags: `--host`, `--port`, `--max-ctx`, `--prefill-dtype` only.
- `make skill` before `deploy_torch_server`; quote `tpu_jax_decode_tokens_per_second`.
- Tests: `python3 -m unittest discover -s tests -v`. System `python3`, never a virtualenv.
