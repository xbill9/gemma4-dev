# AGENTS.md — `gpu-pytorch-g6-2b`

Guidance for coding agents working in this rig. **`CLAUDE.md` is authoritative where these
disagree**; there is no generator, so a convention change has to be applied to all three of
`CLAUDE.md`, `AGENTS.md` and `GEMINI.md` by hand.

## What this rig is

Serves **`google/gemma-4-E2B-it`** under **stock PyTorch + HF transformers** on **AWS EC2
G6** — an **x86_64** host with an **NVIDIA L4** GPU (Ada, SM 8.9, **23034 MiB measured**).

No custom model port, no XLA, no vLLM, no docker. `AutoModelForCausalLM` behind
`torch_openai_server.py`, under systemd (`torch-g6.service`).

**It has served**: 20.93 tok/s median decode, `benchmarks/runs/2026-08-29-first-serve-g6/`.

## Rules that cost a measurement when broken

- **`make skill` before `deploy_torch_server`.** The deploy resolves its payload next to
  `server.py`; from the MCP snapshot it silently ships the previous `make skill` output.
- **Quote `tpu_jax_decode_tokens_per_second`**, never an end-to-end rate. The prefix is an
  identifier, not a description.
- **Warm at the shape you measure.**
- **Never attribute a sibling's number to this rig.** 10.88 tok/s is the T4G; 48.40 is the
  JAX rig on this chip *with 40% fewer weight bytes*; 43–44 is vLLM on a T4G.
- **A flag being accepted is not evidence it did anything.** Cross-check against the physical
  envelope: ~300 GB/s and 23034 MiB.

## Things that are true here and false in a sibling

- **bfloat16 is native.** Resolved from the live compute capability, not from `tpu.env`.
- **`g6.16xlarge` is single-GPU**; multi-GPU sizes are 12/24/48xlarge; no `g6.metal`.
- **G6 has twice G5g's host RAM at each suffix.**
- **The x86_64 DLAMI's name contains no arch string.**
- **`torch.cuda.get_arch_list()` has no `sm_89`, and that is fine** — cubins run on any
  device of the same major with minor ≥ their own. Never write an exact-match arch check.
- **There is no quantization path at all.** `QUANT_MODE`, `PLE_BITS`, `INT8_LM_HEAD`,
  `PREFILL_CHUNK_SIZE` in `tpu.env` are inert; do not plumb them into `_serve_argv`.

## Code style

- Every subprocess call goes through `run_command(cmd: list[str])`. **Never `shell=True`.**
- MCP tools are `async def` returning markdown with emoji status prefixes (✅, ❌, 📡).
- Match the surrounding file's annotation style.
- Tests are **`unittest`, never pytest**: `python3 -m unittest discover -s tests -v`.
- `make lint` lints a **hardcoded file list**; a new top-level module is silently unlinted
  until it is added to it.

## Registration

`.mcp.json`, `.claude-plugin/plugin.json`, `.codex/config.toml` and
`.claude/settings.local.json` must all name `gpu-pytorch-g6-2b`. Only `.mcp.json` is
generated. **Ground truth for tool names is `grep -n "^@mcp.tool" server.py`** — a Codex
approval gate naming a tool that does not exist fails open and says nothing.
