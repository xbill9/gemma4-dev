# CLAUDE.md — `gpu-pytorch-g6-2b-q4w4a16`

Serving rig: **Gemma 4 E2B with Google's QAT 4-bit weights** (`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text`,
vocabulary tables from `-text-emb4`) through the **hand-built PyTorch int4 engine** of
`local-pytorch-1650ti-2b-q4w4a16`, on **AWS EC2 G6**: an x86_64 host with one **NVIDIA L4** (Ada, SM 8.9, 23034 MiB
measured by `gpu-pytorch-g6-2b`).

A full rig: `server.py` (MCP server), skill, plugin manifest, `tpu.env`. Created 2026-10-26 from two parents:

- **The serving payload** — `e2b_engine.py`, `w4gemv.py`, `w4attn.py`, `gemma4_tools.py` — is copied byte-for-byte
  from `local-pytorch-1650ti-2b-q4w4a16`. `openai_server.py` is that rig's server plus `/metrics`, a build id on
  `/health`, `usage.decode_tokens_per_second` and `--prefill-dtype auto`. **Keep the four engine files identical to
  the 1650 Ti rig's**; a fix belongs in both (`ruff.toml` ignores E741 for them so they need not change here).
- **The AWS half** — boto3, SSM, the PyTorch DLAMI, the two-stage bootstrap, systemd — is `gpu-pytorch-g6-2b`'s
  (dense bf16, stock transformers, same instance family). Comments in `server.py` that describe that machinery are
  its, kept where they still hold; its measurements did not come along.

**First light 2026-10-26** (`benchmarks/runs/2026-10-26-first-light-g6/REPORT.md`): decode **141.35 / 133.08 / 96.14 /
51.41 tok/s** at 61 / 4k / 32k / 131k tokens held, 1.61–2.36x the 1650 Ti on the same prompts, prompt reads 2,000–2,700
tok/s in fp16, recall identical to the 1650 Ti prompt for prompt, 3,395 MiB peak. **Single-request decode is at the
card's memory limit**: a 1 GiB copy runs at 241.2 GB/s, the int4 GEMV is 6.0 of 7.3 ms per step and moves its 1.28 GB
at 0.886 of that rate, so the ceiling for this engine is about 151 tok/s. The 1650 Ti tilings stay (autotuning found
nothing faster). Against v5e-1 vLLM on the same QAT int4 model (128.2 / 446.0 / 1,171.0 tok/s at 1 / 4 / 16 requests)
one request is level; the gap is batching, which this engine does not do.

Numbers that belong to other rigs: **87.8 / 21.8 tok/s** (the engine on a GTX 1650 Ti), **101.6–117.5 tok/s** (a Colab
T4 under a 3,732 MiB cap), **20.93 tok/s** (`gpu-pytorch-g6-2b`: this L4, dense bf16, stock transformers).

## What carries from the 1650 Ti, and what may not

- **Compute is float16 on every GPU** (`e2b_engine.DTYPE`), although the L4 has native bf16. Both probes in
  `server.py` (the bootstrap's `verify_gpu` and `verify_gpu_arch`) run an fp16 matmul for that reason, and the
  bootstrap also imports Triton: the GEMV, decode attention and Inductor's compiled step all need it.
- **`w4gemv.py` pins one tiling per shape, tuned on the 1650 Ti, and they hold on the L4**: v4's autotuner found nothing
  faster (`benchmarks/runs/2026-10-26-first-light-g6/tune.jsonl`). Benchmark the GEMV with weights rotated past the
  48 MiB L2 (`cold_gemv.py`); with one copy replayed, the L4 reports L2 speeds above its 241 GB/s memory.
- **Prompt-read dtype is chosen on the device.** fp32 won on the 1650 Ti (no tensor cores; fp16 GEMMs at a quarter
  of fp32), fp16 won on the T4 (36.58 vs 6.68 TFLOP/s). `--prefill-dtype auto` times both at start-up and logs the
  rates in the server's first JSON line; `PREFILL_DTYPE` in `tpu.env` forces one.
- **The SDPA fix carries**: the engine copies sliding-layer K/V rather than expanding it, which avoids torch 2.13's
  wrong last query row at 32k+1-query chunks. The DLAMI's torch is 2.13 too (as measured by `gpu-pytorch-g6-2b`).
- **transformers, compressed-tensors and accelerate are pinned** (`5.16.1`, `0.17.0`, `1.14.0`) in `TORCH_PIP_SPEC`:
  the engine replaces Gemma 4's attention forward and removes the compressed-tensors decompress hook, so a newer
  transformers can break it silently.
- **Memory is not the limit here**: on the 1650 Ti the full 131,072-token context peaks at 3,471 MiB, so
  `MAX_MODEL_LEN=131072` is the default. The L4 has 23034 MiB.

## Order of operations

```
create_g6_instance → get_install_progress → verify_gpu_arch → deploy_torch_server
                   → get_torch_logs → verify_model_health → get_metrics
```

`make skill` before `deploy_torch_server`: the MCP server ships the payload next to itself, so from the skill
snapshot it ships the previous `make skill` output; `verify_model_health` compares build ids and says `STALE DEPLOY`.

The server's start-up downloads both checkpoints, times the GEMMs, loads, and compiles and captures the decode step
before it listens; the first request is already warm. Its first stdout line is JSON: device, `prefill_dtype`,
`gemm_tflops`, `weights_mib`, `kv_mib`, `load_s`. Read it with `get_torch_logs`. The unit is `torch-g6-q4.service`;
files live in `/opt/torch-g6-q4`.

## Serving

- `openai_server.py` defines exactly `--host`, `--port`, `--max-ctx`, `--prefill-dtype`; `_serve_argv` emits those
  and a test reads the set from the source. There is no `--model`: the checkpoints are pinned in `e2b_engine.py`
  (`TEXT`, `EMB4`); `MODEL_NAME` labels requests and metrics only.
- One request at a time (a lock around `generate`), greedy, non-streaming. Concurrency is not an axis.
- `/metrics` keeps the `tpu_jax_*` names `server.py`, `sweep.py` and the sibling reports use. Quote
  `tpu_jax_decode_tokens_per_second`. `tpu_jax_cold_requests_total` is always 0 (the step is captured at start-up).
- `tpu_jax_degenerate_responses_total` counts replies ending in a 1–4-token cycle; `verify_model_health` reads it
  either side of its probe.
- `tpu.env` keys `QUANT_MODE`, `PLE_BITS`, `INT8_LM_HEAD`, `KV_CACHE_DTYPE`, `MAX_NUM_SEQS` and `DTYPE` document the
  engine and are not forwarded.

## Inherited from `gpu-pytorch-g6-2b` and still true

Read that rig's `CLAUDE.md` for the evidence; the short form:

- `g6.2xlarge` default (8 vCPU, 32 GiB, 1 L4). `g6.16xlarge` is single-GPU; nothing shards.
- The AMI comes from the SSM parameter for the x86_64 PyTorch 2.13 / Ubuntu 26.04 DLAMI; never hardcode an id.
- The interpreter is probed (`/opt/torch-g6-q4/PYTHON_BIN`): the DLAMI's torch lives in a venv.
- CUDA cubin compatibility is same-major, minor ≤ device: `sm_86` covers the L4's `sm_89`.
- boto3 and SSM only; explicit subnet / security group / instance profile; HF token from Secrets Manager, never in
  user data; spot by default; discovery scoped to `ManagedBy=gpu-pytorch-g6-2b-q4w4a16`.

## Commands

`python3 -m unittest discover -s tests -v` (offline: no AWS, no GPU). `make lint` lints a hardcoded list; a new
top-level module is unlinted until added (a test checks). `make skill` regenerates both snapshots; `SKILL.md` is a
hand-written source inside `.claude/skills/…`.

MCP registration lives in four places — `.mcp.json` (generated by `project-setup.sh`), `.claude-plugin/plugin.json`,
`.codex/config.toml`, `.claude/settings.local.json` — all naming `gpu-pytorch-g6-2b-q4w4a16`.

Record runs in `benchmarks/runs/<date>-<what>-g6/`. `AGENTS.md` and `GEMINI.md` summarise this file; this file wins.
