# CLAUDE.md — local-llamacpp-3050oem-31b-q4_0

Guidance for working inside this rig. The siblings are not layers; nothing is
imported across a rig boundary. Read this file before changing anything.

## What this rig is

`llama-server` from `ggml-org/llama.cpp`, driven directly, serving Google's QAT Q4_0
GGUF of Gemma 4 **31B**-it (`google/gemma-4-31B-it-qat-q4_0-gguf`, SHA-256 `179cfb99…`,
unmodified) off the **RTX 3050 OEM** in the desktop `gpu`, with the exact Q4_0 build of
Google's 31B drafter for speculative decoding
(`xbill9/gemma-4-31B-it-qat-assistant-q4_0-exact-gguf`, `1c9b4c35…`). Text only.

**Forked 2026-10-18 from `local-llamacpp-3050oem-26b-q4_0`**, its 26B A4B sister on the
same card, host and llama.cpp `7182ef71e`. Slot 4 is the only slot that differs. All the
3050 rigs use port 8080 and only one fits on the card at a time.

**The model does not fit the card, and it is dense.** 60 layers of 257–287 MiB (15.7 GB)
plus a 1.1 GB Q6_K `token_embd`. `N_GPU_LAYERS` puts that many layers on the card and the
rest run on the CPU. Unlike the 26B, which reads only its active experts, **every token
reads every layer**, so the CPU side sets the speed: about 0.05 tok/s per layer moved to the
card. `N_CPU_MOE` is carried from the 26B sister and stays empty.

## Measurements — 2026-10-18

`benchmarks/runs/2026-10-18-first-light-3050oem/` (REPORT.md has every table).

- **Serves** `-ngl 20`, q4_0 cache, `-c 32768`, drafter at 1 token, 7,734 MiB, about 4 tok/s.
- **Layers on the card** (llama-bench tg64): 0 = 1.68 tok/s, 12 = 2.13, 20 = 2.52, 22 = 2.62;
  24 does not load. `-t 8` (2.66 at 22), level with 6 and 12.
- **Drafter:** 1 token 1.59x (4.0 tok/s), 2 tokens 1.21x, 3 tokens 1.37x, 4 tokens 1.09x. One
  run per cell. Longer drafts lose because checking several tokens costs the CPU-side layers
  more than one.
- **Prompt reading:** 145 tok/s at the default `-ub 512`; `-ub 2048` reaches 238 but only fits
  with 12 layers on the card (2.13 tok/s generation).
- **Context that fits**, q4_0: with the drafter 32,768 at `-ngl 20`, 65,536 at 16, 131,072 at
  12; without it 262,144 at 10.

**Answer quality** (`benchmarks/runs/2026-10-19-accuracy-31b-3050oem/`, in progress): BFCL
0.960 / 0.958 with / without the drafter, level with the 26B (0.945) and 12B (0.950). GSM8K running,
about 59 s per question with the drafter.

**Not measured:** generation with a long conversation held, concurrency.

## Inherited and still true here

- **Never pass `--no-mmap`.** The layers run on the CPU are read from the mmap.
- Gemma 4 reasons; `REASONING=off` is the server default and `query_model`
  defaults to `max_tokens` 1024. Use `/v1/chat/completions`, not
  `/v1/completions`.
- `sweep.py`: `decode_tps` counts chunks, not tokens; verify the prompt cache
  was defeated via `/metrics`; llama.cpp splits `--ctx-size` across slots.
- `HOST=0.0.0.0`: anyone on the LAN can use it. No auth.

## Conventions

- Tests are `unittest`, never pytest: `python3 -m unittest discover -s tests -v`.
- Every subprocess call goes through `run_command(cmd: list[str])` using
  `asyncio.create_subprocess_exec`. **Never `shell=True`.**
- MCP tools are `async def` returning markdown strings with emoji status
  prefixes (`✅`, `❌`, `📡`).
- `Optional[str]`, not `X | None`.
- Use the system `python3` and install into it. **Never create a virtualenv.**
- `tpu.env` is the source of truth and is committed. Never add `*.env` to
  `.gitignore`.

## Canonical root references

`@MODELS.md`, `@HARDWARE.md`, `@QUANTIZATION.md`, `@NAMING.md`,
`@RIG-ANALYSIS.md`. Read them before deriving their numbers here, and correct them
there.
