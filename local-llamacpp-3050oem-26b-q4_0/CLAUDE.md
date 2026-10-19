# CLAUDE.md — local-llamacpp-3050oem-26b-q4_0

Guidance for working inside this rig. The siblings are not layers; nothing is
imported across a rig boundary. Read this file before changing anything.

## What this rig is

`llama-server` from `ggml-org/llama.cpp`, driven directly, serving Google's QAT Q4_0
GGUF of Gemma 4 **26B A4B**-it (`google/gemma-4-26B-A4B-it-qat-q4_0-gguf`, SHA-256
`3eca3b8f…`, unmodified) off the **RTX 3050 OEM** in the desktop `gpu`, with the exact
Q4_0 build of Google's 26B drafter for speculative decoding
(`xbill9/gemma-4-26B-A4B-it-qat-assistant-q4_0-exact-gguf`, `6cee35a0…`). Text only.

**Forked 2026-10-18 from `local-llamacpp-3050oem-12b-q4_0`**, its 12B sister on the
same card, host and llama.cpp `7182ef71e`. Slot 4 is the only slot that differs. All
the 3050 rigs use port 8080 and only one fits on the card at a time.

**The model does not fit the card; its experts live in system RAM.** The file is
14.4 GB: 12,251 MiB of expert weights (408 MiB per layer, 30 layers, 128 experts, 8
used per token) and 1,505 MiB of everything else. `N_CPU_MOE` (`-ncmoe`) keeps the
experts of the first N layers in RAM, where the CPU runs them; attention, norms and the
router of every layer stay on the card. **N_CPU_MOE trades speed for context**: each
layer moved to the card adds about 0.5–0.6 tok/s and costs about 410 MiB.

**The CPU is part of the serving path here**, unlike on the sisters: thread count
changes generation speed (`-t 8` serves), and anything else busy on the CPU or memory
bus slows the model.

## Measurements — 2026-10-18

`benchmarks/runs/2026-10-18-first-light-3050oem/` (REPORT.md has every table).

- **Serves** `-ncmoe 21`, q4_0 cache, `-c 131072`, drafter at 2 tokens, 7,850 MiB.
  Code generates at 40.6 / 34.1 / 28.4 / 23.6 tok/s at 0 / 32k / 65k / 98k tokens held,
  prose at 27.9 / 24.0 / 19.5 / 16.6; without the drafter 28.2 / 22.2 / 17.5 / 14.1.
- **Expert layers in RAM** (llama-bench tg128, short context): 30 = 24.35 tok/s,
  24 = 27.66, 20 = 30.65, 18 = 32.27, 17 = 33.31; 16 does not load.
- **Context that fits**, q4_0: 131,072 at `-ncmoe` 19–21 without the drafter and 21–22
  with it; the model's full 262,144 at 22 without it. f16 holds 32,768 at most.
  Full-attention KV is 20 KiB/token at f16.
- **Drafter:** 1.24x overall at 2 tokens (code 1.51x, prose 1.12–1.17x), level with 1
  token; 3 tokens lose on sampled text (0.61x). The 12B's drafter gives 2.01x; with the
  experts on the CPU, extra checked tokens likely cost expert reads (inferred). Its gain
  grows with context held: code 1.44x empty, 1.67x at 98k. Greedy text differs from no
  drafter.

**Answer quality** (`benchmarks/runs/2026-10-18-accuracy-26b-3050oem/`, at the serving setting):
GSM8K 0.965 / 0.962 and BFCL 0.945 / 0.945 with / without the drafter, level with each other and
with the 12B sister (paired ranges include zero). GSM8K takes 1.76x the 12B's time; the drafter
finishes it 1.35x faster.

**Not measured:** the q4_0 cache against f16 for answers, an exact Q4_0 rebuild of the target,
concurrency.

## Inherited and still true here

- **Never pass `--no-mmap`.** The experts kept in RAM are read from the mmap.
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
