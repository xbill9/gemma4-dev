# CLAUDE.md — local-llamacpp-3050oem-12b-q4_0

Guidance for working inside this rig. The siblings are not layers; nothing is
imported across a rig boundary. Read this file before changing anything.

## What this rig is

`llama-server` from `ggml-org/llama.cpp`, driven directly, serving the exact
Q4_0 GGUF of Gemma 4 **12B**-it (`xbill9/gemma-4-12B-it-qat-q4_0-exact-gguf`,
SHA-256 `038337ee…`) off the **RTX 3050 OEM** in the desktop `gpu`, with the exact
Q4_0 build of Google's 12B drafter for speculative decoding
(`xbill9/gemma-4-12B-it-qat-assistant-q4_0-exact-gguf`, `f22f6366…`). Text only.

**Forked 2026-10-16 from `local-llamacpp-3050oem-4b-q4_0`**, its E4B sister.
Slot 4 is the only slot that differs: same card and host. **Not the same binary**:
the shared checkout moved to `7182ef71e` on 2026-10-16, and the sisters' `tpu.env`
still name `77c4540ce`; the 12B-vs-E4B ratios below come from an E4B pass on
`7182ef71e`. All three rigs use port 8080 and only one fits on the card at a time.

**Unlike E2B and E4B, the 12B has no per-layer table**, so nothing large stays in
the mmap and `OVERRIDE_TENSORS` is empty. The whole model is on the card, and the
**context cache sets the limit**, not the weights.

## Measurements — 2026-10-16

`benchmarks/runs/2026-10-16-exact-gguf-12b-3050oem/` (the build),
`2026-10-16-first-light-3050oem/` and `2026-10-16-drafter-q4exact-12b-3050oem/`.

| | 12B (this rig) | E4B, same binary |
| --- | ---: | ---: |
| generation, no drafter (llama-bench tg128) | 28.6 tok/s | 61.2 |
| generation, six prompts, no drafter (geomean) | 28.1 | — |
| generation, six prompts, 2-token drafter (geomean) | **56.4** (2.01x) | — |
| prompt reading (pp512) | 987 t/s | 2,241 |
| weights on the card | 6,390 MiB | — |
| VRAM as served (`-c 24576`, drafter) | 7,820 MiB | — |

- **Exact rebuild against Google's GGUF:** 1.035x generation, prefill level, 247.5 MiB less
  VRAM (`token_embd` Q4_0 instead of Q6_K), and the most context that fits rises from 28,672
  to 40,960 tokens at f16.
- **Context that fits:** f16 KV 40,960 without the drafter, 24,576 with it; q8_0 KV 65,536
  and 49,152. KV is 16 KiB/token plus a fixed 480 MiB for the sliding layers.
- **By conversation length** (no drafter): 28.6 / 26.8 / 26.3 tok/s at 0 / 8k / 16k tokens.
- **Drafter:** exact Q4_0 at 2 tokens 2.01x, acceptance 0.658; Q8_0 1.90x at the same
  acceptance and 202 MiB more; 1 token 1.62x, 3 tokens 2.02x. **Greedy text with the drafter
  differs from greedy text without it** (18 of 18 replies at 2 tokens), though it repeats run to
  run. **The answers do not change**: GSM8K and BFCL are level with and without it.
- **Output divergence from bf16** (CPU, wikitext-2): mean KLD 0.125 for the rebuild against
  0.543 for Google's file, same top token 89.1% against 71.8%. Not comparable with E4B's 0.0010:
  bf16 perplexity on that text is 312 for the 12B, which degrades without its chat format.
- **Answer quality** (`2026-10-16-accuracy-12b-3050oem/`, one request at a time at the serving
  setting): GSM8K 0.9598 with the drafter / 0.9629 without, BFCL 0.9500 / 0.9475, the drafter
  level on both and GSM8K 2.34x faster with it. **Above E4B on both**, paired by question:
  GSM8K +0.030 (95% +0.018 to +0.043), BFCL +0.0375 (+0.010 to +0.065), with the drafter.

**Not re-measured here:** flash attention, f16 KV over q8_0, thread count, concurrency.

## Inherited and still true here

- **Never pass `--no-mmap`.** `token_embd`'s input copy is read from the mmap.
  A test asserts it.
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
