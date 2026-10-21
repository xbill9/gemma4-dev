# CLAUDE.md — local-llamacpp-1650ti-12b-q4_0

Guidance for working inside this rig. The siblings are not layers; nothing is
imported across a rig boundary. Read this file before changing anything.

## What this rig is

`llama-server` from `ggml-org/llama.cpp`, driven directly, serving the exact
Q4_0 GGUF of Gemma 4 **12B**-it (`xbill9/gemma-4-12B-it-qat-q4_0-exact-gguf`,
SHA-256 `038337ee…`) off the **GTX 1650 Ti Max-Q** (TU117, sm_75, 4 GiB) in the laptop
`yogagpu`, with the exact Q4_0 build of Google's 12B drafter for speculative decoding
(`xbill9/gemma-4-12B-it-qat-assistant-q4_0-exact-gguf`, `f22f6366…`). Text only.

**Forked 2026-10-21 from `local-llamacpp-3050oem-12b-q4_0`**: the same GGUF, drafter and
llama.cpp commit (`7182ef71e`) on the RTX 3050 OEM. Slot 3 is the only slot that differs, but the
host differs too (i7-10750H laptop against i7-5820K desktop), so a ratio between the two rigs is a
card-and-host result. Shares port 8080 and the card with the 1650 Ti E2B and E4B rigs; one at a time.

**The weights do not fit the card.** 6.24 GiB in 48 layers of about 133 MiB; `N_GPU_LAYERS` of them
go on the card and the rest run on the CPU, which sets the speed. This is the 3050 31B rig's
situation, not the 3050 12B's: there the whole model is on the card and the context cache is the
limit; here the layers are.

## Verdict — 2026-10-21

**Serves, too slowly for interactive chat.** With the drafter a short question generates at
7.4–10.1 tok/s (15–19 s for 128 tokens); prompt reading is about 60 tok/s, so a 4,096-token document
takes 100 s before the first word and a 16,384-token one 5 minutes. The 3050 sister, all layers on
its card, generates at 56.4 tok/s and reads prompts at 987. On this card E4B is the interactive model.

## Measurements — 2026-10-21

`benchmarks/runs/2026-10-21-first-light-1650ti/REPORT.md` (speed, fit, drafter, long context) and
`2026-10-21-accuracy-12b-1650ti/REPORT.md` (answers).

- **Layers on the card:** each adds about 0.1 tok/s; tg64 3.63 with none, 5.43 at 20, 5.64 at 21,
  5.98 at 24 (`-ub 128`). Prompt reading stays at 58–61 tok/s whatever is on the card and whatever
  `-ub` (512 / 1024 / 2048 tried); a larger `-ub` only costs layers.
- **Exact rebuild against Google's file:** level speed at the same layers; the rebuild's 0.24 GiB
  less buys one to two more layers.
- **Most layers that load** (`-ub 512`), with / without the drafter: q4_0 cache 22 / 24 at 8,192,
  20 / 23 at 32,768, 14 / 19 at 131,072; f16 20 / 23 at 8,192, 19 / 21 at 32,768.
- **Drafter** (three rotated rounds, `-ngl 20 -c 32768`, q4_0): 1 / 2 / 3 drafted tokens 1.50x /
  1.15x / **1.56x** (8.5 tok/s against 5.4); no drafter at 23 layers only 1.09x. 2 tokens below 1
  and 3 on every prompt, as on the 3050's 31B. Greedy text differs from no drafter in 18 of 18, and
  moving layers between card and CPU changes greedy text by itself.
- **Long conversation:** drafter story / code at 0 / 4k / 16k / 30k tokens held 7.4 / 5.8 / 4.1 /
  3.2 and 10.1 / 9.3 / 6.5 / 5.0; no drafter 5.6 / 4.9 / 4.3 / 3.6. From 16k the drafter is behind on
  story and level on summary; prompt reading falls to 38 tok/s at 30k.
- **Threads:** 4–8 level, 12 slower (3.9–4.1 against 4.7–5.0).

## Measuring on this host

- **It is a laptop in use.** CPU and GPU share power and cooling: the GPU clock read 1035–1875 MHz
  during the runs, and the same setting generated 11–17% slower while a browser tab was busy. A fixed-order single pass
  drifted 1.14x between its first and last control arms (`ab-c32768/`, kept, not used). **Compare
  settings in rotated rounds** (`spec_ab_rounds.py`) and cool the CPU package, not only the GPU.
- Load average is not a clean measure of outside load here: llama-server's own threads raise it to
  6–14.
- Some failed loads abort with a core dump instead of a clean out-of-memory; treat both as "does
  not fit".

## Inherited and still true here

- **Never pass `--no-mmap`.** `token_embd`'s input copy is read from the mmap.
  A test asserts it.
- Gemma 4 reasons; `REASONING=off` is the server default and `query_model`
  defaults to `max_tokens` 1024. Use `/v1/chat/completions`, not
  `/v1/completions`.
- `sweep.py`: `decode_tps` counts chunks, not tokens; verify the prompt cache
  was defeated via `/metrics`; llama.cpp splits `--ctx-size` across slots.
- `HOST=127.0.0.1`, as in the 1650 Ti E2B/E4B rigs.

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
- **Never pass a bare `-j` to the llama.cpp build**: it OOMs this 15 GiB host. `BUILD_JOBS=6`.

## Canonical root references

`@MODELS.md`, `@HARDWARE.md`, `@QUANTIZATION.md`, `@NAMING.md`,
`@RIG-ANALYSIS.md`. Read them before deriving their numbers here, and correct them
there.
