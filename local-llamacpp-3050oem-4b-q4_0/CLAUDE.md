# CLAUDE.md — local-llamacpp-3050oem-4b-q4_0

Guidance for working inside this rig. The siblings are not layers; nothing is
imported across a rig boundary. Read this file before changing anything.

## What this rig is

`llama-server` from `ggml-org/llama.cpp`, driven directly, serving the exact
Q4_0 GGUF of Gemma 4 **E4B**-it (`xbill9/gemma-4-E4B-it-qat-q4_0-exact-gguf`,
SHA-256 `5462cc10…`) off the **RTX 3050 OEM** in the desktop `gpu`, with Google's
E4B drafter for speculative decoding.

**Forked 2026-10-13 from `local-llamacpp-3050oem-2b-q4_0`**, its E2B sister.
Slot 4 is the only slot that differs: same card, host, llama.cpp binary
(`77c4540ce`) and serving flags, measured the same day. That makes the pair the
cleanest E2B-against-E4B comparison here. Both rigs use port 8080 and together
need more than the card holds (3,840 + 6,662 MiB), so **only one serves at a
time**: stop one before `make serve` in the other.

Host, build and toolchain notes are the sister rig's; read its `CLAUDE.md`.

## Measurements — 2026-10-13

`benchmarks/runs/2026-10-13-first-light-3050oem/` and
`benchmarks/runs/2026-10-13-spec-mtp-3050oem/`.

| | E4B (this rig) | E2B (sister) |
| --- | ---: | ---: |
| generation, no draft (llama-bench tg128) | 60.5 tok/s | 116.0 |
| generation, six prompts, no draft (geomean) | 58.2 | 108.2 |
| generation, six prompts, 2-token draft (geomean) | **100.3** (1.72x) | 150.0 (1.39x) |
| prompt reading (pp512) | 2,246 t/s | 3,983 |
| VRAM at -c 131072 with drafter | 6,662 MiB | 3,840 |
| KV cache at 131,072 tokens | 2,088 MiB | 780 |

- **Drafter since 2026-10-14: exact Q4_0** (59 MB, from
  `xbill9/gemma-4-E4B-it-qat-assistant-q4_0-exact-gguf`), replacing the Q8_0
  conversion: 100.0 against 100.4 tok/s, same acceptance, GSM8K and BFCL level with
  no drafter at `--parallel 8`, 38 MiB less VRAM
  (`local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-drafter-q4exact-3050oem/`).
  The table above was measured with the Q8_0 drafter.
- **Drafting:** 2 tokens is best overall (1.54x story .. 2.21x code); 3 is
  level and best on code; 4 is worse. **Greedy output at 2 and 3 drafted
  tokens is byte-identical to no drafting** (18 of 18, ~25-token prompts; with 2k+ tokens of text ahead
  it differs, `local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-drafter-divergence-3050oem`), unlike E2B; at 4 it
  differs. Costs 232 MiB and 3.3% of prompt speed.
- **Per-layer table on the card:** prefill 1.023x / 1.014x (pp512 / pp2048),
  decode level, 1,492 MiB. Adopted.
- **By conversation length** (no draft): 60 / 56 / 47 / 39 / 30 tok/s at
  0 / 8k / 32k / 65k / 120k tokens, 0.52x E2B at every length.

**Answer quality against E2B** (`benchmarks/runs/2026-10-13-quality-e2b-vs-e4b-3050oem/`,
both Q4_0 builds, same flags, no drafter): Nimble suite 0.730 vs 0.683 (+0.048,
range +0.035 to +0.061), GSM8K 0.933 vs 0.898 (+0.035), BFCL 0.910 vs 0.925
(−0.015, range includes zero). Each build matches its TPU reference within range,
so the gap is the model's.

**The drafter costs nothing in answer quality as the rig serves** (`-c 131072 --parallel 1`,
`local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-accuracy-fullctx-3050oem/`): GSM8K 0.9295 with it against 0.9310 without, BFCL
0.9125 both; GSM8K finishes 2.23x faster with it, and one request at a time the drafted replies
match undrafted ones word for word on 1,200 of 1,319 GSM8K problems and all 400 BFCL questions.
Prompt cache off changes no score (`local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-accuracy-cache-3050oem/`).

**Concurrency on mixed chat prompts** (`local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-15-concurrency-mixed-3050oem/`, `-kvu`):
the drafter gains 1.61–1.72x for one request, 1.08–1.24x at 2–8, and loses at 32–64 (0.79–0.92x).
Without it output reaches 472 tok/s at 64 requests. Both arms fit 64 slots at `-c 32768`
(6,068 / 6,372 MiB); with the drafter `-c 65536 --parallel 64` does not fit
(`local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-concurrency-drafter-3050oem/`), with `-kvu` it does (6,936 MiB). With thinking on
the gain at 2–8 rises to 1.29–1.47x (`local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-15-concurrency-thinking-3050oem/`); 1 drafted
token loses less at 32–64 (0.87–0.98x) but gains less for one request (1.49–1.51x,
`local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-15-concurrency-draft1-3050oem/`). **KV cache layout for multi-request runs:**
per-slot caches (the default) halved decode speed at random on short requests; a shared cache
(`-kvu`) avoids that but slows as the total context held grows (0.64–0.79x per-slot at 64
requests of up to 640 tokens, `local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-15-concurrency-drafter-kvu-3050oem/`).
**With 1,000–3,500-token prompts the drafter pays up to 16 requests** (`local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-15-concurrency-longmixed-3050oem/`, `-c 131072 --parallel 32`):
1.54x (per-slot) and 1.61x (shared) for one request, 1.08–1.18x at 2–16, 0.99–1.05x at 32, acceptance
0.56–0.62. There per-slot caches are the faster layout (shared 0.61–0.66x at 32 requests) and showed no
random slowdown. Peak 7,741 MiB with the drafter and per-slot caches.

**Carried from the E2B rig, not re-measured here:** flash attention, f16 KV
over q8_0, thread count.

## Inherited and still true here

- **Never pass `--no-mmap`.** `token_embd`'s input copy is read from the mmap
  even with the per-layer table on the card. A test asserts it.
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
