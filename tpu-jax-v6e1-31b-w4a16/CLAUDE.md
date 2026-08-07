# CLAUDE.md — `tpu-jax-v6e1-31b`

## This is an artifact rig, not a serving rig

There is **no `server.py`, no MCP server, no skill, no plugin manifest, and no `tpu.env`** here, and
none is owed. This directory exists to give the 31B measurements a home that names the hardware they
were taken on. It holds two things:

- `benchmarks/runs/2026-07-31-gemma4-31b-v6e1/` — the measurements and the harnesses that produced them
- `docs/gemma4-31b-quirks.md` — the findings, organized by what they are true of

Do not scaffold it into a full rig to make it match its siblings. If a 31B serving rig is ever built,
it gets the full treatment then; the name is already correct for it.

Because there is no MCP server and no skill, none of the derived-name machinery in `NAMING.md`
applies — no `.mcp.json`, no `~/.cache/<rig>/`, no `make skill-install` destination. The directory
name still has to be right, because `benchmarks/rollup.py` discovers rigs by globbing for
`*/benchmarks/`, so this one appears in `ROLLUP.md` and gets its own generated `INDEX.md`.

## Provenance

Migrated from `~/tpu-jax-31b` on 2026-08-07. That repo predates the naming scheme (it has no
hardware slot) and carried a full JAX serving fork; only the measurement artifacts came across.
**The engine code did not** — `jax_engine.py`, `jax_e_model.py`, `ports/gemma4/*` and the test suite
stayed behind. Reports here reference those paths as they existed in the source repo; the line
numbers are a record of what was read at measurement time, not a pointer into this directory.

## What was measured

`google/gemma-4-31B-it-qat-w4a16-ct` on a **spot v6e-1** (`us-central1-a`), 2026-07-31/08-01, on the
pure-JAX engine. 60 layers, 50 sliding / 10 full, no PLE, no KV sharing — architecturally *simpler*
than E2B. Every difficulty was scale, never architecture.

`REPORT.md` measures the engine. `MODEL-INTEL.md` measures the model itself — norm structure,
`layer_scalar`, massive activations, sink reachability, measured W4A16 error.

## Reading these reports

- **Two conclusions in `MODEL-INTEL.md` §5 are retracted by §8**, and the retraction is kept in place
  rather than edited out. Scale dynamic range does not predict quantization damage; measured W4A16
  error is flat to 0.1% across every projection. Do not cite §5's ranking.
- `docs/gemma4-31b-quirks.md` **section D is the most reusable part of this rig** — seven ways the
  measurements lied, each with the wrong conclusion it produced. It generalizes past the 31B.
- Facts here that are true of the *model*, the *chip*, or the *stack* have been promoted to the root
  `MODELS.md` / `HARDWARE.md` / `QUANTIZATION.md`. Correct them there, not here. This rig keeps the
  measurement; the root files keep the conclusion.

## Caveats that travel with the numbers

- Activation figures come from **one** 20-token chat-templated prompt. The sliding-vs-full *ratios*
  are the robust part; absolute magnitudes move with prompt length and content.
- Everything describes the **W4A16 QAT** checkpoint. The bf16 variants may differ, particularly in
  the norm outliers.
- Greedy decoding on this model is **not bit-reproducible across `window_kv`** — different buffer
  sizes compile to different HLO. Hold it fixed across any A/B, and never judge correctness on
  random-token prompts (see D1).
