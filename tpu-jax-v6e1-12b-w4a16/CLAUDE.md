# CLAUDE.md — `tpu-jax-v6e1-12b`

## This is an artifact rig, not a serving rig

No `server.py`, no MCP server, no skill, no plugin manifest, no `tpu.env` — and none is owed. This
directory gives the 12B measurements a home that names the hardware they were taken on. Do not
scaffold it into a full rig to match its siblings.

`benchmarks/rollup.py` discovers rigs by globbing `*/benchmarks/`, so this one does appear in
`ROLLUP.md` and gets a generated `INDEX.md`. Nothing else in `NAMING.md`'s derived-name machinery
applies here.

## Provenance

Migrated from `~/tpu-jax-12b` on 2026-08-07. That repo predates the naming scheme (no hardware slot)
and carried a full JAX serving fork; only the artifacts came across — `benchmarks/runs/`,
`docs/12b-exploration-2026-07-31.md`, and the seven `ports/gemma4/jax_12b_*.py` harnesses.

**The engine code did not.** `jax_engine.py`, `jax_e_model.py` and the test suite stayed behind. The
`ports/gemma4/*.py` scripts here are kept as the record of what was run; they import from an engine
that is not in this directory and **will not execute as they stand**. Treat them as documentation of
method, not as a runnable suite.

`REPORT.md` links artifacts with absolute `file:///home/xbill/tpu-jax-12b/...` URLs pointing at the
source repo. Those are stale by construction; the files sit beside the report.

## What was measured

`google/gemma-4-12B-it-qat-w4a16-ct` on `ct6e-standard-1t` (v6e-1, 33.55 GB HBM), VM
`jax-gemma4-12b` in `europe-west4-a`, 2026-07-31, on the pure-JAX engine.

Note the zone: `europe-west4-a` worked here for **v6e**. The root `CLAUDE.md`'s constraint —
flex-start `v5litepod-1` only being accepted in `us-west4-a` — is about v5e and does not transfer.

## Two things to distrust in `REPORT.md`

The report is a working document and was not re-derived against `MODELS.md`. Two numbers do not
reconcile; **do not cite either without re-measuring**:

1. **The KV accounting double-counts `attention_k_eq_v`.** It states the 8 full-attention layers
   have V aliasing K at "0 extra weight bytes" and then charges them 16 KiB/token. One global KV
   head at `global_head_dim` 512 in bf16 is 8 KiB/token if V really is K, and 16 KiB only if both are
   stored. The stated total of 336 KiB/token inherits whichever is wrong.
2. **"320 KiB/token" for the sliding layers is not a per-token rate.** Those layers ring-buffer at
   `sliding_window = 1024`, so their cost is capped at ~327.68 MB per stream and stops growing.
   Adding a capped figure to an uncapped one and calling the sum "per token" overstates long-context
   KV.

This is the failure mode the root `CLAUDE.md` warns about under **Measurement**: a config flag being
accepted is not evidence it did anything. Cross-check against a boot-time allocation log.

## The result that does hold

**The 12B is a strict simplification of the E-series architecture, and needed zero code changes to
load.** Every MatFormer feature is switched off in `config.json` — no PLE, no KV sharing, no
double-wide MLP — and `config_from_hf` already resolved all of it because `pick()` tests
`is not None`, so the `0`s that disable those features survive. That was the main open question going
in and it is settled.

The digit-string output (`'111111'`) on bare prompts is **not an engine defect** — the HF PyTorch
reference does the same thing. It is a prompt-formatting requirement of the IT QAT checkpoint. With
`<bos>` plus the chat template, JAX and the reference agree at 100% exact token parity. Contrast the
31B, which recovers the scaffolding on its own from a bare prompt.
