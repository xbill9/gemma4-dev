# CLAUDE.md — `tpu-jax-v6e1-26b`

## This is an artifact rig, not a serving rig

No `server.py`, no MCP server, no skill, no plugin manifest, no `tpu.env` — and none is owed. Do not
scaffold it into a full rig to match its siblings.

`benchmarks/rollup.py` discovers rigs by globbing `*/benchmarks/`, so this one appears in `ROLLUP.md`
and gets a generated `INDEX.md`. Nothing else in `NAMING.md`'s derived-name machinery applies.

## The hardware slot is a target, not a measurement

**Read this before citing anything here.** `v6e1` names what the port was *aimed at*. The 26B was
**ported and verified against the HF reference on CPU, and has never been run on a TPU.** Every
memory figure — the 15.25 GB resident projection, the ~18 GB left for KV — is arithmetic from
`quantized_bytes`, not an allocation log. The prefill ceiling is unknown.

This makes the rig name a weaker claim than its siblings', where `v6e1` records the chip the numbers
came off. It is still the right name — the port targets a v6e-1 and the open items are all "run it
there" — but do not read a v6e-1 measurement into a file that has none. `REPORT.md` marks its
projections explicitly; keep that distinction when quoting it.

## Provenance

Migrated from `~/tpu-jax-26b` on 2026-08-07 (predates the naming scheme — no hardware slot). Only
artifacts came across; the engine, `ports/gemma4/jax_26b_port.py`, `jax_q4_0.py` and
`tests/test_moe_parity.py` stayed behind. Reports reference those paths as they were at the time.

`docs/gemma4-26b-quirks.md` was **extracted** from that repo's `docs/gemma4-quirks.md`, which had
grown to cover both E2B and the 26B. Only the 26B half (§15–22) came here — section numbers are
preserved so `REPORT.md`'s cross-references still resolve. The E2B half stayed with
`tpu-jax-v5e1-2b/docs/gemma4-quirks.md`, whose copy is **ahead** of the one in the source repo (it
has a whole Part II on serving-path quirks, and corrects `sliding_window` coverage from "32 of 35
layers" to "28 of 35"). Do not re-merge the old combined file over it.

## What makes the 26B the odd one out

Twice over: it is the **only sparse checkpoint** in the Gemma 4 family, and the **only size with no
`-w4a16-ct` release** (enumerated from the Hub 2026-07-31 — do not assume the suffix set is uniform
across sizes). The only usable export is `-qat-q4_0-unquantized`, 51.61 GB of BF16 against a v6e-1's
33.55 GB.

It fits anyway because **"unquantized" describes the container, not the values** — these are QAT
weights already sitting on a Q4_0 grid, verified by range-reading the shards. Group size 32 is
*measured*, not assumed: all 256 sampled groups of 32 land on a 4-bit grid, and group size 64 fails
the same test.

## The bug worth carrying forward

**The router reads the raw post-attention residual; the experts read a normalized copy.** Passing the
normalized tensor to both is the obvious simplification and it is wrong — the router opens with its
own scale-less RMSNorm, which does not cancel against a learned-weight one, so it reweights the
channels the router sees and changes which experts fire.

Cost of getting it wrong: **0.36 relative error with every unit test green.** Router parity passed.
Expert parity passed. Only end-to-end comparison against `Gemma4TextModel` in float32 caught it.

That is this repo's recurring failure mode stated exactly — parity assertions between two of our own
code paths cannot see an assumption both paths share. It is why `moe_block_jax` takes `router_in` and
`expert_in` as separate arguments, which looks redundant until you know why.

## Open items, carried from `REPORT.md` §6

1. Run the port on a v6e-1 — confirm 15.25 GB resident, get real prefill/decode numbers.
2. Measure the gather/dense crossover rather than trusting the byte-counted T=16 threshold. This
   engine has been wrong before about bytes predicting time.
3. Expert-sorted prefill dispatch — prefill currently does **16x the expert FLOPs** (128/8), optimal
   in bytes moved and wasteful in compute. Land it behind a parity test with the overflow count
   asserted, not inferred: capacity-padded dispatch *drops tokens* when an expert is oversubscribed,
   and a silent drop is precisely the failure mode this engine keeps producing.
