# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Apache 2.0 Colab notebooks for the AI GDE **Marathon: JAX on TPU Tutorial** (deadline 2027-01-31).
Section-by-section source map and known gaps: @docs/OUTLINE.md

## Notebooks are build artifacts

**Edit `src/*.py`. Never edit `notebooks/*.ipynb` directly** — the next `make build` overwrites it.

Sources are jupytext `py:percent`. The build prepends the Apache-2.0 cell and the Colab badge and sets
`metadata.accelerator`, so none of those belong in a source file.

The `.ipynb` files are committed anyway, because that is what Colab serves from GitHub. `make check`
fails if they have drifted.

Cell ids are derived from position and content (`tools/build_notebooks.py`). This is load-bearing:
`nbformat` mints a random id for any cell lacking one, which makes the build non-reproducible and
`make check` fail every time. Do not "simplify" it back to popping the id.

## Commands

```bash
make build        # src/*.py -> notebooks/*.ipynb
make check        # fail if committed notebooks are stale; free and local
make verify       # execute every notebook on a real Colab TPU (v5e-1), fail on any error
make verify-v6e1  # same on v6e-1 -- currently rejected, see below
make lint         # ruff over src/ and tools/ (ruff.toml excludes the generated notebooks/)
make sessions     # list live Colab runtimes
```

`make verify` needs the Colab CLI: `pip install -r requirements-dev.txt` (plain pip is enough; uv is not
required). It is fine to run without asking, but it provisions real hardware — **a session left up keeps
spending compute units.** Check `make sessions` and `colab stop -s <name>` if a run is interrupted.

## Colab runtime facts (verified 2026-09-20)

- **Auth is `--auth adc`**, wired into `tools/verify_on_tpu.py`. The CLI's own default is `oauth2`, which
  blocks on a browser consent code and is unusable from a script. ADC must be an `authorized_user`
  credential — a service account cannot provision a Colab runtime, because runtimes belong to a Google
  user and bill against that user's compute units. `check_adc()` enforces this before provisioning.
- **The target chip is v5e-1**, in `notebooks.json` and as the `verify` default. v6e-1 is rejected on this
  account ("Backend rejected accelerator 'V6E1'") and `make verify-v6e1` exists only for the day that
  changes. Write prose and capacity claims against v5e-1.
- Measured on the v5e-1 runtime: **JAX 0.7.2, one `TPU v5 lite` device, 16.91 GB HBM.**
- **The source measurements in `~/tpu-jax` were taken on v6e-1 (32 GB).** Keep citing them as v6e-1
  results — they are not v5e-1 numbers and must not be relabelled. What changes is the chip the reader
  is on, which is why cells re-measure rather than quote. Speedup ratios (donation, int8 KV) should
  carry over; absolute capacity figures will not.
- `jupyter-kernel-client` is pinned to `0.15.0` — see the comment in `requirements-dev.txt`. Unpinning
  breaks `colab exec` while leaving provisioning working, so the failure looks like a notebook problem.
- **Never retry `colab new` after a timeout.** A 503 is a refusal and nothing was allocated. A
  `ReadTimeout` is silence, and the runtime may exist anyway — retrying builds another. On 2026-09-20
  that produced three orphaned v5e-1 runtimes, which `colab sessions` shows as `[?]` rows and which
  `colab stop -s` **cannot kill**, because it resolves names from `~/.config/colab-cli/sessions.json`.
  They bill until they idle out or a human ends them at
  colab.research.google.com → Runtime → Manage sessions. `provision()` in `tools/verify_on_tpu.py`
  encodes this: it retries a 503 only, and bails with instructions on a timeout or
  `TooManyAssignments`. If verification starts failing for no reason, check for `[?]` rows first.

## Writing notebook content

- **Re-measure, do not quote.** A cell should compute the number on the reader's own chip. Cite the
  source measurement as context ("measured 1.60-1.62x on v6e-1"), never hardcode it as the result.
- **Notebook 01 must need no gated checkpoint and no HF token.** Anyone should be able to run it.
  Notebook 02 reads its token from `google.colab.userdata`, never a literal — this repo goes to GitHub.
- Colab TPU runtimes are `v5e1` and `v6e1`, the same silicon the source rigs used. That is why these
  numbers are reproducible and it is worth saying so in the prose.
- Colab gives **one chip**. Nothing here can measure multi-device sharding. Emulated-device examples must
  be labelled as emulation.
- **Pallas is out of scope** — handled separately with Rubens. No section may depend on it.

## Source material

`~/tpu-jax` is the substantive source repo (engine, loader, 12 CPU tests, 4 benchmark runs under
`benchmarks/runs/`). `~/tpu-jax-{v5e1-2b,v6e1-2b,4b,12b,26b,31b,inf2}` are **near-identical forks** — do
not read them as independent sources or cite them separately. `~/gemma4-dev` is the 30-rig monorepo;
rig names follow its `NAMING.md`.

## Repo conventions

- Solo project, commit to `main` directly.
- `runs/<date>/` holds executed notebooks as evidence a version ran clean — these are committed.
- `notebooks/*_output.ipynb` is gitignored; one appearing there means a verify run was interrupted.
- `notebooks.json` holds the GitHub slug the Colab badges point at. Changing it requires `make build`.
