---
name: new-rig
description: Create a new accelerator rig directory in this monorepo, following the NAMING.md four-slot scheme and forking the right template for the runtime.
disable-model-invocation: true
---

Create a new rig. `$ARGUMENTS` is the intended rig name or a description of the hardware/runtime/model — if it is empty or ambiguous, ask before doing anything.

## 1. Name it

Read `NAMING.md` in full first. Do not fill a slot from memory or by pattern-matching a sibling directory.

The name is `<platform>-<runtime>-<hardware>-<model>` — four positional lowercase slots, exactly three hyphens, no hyphen inside a slot. Confirm each slot against NAMING.md's permitted values, then check the name is unique against the existing directories.

Report the four slots and your reasoning to the user and get confirmation before creating anything.

## 2. Pick the template by runtime

The forks have visibly diverged, so the template matters:

- **`vllm` runtime** → fork `tpu-vllm-v5e1-2b`. Flat layout: `pyproject.toml` (ruff + mypy), `test_agent.py` at the rig root, no skill-snapshot machinery, `tpu.env` + `mcp-run.sh`.
- **`jax` or `pytorch` runtime** → fork the nearest sibling. **Ask the user which one** — `tpu-jax-v5e1-2b`, `tpu-pytorch-v5e1-12b`, `tpu-pytorch-v5e1-2b`, `tpu-pytorch-v6e1-2b`, and `tpu-pytorch-inf2-2b` differ in real ways. These use the skill-snapshot layout: `refresh_skill.py`, `.claude/skills/`, `skills/`, `dist/*.zip`, `tests/`, `make skill|skill-install|skill-package`.

Copy the template directory, then remove artifacts that must not travel: `dist/`, `.mcp.json`, `.env`, `__pycache__/`, `.ruff_cache/`, `.pytest_cache/`, and **`benchmarks/`** — benchmark JSON already travelled with the existing forks and several rigs now carry numbers measured on hardware they are not. Do not repeat that.

## 3. Set the real config

Set `tpu.env` (or `.env` via `set_env.sh`, matching the template's convention) to the new rig's actual values: `MODEL_NAME`, `ACCELERATOR_TYPE`, `TENSOR_PARALLEL_SIZE`, zone/region.

**Never copy a directory-name slot into a config value or a CLI flag.** `v5e1` is the human-facing name; the VM startup path wants `v5litepod-1`. Get the real accelerator string from the template rig's env file or from gcloud, not from the new directory's name.

Chip count in slot 3 is topology, not tensor-parallel size. They coincide at `v5e1`/`TP=1` but are separate settings — set `TENSOR_PARALLEL_SIZE` deliberately.

Also fix the `server.py` module-level defaults. The `tpu-pytorch-v5e1-2b` and `tpu-pytorch-v6e1-2b` forks still default to `v6e-8` / `gemma-4-31B-it` / `TP=8`, contradicting their own names — don't inherit that.

## 4. Rename the plugin everywhere

NAMING.md requires the plugin name to equal the directory name. Update:

- the new rig's `.claude-plugin/plugin.json` and `.claude-plugin/marketplace.json`
- the **monorepo root** `.claude-plugin/marketplace.json` — add an entry for the new rig; this is the marketplace `/plugin` actually reads

Then sweep for leftovers from the template:

```
grep -rn '<template-rig-name>' <new-rig>/ --exclude-dir=.git --exclude-dir=dist
```

Fix docs, JSON-schema `$id`s, and hardcoded `/home/xbill/<rig>/…` paths (these appear in `ports/**` and in `ports/**/*_test.py`).

## 5. Regenerate and verify

For a skill-snapshot rig: `make skill`, then `make skill-package` if `dist/` is meant to ship.
For any rig with `generate_tools_doc.py`: `make tools`.

Then from the new rig directory:

```
make lint
make test
claude plugin validate .
```

`make lint` in the skill-snapshot rigs lints a hardcoded file list in the Makefile — if the new rig added or renamed a top-level module, add it to that list or it is silently unlinted. Tests are `unittest`, not pytest.

## 6. Record it

Add a row to the **Current inventory** table in `NAMING.md`, and add the rig to the variants table in `README.md`. Both are currently stale about `tpu-pytorch-inf2-2b` and `tpu-pytorch-v6e1-2b`; mention that to the user if you notice it, but don't fix it unasked.

Leave the commit to the user unless they ask. Remember the git root is the monorepo, not the rig.
