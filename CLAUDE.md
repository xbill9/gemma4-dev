# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A monorepo of **accelerator rigs** for serving Gemma 4. Each rig serves one checkpoint on one hardware shape through one runtime, and ships a single-file FastMCP server (`server.py`) exposing a devops agent that provisions capacity via `gcloud`, starts the model server, and does SRE diagnostics against the endpoint.

**A rig's MCP server is named after its directory**, and that name is the key it is registered under, so it prefixes every tool: `mcp__tpu-vllm-v5e1-2b__find_tpu`. Every rig used to register as `tpu-devops`, so with two loaded you could not tell which rig a call would reach — and a user-scope `tpu-devops` silently shadowed any rig with no `.mcp.json`. The default is derived (`RIG_NAME` in the vLLM rig, a literal matching the directory elsewhere); `MCP_SERVER_NAME` overrides it, and `project-setup.sh --server-name` sets both the registered key and what the server advertises. Registration lives in four places per rig — `.mcp.json`, `.claude-plugin/plugin.json`, `.codex/config.toml`, and `.claude/settings.local.json`'s `enabledMcpjsonServers` — keep them agreeing.

The same rule now covers the **skill name** (`<rig>-management`, overridable with `SKILL_NAME`) and the **zone-status cache** (`~/.cache/<rig>/tpu_zones_status.md`, overridable with `TPU_ZONES_STATUS_FILE`). Both were shared across rigs and both were destructive: `make skill-install` `rm -rf`s its destination, and `find_tpu` skips zones another rig recorded as failed even though the rigs request different accelerator types. `NAMING.md` holds the table.

One thing deliberately keeps the old shared name: the inf2 rig's AWS `ManagedBy=inf2-devops` tag, because renaming it would orphan already-tagged instances.

**The rigs are siblings, not layers.** Nothing is imported across a rig boundary. They share ancestry and have diverged — do not assume a pattern in one rig holds in another; read the rig you are in. Each rig has its own `CLAUDE.md`; read it before working inside that rig.

Serving rigs: `tpu-jax-v5e1-2b`, `tpu-pytorch-v5e1-12b`, `tpu-pytorch-v5e1-2b`, `tpu-pytorch-v6e1-2b`, `tpu-pytorch-inf2-2b` (AWS Inferentia2, not GCP), `tpu-vllm-v5e1-2b` (the live-demo rig), `tpu-vllm-v6e1-2b`.

**Artifact rigs — measurements only, and a different thing entirely.** `tpu-jax-v6e1-12b-w4a16`, `tpu-jax-v6e1-26b-q4_0`, `tpu-jax-v6e1-31b-w4a16`, and the five `gpu-vllm-l4-*` rigs have **no `server.py`, no MCP server, no skill, no plugin, no `tpu.env`**, and none is owed. They give results a home that names the hardware they came off, for sizes and chips no serving rig covers. Do not scaffold them into full rigs; do not look for an engine in them — the JAX code stayed in `~/tpu-jax-{12b,26b,31b}` and the harnesses they carry will not run as they stand. `benchmarks/rollup.py` still discovers them, because it globs `*/benchmarks/`. `NAMING.md` has the rule.

**The `gpu-vllm-l4-*` rigs came from a tree that duplicated its own artifacts**, and their provenance is weaker than anything else here. `~/gemma4-tips` + `~/gemma4-tips-aws` held 82 `benchmark_report*.md` files that reduce to **20 unique**, and 109 CSVs that reduce to 32; one 12B report sat in 13 directories spanning 2B through 31B, and `g2-48-26B-qat-L4` held the *31B* report. Only the 10 reports whose own `Model:` and `Endpoint:` lines agree were migrated. **Never read a model or a chip off a `~/gemma4-tips` directory name**, and prefer a report's `Endpoint:` line over its parenthetical host label. Each rig's `CLAUDE.md` repeats this, because people read one rig rather than the tree.

`tpu-pytorch-inf2-2b/` and `tpu-pytorch-v6e1-2b/` are real rigs that are not committed yet, and the three artifact rigs are new as of 2026-08-07. `README.md` and `NAMING.md`'s inventory were corrected for all of them on 2026-08-07.

## Canonical root references

Four facts-not-code files at the root, each scoped to one axis that does **not** vary by rig. Read the relevant one before deriving its numbers yourself, and correct it there rather than restating it in a rig:

- **`@MODELS.md`** — checkpoint properties: layer structure, attention/head shape, KV cost per token, bf16 weight footprints. Same whatever serves them.
- **`@HARDWARE.md`** — accelerator properties: usable HBM, bandwidth, and **which numeric formats the MXU natively supports**. v5e and v6e have no native fp8 (int8 is the only low-precision compute win); v7/Ironwood is the first that does, so quantization conclusions do not carry forward across generations.
- **`@QUANTIZATION.md`** — what vLLM + tpu_inference actually support for Gemma 4: which routes are reachable from the JAX model path, which are dead, and how to enable qwix. Stack properties, so they hold across hardware slots.
- **`@NAMING.md`** — how any of the above is spelled in a directory name.

**File by what a fact is true of, not by where it was measured.** A finding from one rig that describes the model, the chip, or the serving stack belongs in the matching root file; only the measurement itself stays in that rig's `benchmarks/runs/`.

## Naming

Rig directories are `<platform>-<runtime>-<hardware>-<model>[-<variant>]` — four positional lowercase slots, plus an optional fifth naming the **weight encoding** when it isn't the reference build (`w4a16`, `q4_0`, `int4`). Three hyphens without a variant, four with one; most rigs have none. Not `qat` (that's a training procedure, not an encoding) and not runtime params like `kv_cache_dtype` — those live in `tpu.env`.

**Read `@NAMING.md` before naming, renaming, or adding a rig.** It holds the permitted value for every slot; do not fill one in from memory or by pattern-matching a sibling.

The directory name is a claim about the rig's `tpu.env`, **not configuration**. `v5e1` is for humans; gcloud wants `v5litepod-1`. **Never copy a slot value into a CLI flag** — read the actual value out of the rig's env file. Benchmark artifacts use a different, date-first scheme documented in the same file.

## Environment and config

- **`tpu.env` is the source of truth** where it exists (`tpu-pytorch-v5e1-12b`, `tpu-vllm-v5e1-2b`). It is committed and deliberately **not** gitignored — never add `*.env` to a `.gitignore`. A real environment variable always wins over it (dotenv doesn't overwrite, `mcp-run.sh` exports only unset keys, Makefiles use `?=`).
- **`.env` is generated** by `set_env.sh`, is mode 0600, gitignored, and contains live API keys. Don't commit or echo it.
- **`.mcp.json` is gitignored** (it embeds the GCP project id). Regenerate with `project-setup.sh`; never commit it.
- Env values are spelled inconsistently across siblings **on purpose in one case and by drift in others**: `ACCELERATOR_TYPE=v5litepod-1` is the value the VM startup path wants, while `v5e` is the directory-naming form. Do not normalize one to the other, and do not assume a sibling's spelling applies. Read the rig's own env file.
- Known drift, not a rule to preserve: `tpu-pytorch-v5e1-2b/server.py` and `tpu-pytorch-v6e1-2b/server.py` default to `v6e-8` / `gemma-4-31B-it` / `TP=8`, contradicting their own directory names and `.env`.

## Commands

The root `Makefile` only fans out into every rig (`make clean|test|lint|install|deploy`). There is no real top-level build — real work happens per rig.

- **Tests are `unittest`, never pytest**, in every rig. `python3 -m unittest discover -s tests -v` in the skill-snapshot rigs; `python test_agent.py` in `tpu-vllm-v5e1-2b`. Stray `.pytest_cache/` dirs exist but are not the sanctioned path.
- **`make lint` lints a hardcoded file list** in the skill-snapshot rigs (e.g. `ruff check server.py refresh_skill.py jax_engine.py …`), then `bash -n` on the four shell scripts. A new top-level module is silently unlinted until it is added to that list. In `tpu-vllm-v5e1-2b`, `make lint` only *checks* formatting — `make format` writes.
- **`make install` is a no-op in four of six rigs** — the target exists with no recipe. Only `tpu-vllm-v5e1-2b` actually pip-installs, despite what `README.md` says. Install manually with `pip install -r requirements.txt`.
- `claude plugin validate .` checks a plugin manifest.

Use the **system `python3`**. Do not create a virtualenv. If a dependency is missing, warn with the `pip install -r requirements.txt` command instead of installing or building an env.

## Generated files — never hand-edit

- `.claude/skills/**` and `skills/**` are **generated copies** of the rig-root `server.py` etc. Edit the source, then `make skill`. Hand-edits are lost.
- `GemmaTools.md` and the `get_help` tool are generated from `mcp.list_tools()`. After adding or removing a tool, run `make tools`. Ground truth for what exists: `grep -n "^@mcp.tool" server.py`.
- `dist/*-skill.zip` is committed so the raw-GitHub-URL install works — rebuild with `make skill-package` when the skill changes.
- The marketplace `/plugin` reads is the **monorepo root** `.claude-plugin/marketplace.json`. Each rig has its own copy that only matters if published standalone. **Keep both in sync**; `NAMING.md` requires the plugin name to equal the directory name.

## Code style

- Every subprocess call goes through `run_command(cmd: list[str])` using `asyncio.create_subprocess_exec`. **Never `shell=True`.**
- MCP tools are `async def` returning markdown strings with emoji status prefixes (`✅`, `❌`, `📡`).
- Existing code uses `Optional[str]`, not `X | None` — match the surrounding file.
- Don't assume `pandas` is installed; prefer stdlib `csv`/`json` in analysis scripts.
- Only two rigs pin a ruff config, and they pin different ones: `tpu-vllm-v5e1-2b/pyproject.toml` (py313, line-length 120) and `tpu-pytorch-inf2-2b/ruff.toml` (py311, line-length 110). The rest run bare `ruff check` on implicit defaults.

## Testing discipline

Unit tests are offline: they mock the whole `mcp` module and the GCP clients **before** importing `server`. Keep cloud, subprocess, and network boundaries mocked. Because `mcp` is a `MagicMock`, anything calling `mcp.list_tools()` needs an explicit `AsyncMock` patch.

`ports/**/*_test.py` are not picked up by `unittest discover -s tests` and carry hardcoded `/home/xbill/<rig>/…` paths.

## Cloud gotchas

- **Flex-start `v5litepod-1` is only accepted in `us-west4-a`** (verified 2026-08-04). `europe-west4-a`/`-b` reject it at the API regardless of quota — the provisioning model is the blocker, not capacity. The `europe-west4-a` defaults in the `.env` files therefore cannot provision these rigs.
- **Don't destroy a queued resource unless asked.** Flex-start capacity can take up to 2 hours to come back.
- `create_tpu_queued_resource` is non-destructive, but **`manage_queued_resource` deletes every QR in the zone that isn't the named primary.** Keep that split.
- **Never hardcode an endpoint** — discovery is dynamic (first `ACTIVE` QR → node → external IP → `:8000`). A QR's node id is derived, not configurable: `<resource_id>-node`.
- `startup_script_template.sh` is rendered through `str.format()`. Any stray literal `{`/`}` (brace expansion, `${VAR}`, JSON) raises at format time and breaks the deploy — escape as `{{`/`}}`. Never reintroduce a `{hf_token}` placeholder: the rendered script is uploaded as VM instance metadata and fetches `hf-token` from Secret Manager at boot instead.
- `tpu_zones_status.md` is **mutable state, not documentation** — `find_tpu` rewrites it in place to skip known-bad zones.
- Raw `/v1/completions` returns an empty completion on `-it` models. `make query` and `benchmarking_suite.py` use it, so empty output there is expected, not a broken deploy. `server.py` uses `/v1/chat/completions`.
- `init.sh` blocks on `read` in its error path — never run it non-interactively.

## Measurement

A config flag being accepted is not evidence it did anything — cross-check a benchmark against an absolute physical bound, not against another config. Most tests here are parity assertions between two of our own code paths, so an assumption both paths share is invisible to all of them.

Benchmark JSON travelled with the forks: several rigs carry numbers measured on hardware they are not. A report's `<hw-short>` is the hardware *measured*, not the rig hosting it.

**Benchmarks are standardized across rigs.** `benchmarks/serving-report.schema.json` and `benchmarks/README.md` at the monorepo root are canonical; every rig holds a synced copy under `<rig>/benchmarks/`, and each rig's own `reports/` and `runs/` stay in that rig. Root targets: `make benchmarks-sync` (push schema + README out), `make benchmarks-rollup` (regenerate `benchmarks/ROLLUP.md` and each rig's `INDEX.md`), `make benchmarks-validate` (validate every report), or `make benchmarks` for all three. **Edit the root copies, never a rig's** — the sync overwrites them. `ROLLUP.md` and `INDEX.md` are generated; a hand-maintained per-rig index is how the old run-index table came to claim other rigs' results.

Schema is at **1.1**: `throughput.sweep[]` gained `input_len`/`output_len` (so a 2-D context × concurrency sweep fits in one report) and `status` (`ok`/`infeasible`/`failed`, so cells that cannot exist on the hardware are recorded rather than dropped). 1.0 reports still validate.

**Count coverage from the aggregator, not from marker files.** The `2026-07-25-vllm-sweep-v6e1` REPORT.md claimed 42 infeasible cells because it counted `.skip` files; 14 were stale markers left beside real results by a fixup re-run, and the true figure is 28. Corrected 2026-08-06 in all four copies.

## Git

The git root is the **monorepo**, not the rig — `git add .` from inside a rig stages only that subdirectory. `origin` is `github.com/xbill9/gemma4-dev`.

**Commit straight to `main`. No branches, no pull requests.** This is a small single-maintainer project and the review flow a branch buys costs more than it returns. Do not branch before committing, even though that is the usual reflex on a default branch; if a branch already exists from earlier work, fast-forward `main` onto it and delete it.

That is about *where* commits land, not how carelessly they are made. **The tree routinely carries several unrelated bodies of in-progress work at once, and some of it is already staged** — on 2026-08-07 it held a 62-entry staged skill rename, an unstaged `NAMING.md` rework, and the v5e rig's unstaged benchmarks reorganization, all at the same time. So `git add -A` and `git commit -a` will sweep someone else's half-finished work into your commit, and `git commit -- <paths>` does not save you: it commits working-tree content and bypasses partial staging. Read `git status` before staging. When the index is already dirty, build the commit through a temporary index (`GIT_INDEX_FILE` + `read-tree` + `write-tree` + `commit-tree`) instead of disturbing the real one, and stage a single hunk of a file others are editing with `git hash-object -w` plus `git update-index --cacheinfo`.

`.claude/settings.json` is committed and shared; `.claude/settings.local.json` is ignored. There is no blanket `*.log` ignore at root — logs under `benchmarks/runs/**` are committed as the record of a measurement. `tpu.md` and `pytorch.md` are private, not-for-publication docs.

## Agent-instruction files

Several rigs carry `CLAUDE.md`, `AGENTS.md`, and `GEMINI.md` covering the same ground, maintained by different tools and already drifted in places. There is no generator. **`CLAUDE.md` is the correct one** where they disagree; a convention change needs to land in all copies manually.
