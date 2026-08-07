---
name: sync-generated
description: Regenerate a rig's derived artifacts after editing server.py — skill snapshots, GemmaTools.md, the packaged zip — and verify the two marketplace.json files agree. Use after adding, removing, or changing an MCP tool.
---

Several files in each rig are **generated from `server.py` and friends**. Editing the source without regenerating leaves the snapshots stale; editing a snapshot directly means the change is silently lost on the next `make skill`.

`$ARGUMENTS` may name a rig. If it is empty, work out which rig from the files that were just changed; if that is ambiguous, ask.

## 1. Check nothing was edited in the wrong place

Before regenerating, confirm no hand-edits are sitting in generated trees — regenerating would destroy them:

```
git status --porcelain <rig>/.claude/skills <rig>/skills
```

`.claude/skills/**` and `skills/**` are copies. If either shows modifications that did not come from `make skill`, **stop and tell the user** — the edit needs to move to the rig-root source (`server.py`, `project-setup.sh`, `requirements.txt`, `tpu.md`) first. Do not regenerate over it.

## 2. Regenerate

From the rig directory:

- **`make skill`** — runs `refresh_skill.py` to rebuild `.claude/skills/<rig>-management/` from `server.py`, then syncs the plugin copy in `skills/`. Skill-snapshot rigs only (`tpu-jax-*`, `tpu-pytorch-*`); `tpu-vllm-v5e1-2b` has no skill machinery.
- **`make tools`** — regenerates `GemmaTools.md` from `mcp.list_tools()`. `tpu-vllm-v5e1-2b` only.
- **`make skill-package`** — rebuilds `dist/<rig>-management-skill.zip`. This zip is **committed on purpose** so the raw-GitHub-URL install works; rebuild it whenever the skill changed.

If a tool was added or removed, `get_help` is generated too — it comes from the same `mcp.list_tools()` call, so `make skill` / `make tools` covers it. Ground truth for what tools exist:

```
grep -n '^@mcp.tool' <rig>/server.py
```

Compare that list against what landed in the regenerated `SKILL.md` / `GemmaTools.md` and report any mismatch.

## 3. Verify the two marketplaces agree

The marketplace `/plugin` actually reads is the **monorepo root** `.claude-plugin/marketplace.json`. Each rig also has its own copy, which matters only if that rig is published standalone. They drift.

Read both and check the entry for this rig matches — name, description, source, version. NAMING.md requires the plugin name, the MCP server name, and the skill-name stem to all equal the rig's directory name; flag any rig still carrying a pre-monorepo or shared name. The shared `tpu-management` skill and `tpu-devops` server were resolved on 2026-08-06 — the one remaining `tpu-management` is a stale duplicate inside `tpu-pytorch-inf2-2b`.

Then:

```
claude plugin validate .
```

## 4. Lint and test the source, not the copies

```
make lint
make test
```

`make lint` lints a hardcoded file list in the Makefile — if `server.py` gained a new sibling module, add it to that list or it is never linted. Tests are `unittest`, not pytest.

## 5. Report

Tell the user exactly which generated files changed (`git status --porcelain`), and whether the tool list, the two marketplace files, and the packaged zip are now consistent. Do not commit unless asked.
