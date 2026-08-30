#!/usr/bin/env python3
"""Refresh generated MCP files in the bundled local-CPU JAX skill.

SKILL.md is a hand-written SOURCE that sits in the same tree and is deliberately
NOT regenerated here — `rm -rf .claude/skills` destroys it permanently.
"""

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Skill name carries the rig directory so sibling rigs do not collide on one
# ~/.claude/skills/<name>; the Makefile passes the same value.
SKILL_NAME = os.getenv("SKILL_NAME", f"{ROOT.name}-management")
SKILL = ROOT / ".claude" / "skills" / SKILL_NAME


def main() -> int:
    # The serving payload travels with the skill because server.py resolves it
    # next to itself first and only then looks up at the rig root — so an
    # installed copy under ~/.claude/skills can still START a serve.
    #
    # That resolution order is also the one hazard this rig inherits from the
    # cloud siblings, in a milder form. There is no deploy step and nothing is
    # shipped anywhere, but an MCP server running from the snapshot will start
    # the SNAPSHOT'S payload, not the working tree you are editing. The build id
    # on /health is what makes that visible; verify_model_health compares it.
    names = (
        "server.py",
        "project-setup.sh",
        "requirements.txt",
        "requirements-serving.txt",
        "jax_openai_server.py",
        "jax_engine.py",
        "ports/gemma4/jax_e_loader.py",
        "ports/gemma4/jax_e_model.py",
    )
    for name in names:
        source = ROOT / name
        destination = SKILL / "mcp" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"copied {name} -> {destination.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
