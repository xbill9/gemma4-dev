#!/usr/bin/env python3
"""Refresh generated MCP files in the bundled Inf2 skill."""

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Skill name carries the rig directory so sibling rigs do not collide on one
# ~/.claude/skills/<name>; the Makefile passes the same value.
SKILL_NAME = os.getenv("SKILL_NAME", f"{ROOT.name}-management")
SKILL = ROOT / ".claude" / "skills" / SKILL_NAME


def main() -> int:
    # The serving payload travels with the skill, not just the MCP control plane,
    # so an installed copy under ~/.claude/skills can still ship the engine to an
    # instance. server.py resolves the payload next to itself first and only then
    # looks up at the rig root.
    names = (
        "server.py",
        "project-setup.sh",
        "requirements.txt",
        "requirements-serving.txt",
        "torch_generate.py",
        "torch_openai_server.py",
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
