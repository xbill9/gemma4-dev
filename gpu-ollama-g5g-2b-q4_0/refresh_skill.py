#!/usr/bin/env python3
"""Refresh generated MCP files in the bundled G5g llama.cpp skill.

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
    # THREE FILES, NOT SIX. The siblings also snapshot a serving payload
    # (`*_openai_server.py`, `*_generate.py`, `requirements-serving.txt`) so an
    # installed skill can still run their deploy tool. This rig HAS NO PAYLOAD:
    # llama-server is built on the box from a pinned upstream ref and fetches its
    # own checkpoint, so there is nothing of ours to ship and no deploy tool to
    # ship it with.
    #
    # That also retires the trap the PyTorch sibling documents at length — its
    # deploy tool ships whichever payload root it resolves, so deploying through
    # the REGISTERED MCP server ships the previous `make skill` output. With no
    # payload there is no stale-payload failure mode here.
    names = (
        "server.py",
        "project-setup.sh",
        "requirements.txt",
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
