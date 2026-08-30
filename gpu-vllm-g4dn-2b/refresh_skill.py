#!/usr/bin/env python3
"""Refresh generated MCP files in the bundled gpu-vllm-g4dn-2b skill."""

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Skill name carries the rig directory so sibling rigs do not collide on one
# ~/.claude/skills/<name>; the Makefile passes the same value.
SKILL_NAME = os.getenv("SKILL_NAME", f"{ROOT.name}-management")
SKILL = ROOT / ".claude" / "skills" / SKILL_NAME

# patch_triton_turing.py is GENERATED INTO THE SKILL for the same reason
# server.py is: `create_g4dn_instance` reads it from beside server.py and ships
# it in user data, so an installed skill copy without it cannot launch an
# instance at all. `gpu-jax-g5g-2b` learned the general form of this the hard
# way -- it deployed the previous `make skill` output with no warning and lost a
# full measure-and-conclude cycle before the digests were compared.
GENERATED = ("server.py", "patch_triton_turing.py", "project-setup.sh", "requirements.txt")


def main() -> int:
    for name in GENERATED:
        source = ROOT / name
        destination = SKILL / "mcp" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"copied {name} -> {destination.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
