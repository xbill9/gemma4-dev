#!/usr/bin/env python3
"""Refresh the bundled skill snapshots from the rig-root sources.

Regenerates:
  .claude/skills/<skill>/mcp/server.py           from  server.py
  .claude/skills/<skill>/mcp/project-setup.sh    from  project-setup.sh
  .claude/skills/<skill>/mcp/requirements.txt    from  requirements.txt
  .claude/skills/<skill>/mcp/tpu.env             from  tpu.env

SKILL.md is hand-maintained and left alone.

There is no references/ guide here. The TPU rigs snapshot a private `tpu.md`
into one; this rig's equivalent knowledge is the porting write-up, which is
published rather than private, so SKILL.md links it instead of embedding a copy
that would drift.
"""

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Skill name carries the rig directory so sibling rigs do not collide on one
# ~/.claude/skills/<name>; the Makefile passes the same value.
SKILL_NAME = os.getenv("SKILL_NAME", f"{ROOT.name}-management")
SKILL = ROOT / ".claude" / "skills" / SKILL_NAME

# tpu.env is copied into the snapshot because the server reads it at import
# time for every default it has. A skill installed without it falls back to the
# literals in server.py, which are the same values — but only until one of them
# changes here and not there.
SOURCES = ["server.py", "project-setup.sh", "requirements.txt", "tpu.env"]


def main() -> int:
    if not (SKILL / "SKILL.md").exists():
        print(f"error: {SKILL} not found — run from the rig root", file=sys.stderr)
        return 1
    (SKILL / "mcp").mkdir(parents=True, exist_ok=True)
    for name in SOURCES:
        src = ROOT / name
        if not src.exists():
            print(f"warning: {name} not found — keeping the existing snapshot")
            continue
        dest = SKILL / "mcp" / name
        # copy() rather than copyfile() so project-setup.sh keeps its +x bit.
        shutil.copy(src, dest)
        print(f"copied {name} -> {dest.relative_to(ROOT)} ({dest.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
