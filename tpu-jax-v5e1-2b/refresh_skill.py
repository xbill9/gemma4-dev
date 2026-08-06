#!/usr/bin/env python3
"""Refresh the bundled tpu-management skill snapshots from the repo-root sources.

Regenerates:
  .claude/skills/tpu-management/mcp/server.py              from  server.py
  .claude/skills/tpu-management/mcp/project-setup.sh       from  project-setup.sh
  .claude/skills/tpu-management/references/tpu-guide.md
                                                            from  tpu.md
                                                            (base64 screenshots stripped)

SKILL.md and mcp/startup_script_template.sh
are hand-maintained and left alone.
"""

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SKILL = ROOT / ".claude" / "skills" / "tpu-management"

IMAGE_DEF = re.compile(r"^\[image\d+\]: <data:image/[^;]+;base64,")

NOTE_TEMPLATE = """\
> **Note:** This is a text-only copy of the repository's `tpu.md` ({size} KB original). The
> embedded base64 screenshots for the Cloud Console walkthrough have been stripped;
> `![][imageN]` markers show where they appeared. See the original `tpu.md` at the repo
> root for the images.
"""


def build_guide(src: Path, dest: Path) -> None:
    lines = src.read_text().splitlines()
    kept = [line for line in lines if not IMAGE_DEF.match(line)]
    while kept and not kept[-1].strip():
        kept.pop()
    note = NOTE_TEMPLATE.format(size=round(src.stat().st_size / 1024))
    # Insert the note right after the document's title line.
    body = [kept[0], "", note.rstrip()] + kept[1:]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(body) + "\n")
    print(f"wrote {dest.relative_to(ROOT)} ({dest.stat().st_size} bytes)")


def main() -> int:
    if not (SKILL / "SKILL.md").exists():
        print(f"error: {SKILL} not found — run from the repo root", file=sys.stderr)
        return 1
    shutil.copyfile(ROOT / "server.py", SKILL / "mcp" / "server.py")
    print(f"copied server.py -> {(SKILL / 'mcp' / 'server.py').relative_to(ROOT)}")
    shutil.copy(ROOT / "project-setup.sh", SKILL / "mcp" / "project-setup.sh")  # copy() preserves +x
    print(f"copied project-setup.sh -> {(SKILL / 'mcp' / 'project-setup.sh').relative_to(ROOT)}")
    shutil.copyfile(ROOT / "requirements.txt", SKILL / "mcp" / "requirements.txt")
    print(f"copied requirements.txt -> {(SKILL / 'mcp' / 'requirements.txt').relative_to(ROOT)}")
    guide_src = ROOT / "tpu.md"
    if guide_src.exists():
        build_guide(guide_src, SKILL / "references" / "tpu-guide.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
