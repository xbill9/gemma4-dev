#!/usr/bin/env python3
"""Build Colab-ready .ipynb from py:percent sources in src/.

Each source declares its own front matter in a module docstring-style header
block (see src/README.md). The build injects, ahead of the authored cells:

  1. an Apache-2.0 license cell (the Marathon asks for Apache 2.0), and
  2. an "Open in Colab" badge pointing at this notebook's path on GitHub.

It also sets ``metadata.accelerator`` (and ``colab.gpuType``) so opening the notebook
in Colab selects the right runtime instead of leaving the reader on CPU.

Usage:
    python3 tools/build_notebooks.py [--check]

``--check`` rebuilds in memory and fails if the committed notebooks are
stale, which is what CI runs.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import pathlib
import sys

import jupytext
import nbformat

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
OUT = ROOT / "notebooks"
CONFIG = ROOT / "notebooks.json"

LICENSE_CELL = """\
##### Copyright 2026 The gemma4-dev Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
[apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0).

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\
"""

BADGE = """\
<table align="left"><td>
  <a target="_blank" href="https://colab.research.google.com/github/{repo}/blob/{branch}/{prefix}notebooks/{name}">
    <img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab
  </a>
</td><td>
  <a target="_blank" href="https://github.com/{repo}/blob/{branch}/{prefix}notebooks/{name}">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub
  </a>
</td></table>\
"""


def load_config() -> dict:
    return json.loads(CONFIG.read_text())


def build_one(source: pathlib.Path, cfg: dict) -> nbformat.NotebookNode:
    nb = jupytext.read(source, fmt="py:percent")
    name = source.stem + ".ipynb"

    header = [
        nbformat.v4.new_markdown_cell(LICENSE_CELL),
        nbformat.v4.new_markdown_cell(
            BADGE.format(repo=cfg["repo"], branch=cfg["branch"], prefix=cfg.get("path", ""), name=name)
        ),
    ]
    nb.cells = header + nb.cells

    # Strip execution state so committed notebooks diff cleanly. Outputs belong
    # in the *_output.ipynb that `colab exec` writes, never in the source.
    #
    # Cell ids must be deterministic or the build is not reproducible: nbformat
    # mints a random id for any cell lacking one, so an unchanged source would
    # produce a different file on every run and `--check` would always fail.
    # Derive them from position and content instead.
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        digest = hashlib.sha256(f"{i}\0{cell.cell_type}\0{cell.source}".encode()).hexdigest()
        cell["id"] = f"c{i:03d}-{digest[:8]}"

    nb.metadata = {
        "accelerator": cfg["notebooks"][source.name].get("accelerator", "GPU"),
        "authors": [{"name": a} for a in cfg.get("authors", [])],
        "colab": {"name": name, "provenance": [], "toc_visible": True,
                  **({"gpuType": cfg["notebooks"][source.name]["gpuType"]}
                     if "gpuType" in cfg["notebooks"][source.name] else {})},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return nb


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    OUT.mkdir(exist_ok=True)
    stale = []

    for source_name in cfg["notebooks"]:
        source = SRC / source_name
        if not source.exists():
            print(f"missing source: {source}", file=sys.stderr)
            return 1
        nb = build_one(source, cfg)
        rendered = nbformat.writes(nb, version=4) + "\n"
        target = OUT / (source.stem + ".ipynb")

        if args.check:
            current = target.read_text() if target.exists() else ""
            if current != rendered:
                stale.append(target.relative_to(ROOT))
                diff = difflib.unified_diff(
                    current.splitlines()[:40],
                    rendered.splitlines()[:40],
                    fromfile="committed",
                    tofile="rebuilt",
                    lineterm="",
                )
                print("\n".join(diff), file=sys.stderr)
        else:
            target.write_text(rendered)
            print(f"built {target.relative_to(ROOT)} ({len(nb.cells)} cells)")

    if stale:
        print(f"\nstale notebooks: {', '.join(map(str, stale))}", file=sys.stderr)
        print("run: make build", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
