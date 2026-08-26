#!/usr/bin/env python3
"""Convert a dev.to markdown article into the Medium variant.

Medium's editor does not render markdown pipe tables — it drops them to run-on text and the
numbers become unreadable. It does render fenced code blocks in monospace, which preserves
column alignment, so every table becomes a space-aligned block. This matches the existing
`tpu-vllm-v5e1-2b/devto-vllm-gemma4-e2b-v5e1-medium.md` by construction rather than by hand.

Inside a code block no markdown is interpreted, so cell contents are flattened: `**bold**`,
backticks and link syntax are stripped, `×`/`≥`/`≤` are transliterated, and a lone em-dash
placeholder becomes blank. Prose outside tables keeps its markdown untouched.

Front matter is dropped — Medium has no equivalent, and the dev.to `series:`/`cover_image:`
keys are meaningless there.

Usage:
    python3 to_medium.py IN.md OUT.md [--cover URL] [--figure PATH|CAPTION ...]
"""

import argparse
import re
import sys
from typing import List, Optional

# Emoji are double-width in a monospace block, so a medal in a cell throws every column
# after it out of alignment — the one thing the code-block treatment exists to preserve.
# They render fine on dev.to, so this is a Medium-only substitution.
SUBS = [
    ("×", "x"),
    ("≥", ">="),
    ("≤", "<="),
    ("→", "->"),
    ("−", "-"),
    ("🥇", "#1"),
    ("🥈", "#2"),
    ("🥉", "#3"),
]


def flatten_cell(text: str) -> str:
    """Strip markup that a code block would show literally rather than render."""
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links/images -> label
    text = text.replace("**", "").replace("`", "")
    text = re.sub(r"(?<!\w)\*([^*]+)\*(?!\w)", r"\1", text)  # italics
    for a, b in SUBS:
        text = text.replace(a, b)
    text = text.strip()
    return "" if text in {"—", "-", "–"} else text


def split_row(line: str) -> List[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [flatten_cell(c) for c in line.split("|")]


def is_separator(line: str) -> bool:
    return bool(re.fullmatch(r"\s*\|?[\s:|-]+\|?\s*", line)) and "-" in line


def render_table(rows: List[List[str]]) -> List[str]:
    """Space-align a table into a monospace block, right-aligning numeric columns."""
    ncols = max(len(r) for r in rows)
    rows = [r + [""] * (ncols - len(r)) for r in rows]
    widths = [max(len(r[c]) for r in rows) for c in range(ncols)]

    def numeric(c: int) -> bool:
        body = [r[c] for r in rows[1:] if r[c]]
        if not body:
            return False
        return sum(bool(re.fullmatch(r"[-+$~]?[\d,.]+\s*[%xX]?.{0,12}", v)) for v in body) >= len(body) * 0.8

    align_right = [numeric(c) for c in range(ncols)]

    def fmt(r: List[str]) -> str:
        cells = []
        for c, v in enumerate(r):
            cells.append(v.rjust(widths[c]) if align_right[c] else v.ljust(widths[c]))
        return "  ".join(cells).rstrip()

    out = ["```", fmt(rows[0]), "  ".join("-" * w for w in widths).rstrip()]
    out += [fmt(r) for r in rows[1:]]
    out.append("```")
    return out


def convert(text: str, cover: Optional[str], figures: List[str]) -> str:
    lines = text.splitlines()

    # Drop YAML front matter.
    if lines and lines[0].strip() == "---":
        end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
        if end is not None:
            lines = lines[end + 1 :]

    out: List[str] = []
    i = 0
    in_fence = False
    n_tables = 0
    while i < len(lines):
        line = lines[i]

        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue

        # A table is a pipe row followed by a separator row.
        if not in_fence and line.strip().startswith("|") and i + 1 < len(lines) and is_separator(lines[i + 1]):
            rows = [split_row(line)]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(split_row(lines[i]))
                i += 1
            out.extend(render_table(rows))
            n_tables += 1
            continue

        # HTML comments are author scaffolding; Medium shows them as text.
        if not in_fence and line.lstrip().startswith("<!--"):
            while i < len(lines) and "-->" not in lines[i]:
                i += 1
            i += 1
            continue

        out.append(line)
        i += 1

    body = "\n".join(out)

    # Blockquotes render as pull-quotes on Medium and survive as-is; images go at the top.
    header: List[str] = []
    if cover:
        header += [f"![]({cover})", ""]
    body = "\n".join(header) + body

    if figures:
        body = body.rstrip() + "\n\n---\n\n## Figures\n\n"
        for spec in figures:
            # Split on "|", not ":" — a colon lands inside the "https://" of an image URL.
            path, _, caption = spec.partition("|")
            body += f"![{caption}]({path})\n*{caption}*\n\n"

    print(f"converted {n_tables} tables", file=sys.stderr)
    return body.rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("outfile")
    ap.add_argument("--cover", default=None)
    ap.add_argument("--figure", action="append", default=[])
    args = ap.parse_args()

    with open(args.infile) as f:
        text = f.read()
    with open(args.outfile, "w") as f:
        f.write(convert(text, args.cover, args.figure))
    print(f"wrote {args.outfile}", file=sys.stderr)


if __name__ == "__main__":
    main()
