#!/usr/bin/env python3
"""Convert a dev.to-flavoured markdown article into a Medium-friendly variant.

Medium has no usable markdown table support: pasted or imported pipe tables come
through as mangled text, which is why they need hand-editing after every import.
This rewrites each table into something Medium *does* render natively, so the
Medium copy is generated rather than repaired.

Strategies (--tables):
  code  aligned ASCII inside a fenced code block. Preserves the grid exactly and
        survives import; monospace, and wide tables scroll rather than wrap.
  list  a bold row label followed by "header: value" bullets. Fully native
        Medium styling and mobile-friendly, but longer.
  auto  code for tables that fit `--width`, list for anything wider (default).

Also strips the front matter (Medium takes the title from the first H1) and
reports any images, which Medium always requires you to upload by hand.

Usage:
  ./to-medium.py devto-vllm-gemma4-e2b-v5e1.md -o medium-article.md
  ./to-medium.py devto-vllm-gemma4-e2b-v5e1.md --tables list
"""

import argparse
import re
import sys


def parse_table(block):
    """Return (headers, rows) from markdown pipe-table lines, or None."""
    rows = []
    for line in block:
        if re.match(r"^\s*\|[\s:|-]+\|\s*$", line):  # separator row
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    if len(rows) < 2:
        return None
    return rows[0], rows[1:]


def strip_md(s):
    """Flatten inline markdown that adds no value inside a code block."""
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"(?<!`)\*(.+?)\*(?!`)", r"\1", s)
    s = re.sub(r"`(.+?)`", r"\1", s)
    s = re.sub(r"\[(.+?)\]\((.+?)\)", r"\1", s)
    return s


def as_code(headers, rows):
    hs = [strip_md(h) for h in headers]
    rs = [[strip_md(c) for c in r] for r in rows]
    n = len(hs)
    rs = [r + [""] * (n - len(r)) for r in rs]
    w = [max(len(hs[i]), *(len(r[i]) for r in rs)) if rs else len(hs[i]) for i in range(n)]
    fmt = lambda cells: "  ".join(c.ljust(w[i]) for i, c in enumerate(cells)).rstrip()
    out = ["```", fmt(hs), "  ".join("-" * x for x in w)]
    out += [fmt(r) for r in rs]
    out.append("```")
    return out


def as_list(headers, rows):
    out = []
    for r in rows:
        r = list(r) + [""] * (len(headers) - len(r))
        label = re.sub(r"\*\*", "", (r[0] or "").strip()) or "—"
        out.append(f"**{label}**")
        for h, c in zip(headers[1:], r[1:]):
            if c:
                h2 = re.sub(r"\*\*", "", h.strip())
                out.append(f"- {h2}: {c}" if h2 else f"- {c}")
        out.append("")
    return out[:-1] if out else out


def convert(md, strategy="auto", width=76):
    lines = md.split("\n")
    # drop YAML front matter
    if lines and lines[0].strip() == "---":
        end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
        if end:
            lines = lines[end + 1 :]
            while lines and not lines[0].strip():
                lines.pop(0)
    out, i, in_code = [], 0, False
    n_code = n_list = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            in_code = not in_code
            out.append(line)
            i += 1
            continue
        if not in_code and line.strip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            parsed = parse_table(block)
            if not parsed:
                out.extend(block)
                continue
            headers, rows = parsed
            code = as_code(headers, rows)
            widest = max(len(x) for x in code)
            use_code = strategy == "code" or (strategy == "auto" and widest <= width)
            if use_code:
                out.extend(code)
                n_code += 1
            else:
                out.extend(as_list(headers, rows))
                n_list += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out), n_code, n_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--output")
    ap.add_argument("--tables", choices=["code", "list", "auto"], default="auto")
    ap.add_argument("--width", type=int, default=76, help="max rendered width before auto falls back to list form")
    a = ap.parse_args()

    md = open(a.input).read()
    out, n_code, n_list = convert(md, a.tables, a.width)
    dest = a.output or a.input.replace(".md", "") + "-medium.md"
    open(dest, "w").write(out)

    imgs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", out)
    print(f"wrote {dest}  ({len(out.splitlines())} lines, {len(out)} bytes)")
    print(f"tables converted: {n_code} as code block, {n_list} as list")
    if imgs:
        print(f"images needing manual upload on Medium: {len(imgs)}")
        for x in imgs:
            print(f"  - {x}")
    else:
        print("images: none inline (cover art is attached separately on Medium)")


if __name__ == "__main__":
    sys.exit(main())
