#!/usr/bin/env python3
"""Aggregate the sweep's per-cell vllm-bench JSONs into report tables.

Usage: python3 aggregate.py <results_root>   # dir with <model>/c<ctx>-u<users>.json

Stdlib only (repo standard). Emits:
  - summary.json  : nested {model: {ctx: {users: metrics}}} + boot info
  - markdown tables to stdout (output tok/s, per-stream tok/s, median TTFT ms)
Cells present as .fail/.skip files are marked "fail"/"skip" in the tables.
"""

import json
import re
import sys
from pathlib import Path

CELL = re.compile(r"c(\d+)-u(\d+)\.(json|fail|skip)$")


def load(root: Path):
    data = {}
    for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
        model = mdir.name
        cells = {}
        boot = None
        bj = mdir / "boot.json"
        if bj.exists():
            boot = json.loads(bj.read_text())
        for f in mdir.iterdir():
            m = CELL.search(f.name)
            if not m:
                continue
            ctx, users, kind = int(m.group(1)), int(m.group(2)), m.group(3)
            if kind == "json":
                try:
                    r = json.loads(f.read_text())
                except json.JSONDecodeError:
                    cells[(ctx, users)] = {"status": "fail"}
                    continue
                cells[(ctx, users)] = {
                    "status": "ok",
                    "output_tok_per_s": round(r.get("output_throughput", 0), 1),
                    "total_tok_per_s": round(r.get("total_token_throughput", 0), 1),
                    "req_per_s": round(r.get("request_throughput", 0), 3),
                    "ttft_ms_median": round(r.get("median_ttft_ms", 0), 1),
                    "ttft_ms_p99": round(r.get("p99_ttft_ms", 0), 1),
                    "tpot_ms_median": round(r.get("median_tpot_ms", 0), 2),
                    "input_len": r.get("random_input_len") or r.get("input_len"),
                    "completed": r.get("completed"),
                }
            else:
                # .fail/.skip never overrides a successful .json for the same cell
                cells.setdefault((ctx, users), {"status": kind})
        data[model] = {"boot": boot, "cells": cells}
    return data


def table(data, model, metric, fmt="{:.0f}"):
    cells = data[model]["cells"]
    ctxs = sorted({c for c, _ in cells})
    users = sorted({u for _, u in cells})
    lines = ["| ctx \\ users | " + " | ".join(str(u) for u in users) + " |"]
    lines.append("|" + "---|" * (len(users) + 1))
    for c in ctxs:
        row = [f"| {c} "]
        for u in users:
            cell = cells.get((c, u))
            if cell is None:
                row.append("· ")
            elif cell["status"] != "ok":
                row.append(cell["status"] + " ")
            else:
                v = cell.get(metric)
                row.append((fmt.format(v) if v is not None else "?") + " ")
        lines.append("|".join(row) + "|")
    return "\n".join(lines)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    data = load(root)
    out = {
        m: {
            "boot": d["boot"],
            "cells": {f"c{c}-u{u}": v for (c, u), v in sorted(d["cells"].items())},
        }
        for m, d in data.items()
    }
    (root / "summary.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {root}/summary.json")
    for m, d in data.items():
        n_ok = sum(1 for v in d["cells"].values() if v["status"] == "ok")
        boot = d["boot"] or {}
        print(f"\n## {m} — {n_ok} cells ok, max_model_len={boot.get('max_model_len')}, "
              f"time_to_healthy={boot.get('time_to_healthy_s')}s")
        for metric, title, fmt in (
            ("output_tok_per_s", "Aggregate output tok/s", "{:.0f}"),
            ("tpot_ms_median", "Median TPOT ms (per-stream latency)", "{:.1f}"),
            ("ttft_ms_median", "Median TTFT ms", "{:.0f}"),
        ):
            print(f"\n### {title}\n")
            print(table(data, m, metric, fmt))


if __name__ == "__main__":
    main()
