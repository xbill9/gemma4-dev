#!/usr/bin/env python3
"""Aggregate the sweep's per-cell vllm-bench JSONs into report tables.

Usage: python3 aggregate.py <results_root>   # dir with <model>/c<ctx>-u<users>.json

Stdlib only (repo standard). Emits:
  - summary.json  : nested {model: {ctx: {users: metrics}}} + boot info
  - markdown tables to stdout (output tok/s, per-stream tok/s, median TTFT ms)
Cells present as .fail/.skip files are marked "failed"/"infeasible" in the tables.

Cell shape follows benchmarks/serving-report.schema.json v1.1, so a cell drops into a
report's throughput.sweep[] without renaming: latency nested under ttft_ms / tpot_ms as
{mean, median, p90, p99}, and status drawn from the schema's ok / infeasible / failed
vocabulary. A .skip marker means the cell could not exist on this hardware+config, which
the schema calls "infeasible"; the older "skip"/"fail" spellings were local to this script.
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
                    cells[(ctx, users)] = {"status": "failed"}
                    continue
                # input_len falls back to the cell coordinate: vLLM writes random_input_len
                # as null in the dump, and it is the second axis of this sweep.
                cells[(ctx, users)] = {
                    "status": "ok",
                    "concurrency": users,
                    "input_len": r.get("random_input_len") or r.get("input_len") or ctx,
                    "output_tok_per_s": round(r.get("output_throughput", 0), 1),
                    "total_tok_per_s": round(r.get("total_token_throughput", 0), 1),
                    "req_per_s": round(r.get("request_throughput", 0), 3),
                    "ttft_ms": {
                        "median": round(r.get("median_ttft_ms", 0), 1),
                        "p99": round(r.get("p99_ttft_ms", 0), 1),
                    },
                    "tpot_ms": {"median": round(r.get("median_tpot_ms", 0), 2)},
                    "completed": r.get("completed"),
                }
            else:
                # .fail/.skip never overrides a successful .json for the same cell
                status = {"skip": "infeasible", "fail": "failed"}[kind]
                cells.setdefault((ctx, users), {"status": status, "concurrency": users, "input_len": ctx})
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
                # metric may be a dotted path into the nested latency objects, e.g. "ttft_ms.median"
                v = cell
                for part in metric.split("."):
                    v = v.get(part) if isinstance(v, dict) else None
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
    # stderr, not stdout: this script is run as `aggregate.py results > tables.md`, and the
    # progress line used to land as the first line of the generated table file.
    print(f"wrote {root}/summary.json", file=sys.stderr)
    for m, d in data.items():
        n_ok = sum(1 for v in d["cells"].values() if v["status"] == "ok")
        n_inf = sum(1 for v in d["cells"].values() if v["status"] == "infeasible")
        n_fail = sum(1 for v in d["cells"].values() if v["status"] == "failed")
        boot = d["boot"] or {}
        # Coverage is stated in full: a measured-only count reads as total coverage.
        print(
            f"\n## {m} — {n_ok} measured, {n_fail} failed, {n_inf} infeasible, "
            f"max_model_len={boot.get('max_model_len')}, "
            f"time_to_healthy={boot.get('time_to_healthy_s')}s"
        )
        for metric, title, fmt in (
            ("output_tok_per_s", "Aggregate output tok/s", "{:.0f}"),
            ("tpot_ms.median", "Median TPOT ms (per-stream latency)", "{:.1f}"),
            ("ttft_ms.median", "Median TTFT ms", "{:.0f}"),
        ):
            print(f"\n### {title}\n")
            print(table(data, m, metric, fmt))


if __name__ == "__main__":
    main()
