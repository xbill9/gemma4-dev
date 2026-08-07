#!/usr/bin/env python3
"""Generate the cross-rig benchmark rollup and each rig's own index.

Writes:
  benchmarks/ROLLUP.md            one table of every report in every rig, deduplicated
  <rig>/benchmarks/INDEX.md       that rig's reports and runs

Both are generated — hand-edits are lost on the next `make benchmarks-rollup`.

Reports are deduplicated by content hash, because the same measurement is currently stored in
several rigs (it travelled with the forks). The rollup lists a measurement once and names every
rig carrying it, so a reader can tell one result copied five times from five results.

Run directories are reported with a completeness count rather than assumed equal: the copies of
one sweep are NOT all the same, and a stub with no results/ reads identical to a full run if you
only look at the directory name.

Stdlib only — pandas is not assumed anywhere in this repo.
"""

import hashlib
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA_PATH = os.path.join(ROOT, "benchmarks", "serving-report.schema.json")


def rigs():
    """Every directory holding a benchmarks/ dir, in sorted order."""
    out = []
    for name in sorted(os.listdir(ROOT)):
        if os.path.isdir(os.path.join(ROOT, name, "benchmarks")) and name != "benchmarks":
            out.append(name)
    return out


def load_reports():
    """(digest -> {'data':..., 'rigs':[...], 'name':...}) over every rig's reports/."""
    by_digest = {}
    for rig in rigs():
        rdir = os.path.join(ROOT, rig, "benchmarks", "reports")
        if not os.path.isdir(rdir):
            continue
        for fn in sorted(os.listdir(rdir)):
            if not fn.endswith(".json"):
                continue
            path = os.path.join(rdir, fn)
            with open(path, "rb") as fh:
                blob = fh.read()
            digest = hashlib.sha256(blob).hexdigest()[:12]
            try:
                data = json.loads(blob)
            except json.JSONDecodeError as e:
                data = {"_parse_error": str(e)}
            entry = by_digest.setdefault(digest, {"data": data, "rigs": [], "names": set()})
            entry["rigs"].append(rig)
            entry["names"].add(fn)
    return by_digest


def peak_output(data):
    """Best output_tok_per_s across measured sweep points, or None."""
    sweep = (data.get("throughput") or {}).get("sweep") or []
    vals = [p.get("output_tok_per_s") for p in sweep if p.get("status", "ok") == "ok"]
    vals = [v for v in vals if isinstance(v, (int, float))]
    return max(vals) if vals else None


def sweep_shape(data):
    """'1-D (n pts)' or '2-D (c ctx x u conc)', plus non-ok cell counts."""
    sweep = (data.get("throughput") or {}).get("sweep") or []
    if not sweep:
        return "—", ""
    ctxs = {p.get("input_len") for p in sweep if p.get("input_len") is not None}
    concs = {p.get("concurrency") for p in sweep}
    bad = defaultdict(int)
    for p in sweep:
        st = p.get("status", "ok")
        if st != "ok":
            bad[st] += 1
    flags = ", ".join(f"{n} {st}" for st, n in sorted(bad.items()))
    if ctxs:
        return f"2-D ({len(ctxs)}ctx × {len(concs)}conc)", flags
    return f"1-D ({len(sweep)} pts)", flags


def cheapest_cost(data):
    entries = (data.get("cost") or {}).get("per_m_output_tokens") or []
    vals = [e.get("usd") for e in entries if isinstance(e.get("usd"), (int, float))]
    return min(vals) if vals else None


def run_dirs():
    """[(rig, run_name, n_files, n_results, n_skip, has_report, has_tables)]"""
    rows = []
    for rig in rigs():
        rdir = os.path.join(ROOT, rig, "benchmarks", "runs")
        if not os.path.isdir(rdir):
            continue
        for run in sorted(os.listdir(rdir)):
            path = os.path.join(rdir, run)
            if not os.path.isdir(path):
                continue
            n_files = n_results = n_skip = 0
            for dirpath, _, filenames in os.walk(path):
                for fn in filenames:
                    n_files += 1
                    rel = os.path.relpath(os.path.join(dirpath, fn), path)
                    if rel.startswith("results" + os.sep):
                        n_results += 1
                    # A .skip marker only means the cell is infeasible if no result exists for
                    # it. A fixup re-run can measure a previously skipped cell and leave the
                    # marker behind; counting markers then overstates infeasibility, which is
                    # exactly how this sweep's REPORT.md came to claim 42 instead of 28.
                    if fn.endswith(".skip"):
                        if not os.path.exists(os.path.join(dirpath, fn[: -len(".skip")] + ".json")):
                            n_skip += 1
            rows.append(
                (
                    rig,
                    run,
                    n_files,
                    n_results,
                    n_skip,
                    os.path.isfile(os.path.join(path, "REPORT.md")),
                    os.path.isfile(os.path.join(path, "tables.md")),
                )
            )
    return rows


def validate(reports):
    """digest -> 'ok' | 'FAIL: ...' | 'skipped (no validator)'"""
    try:
        import warnings

        warnings.filterwarnings("ignore")
        from jsonschema import Draft202012Validator
    except ImportError:
        return dict.fromkeys(reports, "skipped (pip install jsonschema)")
    with open(SCHEMA_PATH) as fh:
        schema = json.load(fh)
    v = Draft202012Validator(schema)
    out = {}
    for digest, entry in reports.items():
        errs = sorted(v.iter_errors(entry["data"]), key=lambda e: list(e.path))
        if not errs:
            out[digest] = "ok"
        else:
            e = errs[0]
            out[digest] = f"FAIL at {list(e.path) or '(root)'}: {e.message[:80]}"
    return out


def fmt(v, suffix="", nd=0):
    if v is None:
        return "—"
    return f"{v:,.{nd}f}{suffix}"


def write_rollup(reports, verdicts):
    lines = [
        "# Benchmark rollup",
        "",
        "**Generated by `benchmarks/rollup.py` — do not hand-edit.** Regenerate with",
        "`make benchmarks-rollup` from the monorepo root.",
        "",
        "One row per distinct measurement. The same report file is currently stored in several",
        "rigs (copies travelled with the forks), so *Carried by* names every rig holding a",
        "byte-identical copy — five rigs on one row is one measurement, not five.",
        "",
        "`Hardware` is what was **measured**, which is not necessarily the rig the file sits in.",
        "",
        "## Reports",
        "",
        "| Report (`run.id`) | Model | Hardware | Deployment | Engine | Sweep | Peak out tok/s | $/M out | Schema | Valid | Carried by |",
        "|---|---|---|---|---|---|---:|---:|---|---|---|",
    ]
    if not reports:
        lines.append("| _none found_ | | | | | | | | | | |")
    for digest, entry in sorted(reports.items(), key=lambda kv: sorted(kv[1]["names"])):
        d = entry["data"]
        run = d.get("run") or {}
        hw = d.get("hardware") or {}
        host = hw.get("host") or {}
        model = d.get("model") or {}
        sw = d.get("software") or {}
        chips = hw.get("chips")
        accel = hw.get("accelerator") or "—"
        hw_str = f"{accel}×{chips}" if chips else accel
        deploy = host.get("provisioning") or "—"
        if host.get("cloud"):
            deploy = f"{deploy} / {host['cloud']}"
        engine = f"{sw.get('engine', '—')} {sw.get('version', '')}".strip()
        if len(engine) > 28:
            engine = engine[:26] + "…"
        shape, flags = sweep_shape(d)
        if flags:
            shape = f"{shape}<br>{flags}"
        names = ", ".join(sorted(entry["names"]))
        lines.append(
            "| `{id}` | {model} | {hw} | {deploy} | {engine} | {shape} | {peak} | {cost} | {ver} | {valid} | {rigs} |".format(
                id=run.get("id") or names,
                model=model.get("id") or "—",
                hw=hw_str,
                deploy=deploy,
                engine=engine or "—",
                shape=shape,
                peak=fmt(peak_output(d)),
                cost=fmt(cheapest_cost(d), nd=2),
                ver=d.get("schema_version") or "—",
                valid=verdicts.get(digest, "?"),
                rigs="<br>".join(entry["rigs"]),
            )
        )

    lines += [
        "",
        "## Sweep run directories",
        "",
        "Copies of one run are **not** interchangeable. A stub with no `results/` looks identical",
        "to a complete run if you only read the directory name, so completeness is counted here.",
        "",
        "| Rig | Run | Files | Result files | Infeasible cells (unshadowed) | REPORT.md | tables.md |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    rows = run_dirs()
    if not rows:
        rows = []
        lines.append("| _none found_ | | | | | | |")
    for rig, run, nf, nr, ns, hr, ht in rows:
        lines.append(
            f"| {rig} | `{run}` | {nf} | {nr} | {ns} | {'yes' if hr else '**no**'} | {'yes' if ht else '**no**'} |"
        )

    lines += [
        "",
        "## Coverage",
        "",
        "| Rig | Reports | Runs |",
        "|---|---:|---:|",
    ]
    per_rig_reports = defaultdict(int)
    for entry in reports.values():
        for rig in entry["rigs"]:
            per_rig_reports[rig] += 1
    per_rig_runs = defaultdict(int)
    for rig, *_ in rows:
        per_rig_runs[rig] += 1
    for rig in rigs():
        lines.append(f"| {rig} | {per_rig_reports.get(rig, 0)} | {per_rig_runs.get(rig, 0)} |")
    lines.append("")

    out = os.path.join(ROOT, "benchmarks", "ROLLUP.md")
    with open(out, "w") as fh:
        fh.write("\n".join(lines))
    return out


def write_rig_indexes(reports, verdicts):
    written = []
    runs_by_rig = defaultdict(list)
    for row in run_dirs():
        runs_by_rig[row[0]].append(row)
    for rig in rigs():
        mine = [(digest, entry) for digest, entry in reports.items() if rig in entry["rigs"]]
        lines = [
            f"# {rig} — benchmark index",
            "",
            "**Generated by the monorepo `benchmarks/rollup.py` — do not hand-edit.**",
            "Regenerate with `make benchmarks-rollup` from the monorepo root.",
            "",
            "Cross-rig comparison lives in the monorepo root `benchmarks/ROLLUP.md`.",
            "",
            "## Reports in this rig",
            "",
            "| File | Model | Hardware measured | Deployment | Valid | Also in |",
            "|---|---|---|---|---|---|",
        ]
        if not mine:
            lines.append("| _none_ | | | | | |")
        for digest, entry in sorted(mine, key=lambda kv: sorted(kv[1]["names"])):
            d = entry["data"]
            hw = d.get("hardware") or {}
            host = hw.get("host") or {}
            chips = hw.get("chips")
            hw_str = f"{hw.get('accelerator', '—')}×{chips}" if chips else (hw.get("accelerator") or "—")
            others = [r for r in entry["rigs"] if r != rig]
            lines.append(
                "| `{f}` | {m} | {hw} | {dep} | {v} | {o} |".format(
                    f=", ".join(sorted(entry["names"])),
                    m=(d.get("model") or {}).get("id") or "—",
                    hw=hw_str,
                    dep=host.get("provisioning") or "—",
                    v=verdicts.get(digest, "?"),
                    o=", ".join(others) if others else "—",
                )
            )
        lines += [
            "",
            "## Run directories in this rig",
            "",
            "| Run | Files | Result files | Infeasible cells | REPORT.md |",
            "|---|---:|---:|---:|---|",
        ]
        mine_runs = runs_by_rig.get(rig, [])
        if not mine_runs:
            lines.append("| _none_ | | | | |")
        for _, run, nf, nr, ns, hr, _ht in mine_runs:
            lines.append(f"| `{run}` | {nf} | {nr} | {ns} | {'yes' if hr else '**no**'} |")
        lines.append("")
        out = os.path.join(ROOT, rig, "benchmarks", "INDEX.md")
        with open(out, "w") as fh:
            fh.write("\n".join(lines))
        written.append(out)
    return written


def main():
    reports = load_reports()
    verdicts = validate(reports)
    rollup = write_rollup(reports, verdicts)
    indexes = write_rig_indexes(reports, verdicts)
    print(f"Wrote {os.path.relpath(rollup, ROOT)}")
    for p in indexes:
        print(f"Wrote {os.path.relpath(p, ROOT)}")
    bad = [v for v in verdicts.values() if v.startswith("FAIL")]
    if bad:
        print(f"\n{len(bad)} report(s) failed validation", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
