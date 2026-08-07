#!/usr/bin/env python3
"""Validate every rig's benchmark reports against the canonical schema.

Exits non-zero if any report fails, so `make benchmarks-validate` can gate a commit.

Prefers the `check-jsonschema` CLI; falls back to the `jsonschema` module. If neither is
present it says so and exits non-zero rather than reporting success it did not establish —
a validator that silently passes when it cannot validate is worse than no validator.
"""

import glob
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA = os.path.join(ROOT, "benchmarks", "serving-report.schema.json")


def reports():
    return sorted(glob.glob(os.path.join(ROOT, "*", "benchmarks", "reports", "*.json")))


def check_filename_matches_run_id(path):
    """NAMING.md and the schema both require the filename to equal run.id."""
    try:
        with open(path) as fh:
            data = json.load(fh)
    except json.JSONDecodeError as e:
        return f"invalid JSON: {e}"
    expected = os.path.basename(path)[: -len(".json")]
    actual = (data.get("run") or {}).get("id")
    if actual and actual != expected:
        return f"run.id {actual!r} != filename {expected!r}"
    return None


def validate_cli(paths):
    cmd = ["check-jsonschema", "--schemafile", SCHEMA, *paths]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.returncode == 0, (res.stdout + res.stderr).strip()


def validate_module(paths):
    import warnings

    warnings.filterwarnings("ignore")
    from jsonschema import Draft202012Validator

    with open(SCHEMA) as fh:
        schema = json.load(fh)
    Draft202012Validator.check_schema(schema)
    v = Draft202012Validator(schema)
    ok = True
    out = []
    for p in paths:
        with open(p) as fh:
            data = json.load(fh)
        errs = sorted(v.iter_errors(data), key=lambda e: list(e.path))
        rel = os.path.relpath(p, ROOT)
        if errs:
            ok = False
            out.append(f"❌ {rel}")
            for e in errs[:5]:
                out.append(f"     at {list(e.path) or '(root)'}: {e.message}")
        else:
            out.append(f"✅ {rel}")
    return ok, "\n".join(out)


def main():
    paths = reports()
    if not paths:
        print("No reports found under */benchmarks/reports/.")
        return 0

    if shutil.which("check-jsonschema"):
        ok, out = validate_cli(paths)
    else:
        try:
            ok, out = validate_module(paths)
        except ImportError:
            print(
                "❌ No validator available. Install one (system python3, no virtualenv):\n"
                "     pip install check-jsonschema\n"
                "   or pip install jsonschema",
                file=sys.stderr,
            )
            return 1
    print(out)

    naming = [(p, msg) for p in paths if (msg := check_filename_matches_run_id(p))]
    for p, msg in naming:
        print(f"❌ {os.path.relpath(p, ROOT)}: {msg}")

    if ok and not naming:
        print(f"\n✅ {len(paths)} report(s) valid against {os.path.relpath(SCHEMA, ROOT)}")
        return 0
    print(f"\n❌ validation failed ({len(naming)} naming problem(s))", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
