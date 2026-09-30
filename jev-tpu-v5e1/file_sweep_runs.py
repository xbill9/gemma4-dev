#!/usr/bin/env python3
"""file_sweep_runs.py -- file finished sweep runs into the rig named for each arm's chip and checkpoint.

  file_sweep_runs.py [--dry-run] <run> [<run> ...]

For each run (e.g. 2026-09-30-fillA-v5e1): pulls results/ and logs/ from the bucket into results/ and
results/<run>-logs/, reads which checkpoint each arm served from run.log, and moves every directory
results/<run>-<arm>{,-latency,-smoke,-suite,-gen} to
../<rig>/benchmarks/runs/<date>-<short>-<arm>-v5e1/, leaving a relative symlink at the old path, as
NAMING.md ("The repack sweeps") describes. The arm's own logs (<arm>.* and the boot log of its
checkpoint) are copied beside it under logs/<run>-logs/; run-wide logs stay here. Arms that produced no
result directory are filed the same way with their logs only. ../benchmarks/sweep-moves.json gains an
entry for every moved path. A rig in NEW_RIGS that does not exist yet is created as an artifact rig
from tpu-vllm-v5e1-2b-w8a8rtn's CLAUDE.md and README.md. Each rig's README table is left to the caller.
"""

import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BUCKET = "gs://aisprint-491218-bucket/jev-tpu-v5e1"
SUFFIXES = ("", "-latency", "-smoke", "-suite", "-gen")

# Checkpoint basename -> rig (slot 5 per NAMING.md). Reads the checkpoint from run.log, never the arm tag.
RIGS = {
    "gemma-4-E2B-it": "tpu-vllm-v5e1-2b",
    "gemma-4-E2B-it-qat-q4_0-unquantized": "tpu-vllm-v5e1-2b-q4_0",
    "gemma-4-E2B-it-qat-w4a16-ct": "tpu-vllm-v5e1-2b-w4a16",
    "gemma-4-E2B-it-qat-q4_0-w4a16-ct": "tpu-vllm-v5e1-2b-q4w4a16",
    "gemma-4-E2B-it-qat-q4_0-w4a16-ct-text": "tpu-vllm-v5e1-2b-q4w4a16",
    "gemma-4-E2B-it-qat-w8a8-int8": "tpu-vllm-v5e1-2b-w8a8",
    "gemma-4-E2B-it-qat-w8a8-int8-emb4": "tpu-vllm-v5e1-2b-w8a8emb4",
    "gemma-4-E2B-it-qat-w8a8-ct-text-emb4": "tpu-vllm-v5e1-2b-w8a8emb4",
    "gemma-4-E4B-it-qat-w4a16-ct": "tpu-vllm-v5e1-4b-w4a16",
    "gemma-4-E4B-it-qat-q4_0-w4a16-ct": "tpu-vllm-v5e1-4b-q4w4a16",
    "gemma-4-E4B-it-qat-w8a8-int8-emb4": "tpu-vllm-v5e1-4b-w8a8emb4",
    "gemma-4-E4B-it-W8A8-INT8-glenic-notiedhead": "tpu-vllm-v5e1-4b-w8a8rtn",
    "gemma-4-12B-it-qat-w4a16-ct": "tpu-vllm-v5e1-12b-w4a16",
    "gemma-4-12B-it-qat-q4_0-w4a16-ct": "tpu-vllm-v5e1-12b-q4w4a16",
    "gemma-4-12B-it-qat-q4_0-w4a16-ct-text-emb4": "tpu-vllm-v5e1-12b-q4w4a16emb4",
    "gemma-4-12B-it-qat-w8a8-int8-emb4": "tpu-vllm-v5e1-12b-w8a8emb4",
    "gemma-4-12B-it-qat-q4_0-fp8-text": "tpu-vllm-v5e1-12b-fp8",
    "gemma-4-12B-it-W8A8-INT8-glenic-notiedhead": "tpu-vllm-v5e1-12b-w8a8rtn",
    "gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct": "tpu-vllm-v5e1-26b-q4w4a16",
    "gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct-text-emb4": "tpu-vllm-v5e1-26b-q4w4a16emb4",
}
NEW_RIGS = {
    "tpu-vllm-v5e1-4b-w8a8rtn": (
        "glenic/gemma-4-E4B-it-W8A8-INT8",
        "w8a8rtn",
        "int8 W8A8 rounded to nearest from the bf16 release, served with its stored `lm_head` "
        "(byte-identical to the tied `embed_tokens`) removed by `jev-tpu-v5e1/strip_tied_head.py`",
    ),
    "tpu-vllm-v5e1-12b-w8a8rtn": (
        "glenic/gemma-4-12B-it-W8A8-INT8",
        "w8a8rtn",
        "int8 W8A8 rounded to nearest from the bf16 release, served with its stored `lm_head` "
        "(byte-identical to the tied `embed_tokens`) removed by `jev-tpu-v5e1/strip_tied_head.py`",
    ),
    "tpu-vllm-v5e1-12b-fp8": (
        "gemma-4-12B-it-qat-q4_0-fp8-text",
        "fp8",
        "fp8 W8A8 from the QAT weights, text only (local build)",
    ),
    "tpu-vllm-v5e1-26b-q4w4a16emb4": (
        "xbill9/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct-text-emb4",
        "q4w4a16emb4",
        "the QAT W4A16 repack with `embed_tokens` and an untied `lm_head` int4, text only",
    ),
}
TEMPLATE_RIG = "tpu-vllm-v5e1-2b-w8a8rtn"


def sh(*cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def pull(run):
    res = os.path.join(HERE, "results")
    for sub, dest in (("results", res), ("logs", os.path.join(res, f"{run}-logs"))):
        os.makedirs(dest, exist_ok=True)
        subprocess.run(
            ["gcloud", "storage", "rsync", "-r", "-q", f"{BUCKET}/{run}/{sub}", dest], check=False, capture_output=True
        )


def arms(run):
    """[(tag, checkpoint path as served)] in the order run.log served them."""
    log = os.path.join(HERE, "results", f"{run}-logs", "run.log")
    out = []
    for m in re.finditer(r"\] serving (\S+) as (\S+) \(", open(log, errors="replace").read()):
        if m.group(2) not in [t for t, _ in out]:
            out.append((m.group(2), m.group(1)))
    return out


def new_rig(rig):
    model, token, what = NEW_RIGS[rig]
    d = os.path.join(ROOT, rig)
    os.makedirs(os.path.join(d, "benchmarks", "runs"), exist_ok=True)
    t = os.path.join(ROOT, TEMPLATE_RIG)
    claude = (
        open(os.path.join(t, "CLAUDE.md"))
        .read()
        .replace(TEMPLATE_RIG, rig)
        .replace("glenic/gemma-4-E2B-it-W8A8-INT8", model)
    )
    open(os.path.join(d, "CLAUDE.md"), "w").write(claude)
    open(os.path.join(d, "README.md"), "w").write(
        f"# {rig}\n\nResults for one cell of the Gemma 4 repack sweeps: **`{model}`** through vLLM on "
        f"**TPU v5e-1 (`v5litepod-1`)**, provisioned as a Cloud TPU queued resource. An artifact rig: "
        f"measurements only.\n\nSlot 5, `{token}`: {what}.\n\n"
    )
    for f in ("README.md", "serving-report.schema.json"):
        shutil.copy(os.path.join(ROOT, "benchmarks", f), os.path.join(d, "benchmarks", f))


def run_readme(run, tag, model, dirs, comparisons):
    rel = "../../../../jev-tpu-v5e1"
    lines = [
        f"# {os.path.basename(dirs['dest'])}",
        "",
        f"One arm of the `{run}` run of the sweep in `{rel}/`, filed here because this rig names its chip and "
        "checkpoint.",
        "",
        "| | |",
        "|---|---|",
        f"| Checkpoint | `{model}` |",
        "| Hardware | `v5e1` |",
        f"| Arm | `{tag}` (harness id `{run}-{tag}`) |",
    ]
    if dirs["moved"]:
        lines.append("| Results | " + ", ".join(f"`{d}/`" for d in dirs["moved"]) + " |")
    elif dirs["throughput"]:
        lines.append(f"| Results | throughput only: `logs/{run}-logs/{tag}.load*.json` / `{tag}.long*.json` |")
    else:
        lines.append("| Results | none: the arm did not serve; its boot log says why |")
    lines += [
        "| This arm's logs | `logs/<run log dir>/`"
        + (
            ". No boot log: the next arm of this run served the same checkpoint and the VM overwrote it; "
            "`run.log` has this arm's result |"
            if dirs.get("boot_lost")
            else " |"
        ),
        f"| Run-wide logs (run.log, patches, checksums) | `{rel}/results/{run}-logs/` |",
        "",
    ]
    if comparisons:
        lines += ["Paired comparisons that include this run (they cover two cells, so they stay with the sweep):", ""]
        lines += [f"- `{rel}/results/{c}`" for c in comparisons]
    return "\n".join(lines) + "\n"


def file_run(run, dry):
    pull(run) if not dry else None
    res = os.path.join(HERE, "results")
    logs = os.path.join(res, f"{run}-logs")
    date, short = run[:10], run[11:].rsplit("-v5e1", 1)[0]
    moves = {}
    served_by = {}
    for tag, served in arms(run):
        served_by[served] = tag  # the VM names boot logs by checkpoint, so the last arm to serve one owns it
    for tag, served in arms(run):
        model = os.path.basename(served)
        rig = RIGS.get(model)
        if rig is None:
            sys.exit(f"{run}: no rig for checkpoint {served} (arm {tag}); add it to RIGS")
        if not os.path.isdir(os.path.join(ROOT, rig)):
            if rig not in NEW_RIGS:
                sys.exit(f"{rig} does not exist and is not in NEW_RIGS")
            print(f"create artifact rig {rig}")
            dry or new_rig(rig)
        dest = os.path.join(ROOT, rig, "benchmarks", "runs", f"{date}-{short}-{tag}-v5e1")
        moved = []
        for s in SUFFIXES:
            src = os.path.join(res, f"{run}-{tag}{s}")
            if os.path.isdir(src) and not os.path.islink(src):
                moved.append(f"{run}-{tag}{s}")
                print(f"  {os.path.relpath(src, ROOT)} -> {os.path.relpath(dest, ROOT)}/")
                if not dry:
                    os.makedirs(dest, exist_ok=True)
                    shutil.move(src, os.path.join(dest, f"{run}-{tag}{s}"))
                    os.symlink(os.path.relpath(os.path.join(dest, f"{run}-{tag}{s}"), res), src)
                moves[os.path.relpath(src, ROOT)] = os.path.relpath(os.path.join(dest, f"{run}-{tag}{s}"), ROOT)
        arm_logs = [f for f in os.listdir(logs) if f.startswith(tag + ".")]
        boot = served.replace("/", "_") + ".boot.log"
        boot_lost = served_by[served] != tag
        if os.path.exists(os.path.join(logs, boot)) and not boot_lost:
            arm_logs.append(boot)
        print(f"  {run} {tag} -> {rig}: {len(moved)} result dirs, {len(arm_logs)} log files")
        if dry:
            continue
        os.makedirs(os.path.join(dest, "logs", f"{run}-logs"), exist_ok=True)
        for f in arm_logs:
            shutil.copy(os.path.join(logs, f), os.path.join(dest, "logs", f"{run}-logs", f))
        # Only the run's comparisons that name this arm: a run's comparison files cover all its arms.
        comps = sorted(
            f
            for f in os.listdir(res)
            if f.startswith(run)
            and ("VS" in f or "QUANT" in f)
            and f.endswith(".md")
            and re.search(rf"(^|[#\s|]){re.escape(tag)}(\s|$)", open(os.path.join(res, f)).read(), re.M)
        )
        open(os.path.join(dest, "README.md"), "w").write(
            run_readme(
                run,
                tag,
                served,
                {"dest": dest, "moved": moved, "boot_lost": boot_lost,
                    "throughput": any(".load." in f or ".long" in f for f in arm_logs)},
                comps,
            )
        )
    return moves


def main():
    args = sys.argv[1:]
    dry = "--dry-run" in args
    runs = [a for a in args if a != "--dry-run"]
    allmoves = {}
    for r in runs:
        print(r)
        allmoves.update(file_run(r, dry))
    if dry:
        return
    path = os.path.join(ROOT, "benchmarks", "sweep-moves.json")
    m = json.load(open(path))
    m.update(allmoves)
    json.dump(dict(sorted(m.items())), open(path, "w"), indent=1)
    open(path, "a").write("\n")
    print(f"sweep-moves.json: {len(allmoves)} new entries, {len(m)} total")


if __name__ == "__main__":
    main()
