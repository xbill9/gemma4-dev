"""Run the E2B data-type sweep on one MI300X: every arm in DTYPE-SWEEP.md, in order.

Each arm is a sibling rig that differs only in the checkpoint lines of
`tpu.env`. For each one this driver deploys the rig's own server.py (through
its own `benchmarking_suite._load_server`, so nothing is reimplemented), waits
for the endpoint, files the boot log and the capability probe, runs the rig's
`benchmarking_suite.py` over the preregistered grid, and stops the container.

Every arm runs on ONE image digest, passed in as --image and exported as
VLLM_IMAGE (a real environment variable wins over tpu.env). Every arm gets its
own --seed-base, so no arm reads another's prefix-cache entries.

    python3 dtype_sweep.py --droplet <name> --image vllm/vllm-openai-rocm@sha256:… \
        --run-id 2026-10-XX-dtype-sweep-mi300x

An arm that does not come up is recorded as failed with its log and the sweep
moves on; `summary.json` beside this file's DTYPE-SWEEP.md is computed here,
never by hand.

Stdlib only.
"""

import argparse
import asyncio
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent

# Order is the run order. bf16 first: it is the reference every ratio divides by.
ARMS = [
    ("bf16", "gpu-vllm-mi300x-2b"),
    ("q4w4a16", "gpu-vllm-mi300x-2b-q4w4a16"),
    ("w8a8", "gpu-vllm-mi300x-2b-w8a8"),
    ("fp8", "gpu-vllm-mi300x-2b-fp8"),
    ("q4w4a16ple4", "gpu-vllm-mi300x-2b-q4w4a16ple4"),
    ("q4w4a16emb4", "gpu-vllm-mi300x-2b-q4w4a16emb4"),
    ("w8a8emb4", "gpu-vllm-mi300x-2b-w8a8emb4"),
    ("fp8emb4", "gpu-vllm-mi300x-2b-fp8emb4"),
]

# The bf16 rig's committed tpu.env admits images; every arm here serves text only.
TEXT_ONLY = '{"image": 0, "audio": 0}'

GRID = ["--concurrency", "1", "8", "64", "--input-len", "128", "1024", "8192", "--output-len", "512", "--repeat", "3"]
BOOT_TIMEOUT_S = 1800
BOOT_POLL_S = 20
BOOT_LOG_PATTERNS = re.compile(
    r"Kernel|Using |ScaledMM|scaled_mm|Model loading took|KV cache size|Maximum concurrency|"
    r"quantiz|fp8|fnuz|Error|Traceback|raise ",
    re.I,
)


# --------------------------------------------------------------------------- tool mode


def _load_rig_server(rig: Path):
    spec = importlib.util.spec_from_file_location("benchmarking_suite", rig / "benchmarking_suite.py")
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    return suite._load_server()


def tool_main(rig: str, fn: str, kwargs: str) -> int:
    """Call one of a rig's MCP tools in a fresh process and print its markdown.

    `server_logs` is the exception: the tool truncates at 6,000 characters, which
    cuts the boot log before the kernel, model-loading and KV-cache lines. Here it
    is a full `docker logs` through the rig's own ssh path, request lines dropped.
    """
    os.chdir(ROOT / rig)
    srv = _load_rig_server(ROOT / rig)
    args = json.loads(kwargs)
    if fn == "server_logs":
        cmd = f"docker logs {srv.VLLM_CONTAINER} 2>&1 | grep -v -e 'GET /metrics' -e 'POST /v1'"
        rc, out, err = asyncio.run(srv._remote(args["droplet"], cmd))
        print(out if rc == 0 else f"❌ docker logs exited {rc}\n{err}")
        return 0
    print(asyncio.run(getattr(srv, fn)(**args)))
    return 0


# --------------------------------------------------------------------------- driver


def _env(enc: str, image: str) -> dict:
    env = dict(os.environ, VLLM_IMAGE=image)
    if enc == "bf16":
        env["LIMIT_MM_PER_PROMPT"] = TEXT_ONLY
    return env


def call(rig: str, enc: str, image: str, fn: str, timeout: int = 900, **kwargs) -> str:
    out = subprocess.run(
        [sys.executable, __file__, "tool", rig, fn, json.dumps(kwargs)],
        env=_env(enc, image),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return out.stdout + (f"\n[stderr]\n{out.stderr}" if out.returncode else "")


def wait_ready(rig: str, enc: str, image: str, droplet: str) -> tuple[bool, float, str]:
    start = time.monotonic()
    status = ""
    while time.monotonic() - start < BOOT_TIMEOUT_S:
        status = call(rig, enc, image, "serving_status", droplet=droplet)
        if status.startswith("✅"):
            return True, time.monotonic() - start, status
        if re.search(r"container: (Exited|Restarting|Dead)", status) or status.startswith("❌"):
            return False, time.monotonic() - start, status
        time.sleep(BOOT_POLL_S)
    return False, time.monotonic() - start, status


def boot_facts(log: str) -> dict:
    facts: dict = {}
    m = re.search(r"Model loading took ([\d.]+) GiB", log)
    if m:
        facts["model_loading_gib"] = float(m.group(1))
    m = re.search(r"KV cache size: ([\d,]+) tokens", log)
    if m:
        facts["kv_tokens"] = int(m.group(1).replace(",", ""))
    facts["kernels"] = sorted(set(re.findall(r"\b(\w+(?:LinearKernel|ScaledMM\w*|MMLinearKernel))\b", log)))
    return facts


def run_arm(i: int, enc: str, rig: str, args) -> dict:
    run_dir = ROOT / rig / "benchmarks" / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {"encoding": enc, "rig": rig, "run_dir": str(run_dir.relative_to(ROOT))}
    print(f"\n=== [{i + 1}/{len(ARMS)}] {enc} ({rig})", flush=True)

    (run_dir / "download.txt").write_text(
        call(rig, enc, args.image, "download_weights", timeout=3600, droplet=args.droplet)
    )
    (run_dir / "deploy.txt").write_text(call(rig, enc, args.image, "deploy_vllm", droplet=args.droplet, replace=True))
    ready, secs, status = wait_ready(rig, enc, args.image, args.droplet)
    result["boot_seconds"] = round(secs, 1)
    (run_dir / "serving_status.txt").write_text(status)

    log = call(rig, enc, args.image, "server_logs", droplet=args.droplet, lines=20000)
    (run_dir / "boot.log").write_text(log)
    (run_dir / "boot-highlights.txt").write_text(
        "\n".join(ln for ln in log.splitlines() if BOOT_LOG_PATTERNS.search(ln))
    )
    result.update(boot_facts(log))

    if not ready:
        result["status"] = "failed"
        print(f"    FAILED to serve after {secs:.0f}s — log filed", flush=True)
        call(rig, enc, args.image, "stop_vllm", droplet=args.droplet)
        return result

    (run_dir / "verify_capabilities.txt").write_text(
        call(rig, enc, args.image, "verify_capabilities", droplet=args.droplet)
    )
    bench = subprocess.run(
        [
            sys.executable,
            "benchmarking_suite.py",
            "--droplet",
            args.droplet,
            "--run-id",
            args.run_id,
            "--seed-base",
            str(args.seed_base + (i + 1) * 1_000_000),
            *GRID,
        ],
        cwd=ROOT / rig,
        env=_env(enc, args.image),
        capture_output=True,
        text=True,
    )
    (run_dir / "suite.txt").write_text(bench.stdout + bench.stderr)
    report = ROOT / rig / "benchmarks" / "reports" / f"{args.run_id}.json"
    result["status"] = "ok" if bench.returncode == 0 and report.exists() else "bench-failed"
    result["report"] = str(report.relative_to(ROOT)) if report.exists() else None
    call(rig, enc, args.image, "stop_vllm", droplet=args.droplet)
    print(
        f"    {result['status']}: loading {result.get('model_loading_gib')} GiB, KV {result.get('kv_tokens')}, "
        f"kernels {result.get('kernels')}",
        flush=True,
    )
    return result


def cells(report_path: Optional[str]) -> dict:
    if not report_path:
        return {}
    rep = json.loads((ROOT / report_path).read_text())
    out = {}
    for c in rep.get("throughput", {}).get("sweep", []):
        if c.get("status", "ok") == "ok":
            out[(c["concurrency"], c["input_len"])] = c
    return out


def summarize(results: list[dict]) -> dict:
    """Ratios against bf16, per cell, computed here — never by reading a table."""
    base = cells(next((r.get("report") for r in results if r["encoding"] == "bf16"), None))
    for r in results:
        mine = cells(r.get("report"))
        r["cells"] = []
        for key, c in sorted(mine.items()):
            ref = base.get(key)
            row = {
                "concurrency": key[0],
                "input_len": key[1],
                "output_tok_s": c.get("output_tok_per_s"),
                "tpot_ms_median": (c.get("tpot_ms") or {}).get("median"),
            }
            if ref and ref.get("output_tok_per_s") and c.get("output_tok_per_s"):
                row["output_ratio_vs_bf16"] = round(c["output_tok_per_s"] / ref["output_tok_per_s"], 3)
            r["cells"].append(row)
        ratios = [x["output_ratio_vs_bf16"] for x in r["cells"] if "output_ratio_vs_bf16" in x]
        if ratios:
            r["ratio_min"], r["ratio_max"] = min(ratios), max(ratios)
            r["ratio_cells"] = len(ratios)
    return {"arms": results}


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "tool":
        return tool_main(*sys.argv[2:5])
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--droplet", required=True)
    p.add_argument("--image", required=True, help="pinned digest, vllm/vllm-openai-rocm@sha256:…")
    p.add_argument("--run-id", required=True)
    p.add_argument("--seed-base", type=int, default=50_000_000)
    p.add_argument("--only", nargs="+", help="run these encodings only")
    args = p.parse_args()
    if "@sha256:" not in args.image:
        sys.exit("--image must be a pinned digest: arms on different nightlies are not comparable")

    summary_path = Path(__file__).resolve().parent / "benchmarks" / "runs" / args.run_id / "dtype-summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(summary_path.read_text())["arms"] if summary_path.exists() else []
    done = {r["encoding"] for r in results if r.get("status") in ("ok", "failed")}
    for i, (enc, rig) in enumerate(ARMS):
        if enc in done or (args.only and enc not in args.only):
            continue
        results = [r for r in results if r["encoding"] != enc] + [run_arm(i, enc, rig, args)]
        summary_path.write_text(json.dumps(summarize(results), indent=2) + "\n")
    summary_path.write_text(json.dumps(summarize(results), indent=2) + "\n")
    print(f"\nwrote {summary_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
