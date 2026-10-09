"""Run a Gemma 4 data-type sweep on one MI300X: every arm of one model size, in order.

E2B (`--size 2b`, the default) is the sweep DTYPE-SWEEP.md preregistered: each arm is a
sibling serving rig that differs only in the checkpoint lines of `tpu.env`, and this
driver deploys that rig's own server.py (through its own
`benchmarking_suite._load_server`, so nothing is reimplemented).

E4B, 12B, 26B-A4B and 31B (`--size 4b|12b|26b|31b`) have no serving rig per arm. Their
arms serve through one engine rig, `gpu-vllm-mi300x-2b-q4w4a16`, whose server.py takes
every checkpoint setting from the environment, and each arm is filed in the artifact
rig named for its chip and checkpoint (`SUITE_OUTPUT_DIR`). A checkpoint that is not on
Hugging Face yet is served from the droplet's disk under `/opt/hf-cache/local/`, which
the engine mounts at `/root/.cache/huggingface/local/`; its report names the repo it is
meant for (`REPORT_MODEL_ID`).

For each arm: download, deploy, wait for the endpoint, file the boot log and the
capability probe, run the suite over the preregistered grid, stop the container.

Every arm runs on ONE image digest, passed in as --image and exported as VLLM_IMAGE
(a real environment variable wins over tpu.env). Every arm gets its own --seed-base, so
no arm reads another's prefix-cache entries.

    python3 dtype_sweep.py --droplet <name> --image vllm/vllm-openai-rocm@sha256:… \\
        --run-id 2026-10-XX-dtype-sweep-mi300x [--size 4b]

An arm that does not come up is recorded as failed with its log and the sweep moves
on; `dtype-summary.json`, under the size's bf16 rig, is computed here, never by hand.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent

# Serves every arm of the larger sizes; its server.py reads all checkpoint settings from env.
ENGINE = "gpu-vllm-mi300x-2b-q4w4a16"
LOCAL = "/root/.cache/huggingface/local/"  # HF_CACHE/local/ on the droplet
# The droplet's 5 TB scratch disk (mounted by hand at /mnt/scratch): the root disk cannot hold
# the larger sizes' checkpoints beside the local builds.
ENGINE_HF_CACHE = "/mnt/scratch/hf-cache"

# The bf16 checkpoints are multimodal; every arm here serves text only.
TEXT_ONLY = '{"image": 0, "audio": 0}'


@dataclass
class Arm:
    enc: str
    rig: str
    model: Optional[str] = None  # None: the rig's own server.py and tpu.env (E2B)
    local: bool = False  # served from LOCAL on the droplet, not pulled from the Hub
    env: dict = field(default_factory=dict)

    @property
    def server_rig(self) -> str:
        return self.rig if self.model is None else ENGINE


# Per encoding: (EXPECT_MP_KERNEL, WEIGHTS_DTYPE, CHECKPOINT_QUANTIZATION), as the E2B rigs spell them.
LABELS = {
    "bf16": ("", "bfloat16", "none"),
    "q4w4a16": ("TritonW4A16LinearKernel", "int4", "qat-q4_0-repack-w4a16-compressed-tensors-g32"),
    "q4w4a16emb4": ("TritonW4A16LinearKernel", "int4", "qat-q4_0-repack-w4a16-compressed-tensors-g32-emb4"),
    "q4w4a16ple4": ("TritonW4A16LinearKernel", "int4", "qat-q4_0-repack-w4a16-compressed-tensors-g32-ple4"),
    "w8a8": ("", "int8", "qat-w8a8-int8-compressed-tensors-channel-dyntoken"),
    "w8a8emb4": ("", "int8", "qat-w8a8-int8-compressed-tensors-emb4"),
    "fp8": ("", "fp8", "qat-q4_0-fp8-e4m3fn-compressed-tensors-channel-dyntoken"),
    "fp8emb4": ("", "fp8", "qat-q4_0-fp8-e4m3fn-compressed-tensors-emb4"),
    "fp8fnuz": ("", "fp8", "qat-q4_0-fp8-e4m3fnuz-compressed-tensors-channel-dyntoken"),
    "fp8fnuzemb4": ("", "fp8", "qat-q4_0-fp8-e4m3fnuz-compressed-tensors-emb4"),
}


def engine_arm(size: str, params_b: str, enc: str, repo: str, local: bool = False) -> Arm:
    kernel, wdtype, quant = LABELS[enc]
    name = repo.split("/", 1)[1]
    env = {
        "VLLM_MODEL": LOCAL + name if local else repo,
        "LIMIT_MM_PER_PROMPT": TEXT_ONLY if enc == "bf16" else "",
        "EXPECT_MP_KERNEL": kernel,
        "MODEL_PARAMETERS_B": params_b,
        "WEIGHTS_DTYPE": wdtype,
        "CHECKPOINT_QUANTIZATION": quant,
        "HF_CACHE": ENGINE_HF_CACHE,
    }
    if local:
        env["REPORT_MODEL_ID"] = repo
    rig = f"gpu-vllm-mi300x-{size}" + ("" if enc == "bf16" else f"-{enc}")
    return Arm(enc, rig, model=repo, local=local, env=env)


def size_arms(size: str, params_b: str, google: str, repos: dict) -> list[Arm]:
    """repos: encoding -> (repo, local). bf16 first: it is the reference every ratio divides by."""
    arms = [engine_arm(size, params_b, "bf16", google)]
    return arms + [engine_arm(size, params_b, enc, repo, local) for enc, (repo, local) in repos.items()]


X = "xbill9/gemma-4-"
SIZES = {
    "2b": [
        Arm("bf16", "gpu-vllm-mi300x-2b", env={"LIMIT_MM_PER_PROMPT": TEXT_ONLY}),
        Arm("q4w4a16", "gpu-vllm-mi300x-2b-q4w4a16"),
        Arm("w8a8", "gpu-vllm-mi300x-2b-w8a8"),
        Arm("fp8", "gpu-vllm-mi300x-2b-fp8"),
        Arm("q4w4a16ple4", "gpu-vllm-mi300x-2b-q4w4a16ple4"),
        Arm("q4w4a16emb4", "gpu-vllm-mi300x-2b-q4w4a16emb4"),
        Arm("w8a8emb4", "gpu-vllm-mi300x-2b-w8a8emb4"),
        Arm("fp8emb4", "gpu-vllm-mi300x-2b-fp8emb4"),
        # Added after the eight above were preregistered: the same fp8 builds stored as e4m3fnuz.
        engine_arm("2b", "2", "fp8fnuz", X + "E2B-it-qat-q4_0-fp8fnuz-text", local=True),
        engine_arm("2b", "2", "fp8fnuzemb4", X + "E2B-it-qat-q4_0-fp8fnuz-text-emb4", local=True),
    ],
    "4b": size_arms(
        "4b",
        "4",
        "google/gemma-4-E4B-it",
        {
            "q4w4a16": (X + "E4B-it-qat-q4_0-w4a16-ct-text", False),
            "w8a8": (X + "E4B-it-qat-w8a8-int8", False),
            "fp8": (X + "E4B-it-qat-q4_0-fp8-text", False),
            "q4w4a16ple4": (X + "E4B-it-qat-q4_0-w4a16-ct-text-ple4", True),
            "q4w4a16emb4": (X + "E4B-it-qat-q4_0-w4a16-ct-text-emb4", False),
            "w8a8emb4": (X + "E4B-it-qat-w8a8-int8-emb4", False),
            "fp8emb4": (X + "E4B-it-qat-q4_0-fp8-text-emb4", True),
        },
    ),
    "12b": size_arms(
        "12b",
        "12",
        "google/gemma-4-12B-it",
        {
            "q4w4a16": (X + "12B-it-qat-q4_0-w4a16-ct-text", False),
            "w8a8": (X + "12B-it-qat-w8a8-int8", False),
            "fp8": (X + "12B-it-qat-q4_0-fp8-text", False),
            "q4w4a16emb4": (X + "12B-it-qat-q4_0-w4a16-ct-text-emb4", False),
            "w8a8emb4": (X + "12B-it-qat-w8a8-int8-emb4", False),
            "fp8emb4": (X + "12B-it-qat-q4_0-fp8-text-emb4", True),
        },
    ),
    "26b": size_arms(
        "26b",
        "26",
        "google/gemma-4-26B-A4B-it",
        {
            "q4w4a16": (X + "26B-A4B-it-qat-q4_0-w4a16-ct-text", False),
            "w8a8": (X + "26B-A4B-it-qat-w8a8-int8", True),
            "fp8": (X + "26B-A4B-it-qat-q4_0-fp8-text", True),
            "q4w4a16emb4": (X + "26B-A4B-it-qat-q4_0-w4a16-ct-text-emb4", False),
            "w8a8emb4": (X + "26B-A4B-it-qat-w8a8-int8-emb4", True),
            "fp8emb4": (X + "26B-A4B-it-qat-q4_0-fp8-text-emb4", True),
        },
    ),
    "31b": size_arms(
        "31b",
        "31",
        "google/gemma-4-31B-it",
        {
            "q4w4a16": (X + "31B-it-qat-q4_0-w4a16-ct-text", False),
            "w8a8": (X + "31B-it-qat-w8a8-int8", True),
            "fp8": (X + "31B-it-qat-q4_0-fp8-text", True),
            "q4w4a16emb4": (X + "31B-it-qat-q4_0-w4a16-ct-text-emb4", False),
            "w8a8emb4": (X + "31B-it-qat-w8a8-int8-emb4", True),
            "fp8emb4": (X + "31B-it-qat-q4_0-fp8-text-emb4", True),
        },
    ),
}

GRID = ["--concurrency", "1", "8", "64", "--input-len", "128", "1024", "8192", "--output-len", "512", "--repeat", "3"]
BOOT_TIMEOUT_S = 3600  # 31B bf16 pulls 62 GB and compiles longer than E2B's 30 minutes allowed
BOOT_POLL_S = 20
BOOT_LOG_PATTERNS = re.compile(
    r"Kernel|Using |ScaledMM|scaled_mm|Model loading took|KV cache size|Maximum concurrency|"
    r"quantiz|fp8|fnuz|MoE|Error|Traceback|raise ",
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


ENGINE_ENV = ""  # --engine-env: KEY=VALUE pairs for the model server's container


def _env(arm: Arm, image: str) -> dict:
    env = dict(os.environ, VLLM_IMAGE=image, **arm.env)
    if ENGINE_ENV:
        env["VLLM_DOCKER_ENV"] = ENGINE_ENV
    if arm.model is not None:
        env["SUITE_OUTPUT_DIR"] = str(ROOT / arm.rig)
    return env


def call(arm: Arm, image: str, fn: str, timeout: int = 900, **kwargs) -> str:
    out = subprocess.run(
        [sys.executable, __file__, "tool", arm.server_rig, fn, json.dumps(kwargs)],
        env=_env(arm, image),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return out.stdout + (f"\n[stderr]\n{out.stderr}" if out.returncode else "")


def wait_ready(arm: Arm, image: str, droplet: str) -> tuple[bool, float, str]:
    start = time.monotonic()
    status = ""
    while time.monotonic() - start < BOOT_TIMEOUT_S:
        status = call(arm, image, "serving_status", droplet=droplet)
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
    facts["kernels"] = sorted(set(re.findall(r"\b(\w+(?:LinearKernel|ScaledMM\w*|MMLinearKernel|MoEMethod))\b", log)))
    return facts


def run_arm(i: int, n: int, arm: Arm, args) -> dict:
    run_dir = ROOT / arm.rig / "benchmarks" / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {"encoding": arm.enc, "rig": arm.rig, "run_dir": str(run_dir.relative_to(ROOT))}
    if arm.model is not None:
        result["model"] = arm.model
        result["served_from"] = "droplet disk (not on the Hub yet)" if arm.local else "huggingface"
    print(f"\n=== [{i + 1}/{n}] {arm.enc} ({arm.rig})", flush=True)

    if not arm.local:
        (run_dir / "download.txt").write_text(
            call(arm, args.image, "download_weights", timeout=3600, droplet=args.droplet)
        )
    (run_dir / "deploy.txt").write_text(call(arm, args.image, "deploy_vllm", droplet=args.droplet, replace=True))
    ready, secs, status = wait_ready(arm, args.image, args.droplet)
    result["boot_seconds"] = round(secs, 1)
    (run_dir / "serving_status.txt").write_text(status)

    log = call(arm, args.image, "server_logs", droplet=args.droplet, lines=20000)
    (run_dir / "boot.log").write_text(log)
    (run_dir / "boot-highlights.txt").write_text(
        "\n".join(ln for ln in log.splitlines() if BOOT_LOG_PATTERNS.search(ln))
    )
    result.update(boot_facts(log))

    if not ready:
        result["status"] = "failed"
        print(f"    FAILED to serve after {secs:.0f}s — log filed", flush=True)
        call(arm, args.image, "stop_vllm", droplet=args.droplet)
        return result

    (run_dir / "verify_capabilities.txt").write_text(call(arm, args.image, "verify_capabilities", droplet=args.droplet))
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
        cwd=ROOT / arm.server_rig,
        env=_env(arm, args.image),
        capture_output=True,
        text=True,
    )
    (run_dir / "suite.txt").write_text(bench.stdout + bench.stderr)
    report = ROOT / arm.rig / "benchmarks" / "reports" / f"{args.run_id}.json"
    result["status"] = "ok" if bench.returncode == 0 and report.exists() else "bench-failed"
    result["report"] = str(report.relative_to(ROOT)) if report.exists() else None
    call(arm, args.image, "stop_vllm", droplet=args.droplet)
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
    p.add_argument("--size", default="2b", choices=sorted(SIZES))
    p.add_argument("--seed-base", type=int, default=50_000_000)
    p.add_argument("--only", nargs="+", help="run these encodings only")
    p.add_argument("--engine-env", default="", help="KEY=VALUE[,KEY=VALUE] for the server container, e.g. VLLM_ROCM_USE_AITER=1")
    args = p.parse_args()
    global ENGINE_ENV
    ENGINE_ENV = args.engine_env
    if "@sha256:" not in args.image:
        sys.exit("--image must be a pinned digest: arms on different nightlies are not comparable")

    arms = SIZES[args.size]
    missing = [a.rig for a in arms if not (ROOT / a.rig).is_dir()]
    if missing:
        sys.exit(f"no rig directory for: {', '.join(missing)}")
    summary_path = ROOT / arms[0].rig / "benchmarks" / "runs" / args.run_id / "dtype-summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    (summary_path.parent / "engine-env.txt").write_text((ENGINE_ENV or "(none: image defaults)") + "\n")
    results = json.loads(summary_path.read_text())["arms"] if summary_path.exists() else []
    done = {r["encoding"] for r in results if r.get("status") in ("ok", "failed")}
    for i, arm in enumerate(arms):
        if arm.enc in done or (args.only and arm.enc not in args.only):
            continue
        results = [r for r in results if r["encoding"] != arm.enc] + [run_arm(i, len(arms), arm, args)]
        summary_path.write_text(json.dumps(summarize(results), indent=2) + "\n")
    summary_path.write_text(json.dumps(summarize(results), indent=2) + "\n")
    print(f"\nwrote {summary_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
