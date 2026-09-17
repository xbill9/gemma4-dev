"""Drive a 2-D serving sweep against the MI300X and emit a schema-valid report.

The sibling vLLM rigs carry a `benchmarking_suite.py` that talks to the
endpoint directly with httpx. That cannot work here: the serving port is
published on the droplet's own interface and nothing promises a firewall lets
this workstation reach it, which is why `server.py` curls `127.0.0.1` over SSH
rather than dialling the public address. So this suite is a *driver* — it
loops the grid and hands each cell to `server.bench_cell`, the same code path
the `run_vllm_benchmark` MCP tool uses. The docker argv and the result parsing
have exactly one implementation; this file only decides which cells to run and
what the report around them says.

Grid defaults match `tpu-vllm-v5e1-2b` and `tpu-vllm-v6e1-2b` so the shape of
the sweep is comparable. **The numbers are not**: nothing else in this
monorepo serves this checkpoint on AMD, so a figure from here must never be
differenced against a TPU rig and read as a hardware result.

    python3 benchmarking_suite.py --droplet <name> --run-id 2026-09-16-...

Nothing is installed: stdlib plus the httpx that `server.py` already needs.
"""

import argparse
import asyncio
import importlib.util
import json
import os
import re
import sys
import time
import types
from datetime import date
from pathlib import Path
from typing import Any, Optional

PROJECT_DIR = Path(__file__).resolve().parent

# The sibling grid: four concurrencies against four context lengths, 128 output
# tokens throughout. Cells that do not fit max_model_len are recorded
# infeasible rather than dropped — schema 1.1 has a status for exactly this,
# and a missing cell is indistinguishable from one nobody ran.
CONCURRENCIES = [1, 4, 16, 64]
INPUT_LENS = [128, 1024, 8192, 32768]
OUTPUT_LEN = 128


class _StubMCPServer:
    """Enough of MCPServer to import server.py without the MCP SDK.

    `tool()` must be a pass-through decorator: a bare MagicMock makes
    `@mcp.tool()` return a mock instead of the coroutine, and every call then
    fails with "object can't be awaited". Same fake the unit tests use.
    """

    def __init__(self, name):
        self.name = name

    def tool(self, *args, **kwargs):
        def decorator(fn):
            return fn

        return decorator

    def run(self):
        raise AssertionError("benchmarking_suite must never start the MCP server")


def _stub_mcp() -> None:
    """Satisfy server.py's `mcp` imports with a stub, deliberately.

    This is a load generator driven from a terminal. It calls `bench_cell` and
    the SSH helpers and never serves a tool, so the MCP SDK is a dependency of
    the module it imports rather than of the work it does — and the SDK's major
    version is shared with every other rig on this machine. Requiring it here
    means a sweep stops because something unrelated changed `mcp`, which is
    what happened on 2026-09-16: the installed package went 2.2.0 -> 1.30.0
    mid-session and `mcp.server.mcpserver` disappeared.
    """
    stub = types.ModuleType("mcp.server.mcpserver")
    stub.MCPServer = _StubMCPServer
    types_module = types.ModuleType("mcp.types")
    types_module.ToolAnnotations = lambda **kwargs: None
    server_pkg = types.ModuleType("mcp.server")
    server_pkg.mcpserver = stub
    root = types.ModuleType("mcp")
    root.server = server_pkg
    sys.modules.setdefault("mcp", root)
    sys.modules["mcp.server"] = server_pkg
    sys.modules["mcp.server.mcpserver"] = stub
    sys.modules["mcp.types"] = types_module


def _load_server() -> Any:
    """Import the rig's server.py without requiring it to be on sys.path."""
    _stub_mcp()
    spec = importlib.util.spec_from_file_location("server", PROJECT_DIR / "server.py")
    if spec is None or spec.loader is None:  # pragma: no cover - import plumbing
        raise RuntimeError("could not load server.py next to this script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _num_prompts(concurrency: int) -> int:
    """Two rounds of every in-flight slot, with a floor of 8.

    The floor matters at concurrency 1: fewer than 8 requests and one slow
    first request dominates the median. The rule is the sibling reports'.
    """
    return max(8, 2 * concurrency)


def plan(concurrencies: list[int], input_lens: list[int], output_len: int, max_model_len: int) -> list[dict]:
    """Decide every cell up front, marking the ones the context window forbids."""
    cells = []
    for input_len in input_lens:
        for concurrency in concurrencies:
            cell: dict = {"concurrency": concurrency, "input_len": input_len, "output_len": output_len}
            if input_len + output_len > max_model_len:
                cell["status"] = "infeasible"
                cell["error"] = (
                    f"input_len {input_len} + output_len {output_len} exceeds configured max_model_len {max_model_len}"
                )
            else:
                cell["status"] = "pending"
            cells.append(cell)
    return cells


def _cv_pct(values: list[float]) -> Optional[float]:
    """Coefficient of variation, as a percentage. None below two samples."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    if not mean:
        return None
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return round(100 * variance**0.5 / mean, 2)


def _reduce(points: list[dict]) -> dict:
    """Pick the median-throughput repeat as the cell, and record the spread.

    The median run is reported rather than the mean of the runs because every
    latency field is itself a distribution: averaging a p99 across repeats
    produces a number no single run ever saw. Taking one real run keeps the
    cell internally consistent, and `repeats` says how far apart the runs were
    so a reader can tell a 3% difference from a real one.
    """
    ordered = sorted(points, key=lambda p: p.get("output_tok_per_s") or 0)
    chosen = dict(ordered[len(ordered) // 2])
    rates = [p["output_tok_per_s"] for p in points if isinstance(p.get("output_tok_per_s"), (int, float))]
    if len(points) > 1:
        chosen.setdefault("raw", {})
        chosen["raw"] = dict(chosen["raw"])
        chosen["raw"]["repeats"] = {
            "n": len(points),
            "output_tok_per_s": rates,
            "cv_pct": _cv_pct(rates),
        }
    return chosen


def _seed(base: int, index: int, rep: int) -> int:
    """A seed no other cell or repeat in the sweep will use.

    MEASURED 2026-09-16, and it cost a whole sweep. `vllm bench serve` seeds at
    0 and derives its prompts from the seed, so two runs sharing one generate
    the same prompts — and prefix caching is on, so the second is a cache read.
    Seeding per repeat alone is not enough: cells at the same context length
    draw from the same pool, so `c4-in128` replayed `c1-in128`'s prompts. The
    contaminated sweep is kept as `...-seedcollision` because the size of the
    effect is itself a result: 2.2x at 8192 context.
    """
    return base + index * 100 + rep


async def run_sweep(
    srv: Any,
    droplet: str,
    cells: list[dict],
    log_dir: Optional[Path],
    repeat: int = 1,
    seed_base: int = 0,
) -> list[dict]:
    """Run every pending cell in order, recording failures rather than raising.

    A cell that fails is recorded `failed` with the error and the sweep
    continues: one bad cell should not cost the other eleven, and an
    unexplained gap in a report is worse than a recorded failure. A cell whose
    repeats partly fail keeps the repeats that worked.
    """
    done = []
    for index, cell in enumerate(cells, start=1):
        label = f"c{cell['concurrency']}-in{cell['input_len']}-out{cell['output_len']}"
        if cell["status"] == "infeasible":
            print(f"[{index}/{len(cells)}] {label}: infeasible — {cell['error']}", flush=True)
            done.append(cell)
            continue

        prompts = _num_prompts(cell["concurrency"])
        print(f"[{index}/{len(cells)}] {label}: {prompts} prompts x{repeat} …", end=" ", flush=True)
        started = time.time()
        points: list[dict] = []
        error = None
        for rep in range(repeat):
            try:
                point, stdout = await srv.bench_cell(
                    droplet,
                    num_prompts=prompts,
                    input_len=cell["input_len"],
                    output_len=cell["output_len"],
                    max_concurrency=cell["concurrency"],
                    seed=_seed(seed_base, index, rep),
                )
            except Exception as exc:  # noqa: BLE001 - recorded, not raised
                error = str(exc)[:500]
                continue
            points.append(point)
            if log_dir:
                suffix = f".rep{rep + 1}" if repeat > 1 else ""
                (log_dir / f"{label}{suffix}.log").write_text(stdout)

        elapsed = time.time() - started
        if not points:
            print(f"FAILED after {elapsed:.0f}s")
            cell["status"] = "failed"
            cell["error"] = error or "every repeat failed"
            done.append(cell)
            continue

        reduced = _reduce(points)
        spread = (reduced.get("raw", {}).get("repeats") or {}).get("cv_pct")
        print(
            f"{reduced.get('output_tok_per_s')} out tok/s in {elapsed:.0f}s"
            + (f" (cv {spread}%)" if spread is not None else "")
        )
        done.append(reduced)
    return done


def _worst_cv(sweep: list[dict]) -> Optional[float]:
    """The widest spread any cell saw — the report's honest noise floor."""
    seen = [(c.get("raw", {}).get("repeats") or {}).get("cv_pct") for c in sweep if isinstance(c.get("raw"), dict)]
    values = [v for v in seen if isinstance(v, (int, float))]
    return max(values) if values else None


def build_report(
    srv: Any,
    run_id: str,
    sweep: list[dict],
    facts: dict,
    operator: Optional[str] = None,
    repeat: int = 1,
) -> dict:
    """Assemble the serving-report.schema.json v1.1 document.

    Everything the engine told us goes in `software` and `memory`; everything
    that is arithmetic rather than measurement is labelled as such in `notes`.
    """
    return {
        "schema_version": "1.1",
        "run": {
            "id": run_id,
            "date": date.today().isoformat(),
            **({"operator": operator} if operator else {}),
            "source": f"benchmarks/runs/{run_id}/",
            "notes": (
                "Single run per cell, no repeats — treat differences under a few percent as noise. "
                "The 32768-context row is recorded infeasible from max_model_len, not attempted. "
                "Load was generated by vLLM's own bench client in a separate container with no GPU "
                "device attached, on the same host as the server."
            ),
        },
        "hardware": {
            "accelerator": "amd-mi300x",
            "chips": 1,
            "topology": "1x1",
            "hbm_gb_per_chip": 192,
            "machine_type": facts.get("size_slug"),
            "host": {
                "cloud": "digitalocean",
                "zone": facts.get("region"),
                "provisioning": "on-demand",
                "instance_name": facts.get("droplet_name"),
            },
            **({"pricing": facts["pricing"]} if facts.get("pricing") else {}),
        },
        "model": {
            "id": srv.VLLM_MODEL,
            "family": "gemma-4",
            "parameters_b": 2,
            "weights_dtype": "bfloat16",
            "quantization": "none",
            "max_model_len": int(srv.MAX_MODEL_LEN),
            "architecture_notes": (
                "E2B is hybrid: sliding-attention layers are 256-dim and full-attention layers 512-dim. "
                "That is the same split that makes the newest rocm/vllm image raise "
                "AmbiguousGlobalPerLayerAttributeError on head_dim — see this rig's CLAUDE.md."
            ),
        },
        "software": {
            "engine": "vllm",
            "version": facts.get("vllm_version", "unknown"),
            "container_image": srv.VLLM_IMAGE,
            "backend": "rocm (PyTorch, gfx942)",
            "tensor_parallel_size": 1,
            "serve_args": facts.get("serve_args", []),
        },
        "throughput": {
            "workload": {
                "tool": "vllm bench serve",
                "dataset": "random",
                "output_len": OUTPUT_LEN,
                "runs_per_point": repeat,
                **({"noise_floor_pct": _worst_cv(sweep)} if _worst_cv(sweep) is not None else {}),
                "notes": (
                    "num_prompts = max(8, 2 x concurrency), matching the sibling vLLM reports. "
                    "Each cell reports its median-throughput repeat; raw.repeats carries the spread. "
                    "Every cell and every repeat uses a prompt seed no other run in the sweep uses, so "
                    "no cell reads another's prefix-cache entries."
                ),
            },
            "sweep": sweep,
        },
        **({"memory": facts["memory"]} if facts.get("memory") else {}),
        **({"startup": facts["startup"]} if facts.get("startup") else {}),
        "notes": [
            (
                "Not an A/B twin of anything in this monorepo: nothing else here serves this checkpoint "
                "on AMD, so these figures must not be differenced against a TPU or EC2 rig and read as a "
                "hardware or control-plane result."
            ),
            (
                "Prefix caching is enabled (vLLM's default) and was left on, because that is how the rig "
                "actually serves. Each repeat uses a different bench seed so repeats are independent "
                "samples rather than replays into a warm cache, but the random dataset still shares some "
                "prefixes within a run, so prefill-bound cells are optimistic relative to unique traffic."
            ),
            (
                "The load generator ran on the same host as the server, in a container with no GPU device "
                "attached. It costs host CPU, not card time."
            ),
        ],
    }


async def collect_facts(srv: Any, droplet: str) -> dict:
    """Read the run's metadata off the droplet rather than restating tpu.env.

    tpu.env says what we asked for; the engine's own log says what it did. Where
    they disagree the log is right, and that disagreement is the whole reason
    this is read back instead of copied.
    """
    facts: dict = {}
    item = await srv._resolve(droplet)
    facts["droplet_name"] = item.get("name")
    facts["size_slug"] = item.get("size_slug")
    facts["region"] = (item.get("region") or {}).get("slug")

    size = (item.get("size") or {}).get("price_hourly")
    if isinstance(size, (int, float)):
        facts["pricing"] = {
            "currency": "USD",
            "rate_per_chip_hour": round(float(size), 4),
            "source": (
                f"DigitalOcean v2 API, droplet.size.price_hourly for `{facts['size_slug']}`, "
                f"read {date.today().isoformat()}"
            ),
        }

    _, version, _ = await srv._remote(droplet, f"docker exec {srv.VLLM_CONTAINER} vllm --version", timeout=120)
    if version.strip():
        facts["vllm_version"] = version.strip().splitlines()[-1]

    _, args, _ = await srv._remote(
        droplet,
        f"docker inspect --format '{{{{json .Args}}}}' {srv.VLLM_CONTAINER}",
        timeout=60,
    )
    try:
        facts["serve_args"] = json.loads(args)
    except ValueError:
        pass

    # vLLM prints its whole HBM anatomy once at startup. Reading it back is how
    # the report carries numbers the engine computed instead of ones we derived.
    _, kv, _ = await srv._remote(
        droplet,
        f"docker logs {srv.VLLM_CONTAINER} 2>&1 | grep -E "
        "'GPU KV cache size|Available KV cache memory|Actual usage is|init engine' "
        "| grep -v queue_controller | tail -6",
        timeout=120,
    )
    if kv.strip():
        memory: dict = {"notes": kv.strip()}
        # The engine computed these; parsing them out beats restating the
        # arithmetic, and a line that stops matching leaves the field absent
        # rather than wrong.
        tokens = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", kv)
        if tokens:
            memory["resident_kv_tokens"] = int(tokens.group(1).replace(",", ""))
        gib = re.search(r"KV cache memory:\s*([\d.]+)\s*GiB", kv)
        if gib:
            memory["kv_cache_gib"] = float(gib.group(1))
        weights = re.search(r"Actual usage is ([\d.]+) GiB for consumed memory", kv)
        if weights:
            memory["weights_gib"] = float(weights.group(1))
        activation = re.search(r"([\d.]+) GiB for peak activation", kv)
        graphs = re.search(r"([\d.]+) GiB for CUDAGraph memory", kv)
        workspace = sum(float(m.group(1)) for m in (activation, graphs) if m)
        if workspace:
            memory["workspace_gib"] = round(workspace, 2)
        hbm = re.search(r"Free memory on device \([\d.]+/([\d.]+) GiB\)", kv)
        if hbm:
            memory["usable_hbm_gib"] = float(hbm.group(1))
        # Derived, and the derivation is the point: it is the engine's own pool
        # divided by the engine's own token count, which is what makes it
        # comparable to MODELS.md's geometry figure rather than a restatement
        # of it.
        if memory.get("kv_cache_gib") and memory.get("resident_kv_tokens"):
            per_token = memory["kv_cache_gib"] * (1024**3) / memory["resident_kv_tokens"]
            memory["kv_bytes_per_token"] = round(per_token, 1)
        facts["memory"] = memory

    startup: dict = {}
    init = re.search(r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s", kv)
    if init:
        startup["engine_init_s"] = float(init.group(1))
    compile_s = re.search(r"compilation: ([\d.]+) s", kv)
    if compile_s:
        startup["compile_s"] = float(compile_s.group(1))
    if startup:
        startup["notes"] = (
            "Engine init only, from the serving container's own log — not time from droplet "
            "power-on, which this run did not measure because the server was already up."
        )
        facts["startup"] = startup
    return facts


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--droplet", required=True, help="droplet name or id, as list_droplets shows it")
    parser.add_argument("--run-id", default=f"{date.today().isoformat()}-vllm-sweep-mi300x")
    parser.add_argument("--concurrency", type=int, nargs="+", default=CONCURRENCIES)
    parser.add_argument("--input-len", type=int, nargs="+", default=INPUT_LENS)
    parser.add_argument("--output-len", type=int, default=OUTPUT_LEN)
    parser.add_argument("--operator", default=os.environ.get("USER"))
    parser.add_argument(
        "--seed-base",
        type=int,
        default=0,
        help="offset for every prompt seed; change it to re-run a sweep against a cold prefix cache",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="runs per cell; the median-throughput run is reported and the spread recorded",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit without loading the card")
    args = parser.parse_args()

    srv = _load_server()
    max_model_len = int(srv.MAX_MODEL_LEN)
    cells = plan(args.concurrency, args.input_len, args.output_len, max_model_len)

    feasible = [c for c in cells if c["status"] == "pending"]
    print(
        f"{len(cells)} cells, {len(feasible)} runnable, {len(cells) - len(feasible)} infeasible at max_model_len {max_model_len}"
    )
    if args.dry_run:
        for cell in cells:
            print(f"  c{cell['concurrency']:<3} in{cell['input_len']:<6} out{cell['output_len']:<5} {cell['status']}")
        return 0

    run_dir = PROJECT_DIR / "benchmarks" / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    facts = await collect_facts(srv, args.droplet)
    sweep = await run_sweep(srv, args.droplet, cells, run_dir, repeat=args.repeat, seed_base=args.seed_base)
    report = build_report(srv, args.run_id, sweep, facts, operator=args.operator, repeat=args.repeat)

    reports_dir = PROJECT_DIR / "benchmarks" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / f"{args.run_id}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(PROJECT_DIR)}")
    print(f"      {run_dir.relative_to(PROJECT_DIR)}/ ({len(list(run_dir.glob('*.log')))} cell logs)")

    failed = [c for c in sweep if c.get("status") == "failed"]
    if failed:
        print(f"⚠️  {len(failed)} cell(s) failed — they are recorded in the report, not dropped.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
