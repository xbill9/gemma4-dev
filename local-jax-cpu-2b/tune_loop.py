#!/usr/bin/env python3
"""One iteration of the measure -> profile -> tune -> re-measure loop.

Run it from the RIG DIRECTORY. There is no instance to point it at:

    python3 tune_loop.py --label baseline
    # ... change the model port ...
    python3 tune_loop.py --label some-change \\
        --compare benchmarks/runs/2026-08-29-baseline-cpu

WHY THIS EXISTS. Every ingredient was already here and none of them composed:
`sweep.py` lived INSIDE a run directory and was copy-pasted per run, so each
iteration re-derived its own harness; `profile_decode.py` sat at the rig root and
had to be driven by hand; and the xprof extraction was prose in
docs/profiling-recipes.md. Three sources of drift between two numbers that are
supposed to be comparable.

The loop is deliberately opinionated about three things, each of which has cost
this rig a measurement before:

  * RESTART WHAT YOU MEASURE, and assert the build id the server reports equals
    the local payload digest. There is no deploy here, so the hazard changes
    shape rather than disappearing: an already-running process is serving the
    code as it was when it STARTED, and editing the model port does not disturb
    it. That is the same silent-stale-code failure that cost the G5g parent a
    full measure-and-conclude cycle on 2026-08-24, and it is easier to hit here,
    because there is no deploy step to remind you.
  * WARM AT THE SHAPE YOU MEASURE. max_new_tokens is a static_argnames entry, so
    (bucket, max_tokens) IS the compiled shape. Warming at a different max_tokens
    was previously a 4x error here (3.4 vs 13.5 tok/s).
  * MEDIAN, NOT MEAN, over repeats. The cold/warm gap here is larger than on any
    sibling -- XLA compiles the shape on the same CPU that then has to run it.

WHAT THE FORK DELETED. The cloud version shipped profile_decode.py to the
instance over SSM, ran it there, and brought the artifacts back through S3
because SSM truncates command output at 24,000 characters and an xprof kernel
table exceeds it. None of that transport exists here: the profiler runs as a
subprocess and writes into the run directory directly. That removes the whole
truncation hazard rather than working around it -- which is worth stating,
because "artifacts come back through S3" was load-bearing advice on the parent
and would read as a missing feature here.

The profiler wants the machine to itself, so the serving process is stopped for
that phase and restarted after. That is why the sweep runs FIRST -- and it bites
harder here than on a GPU rig, where "the device" and "the machine running the
harness" were two different things.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import glob
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server

FILLER = "token "


def post(base: str, payload: dict, timeout: float = 300.0):
    req = urllib.request.Request(
        f"{base}/chat/completions",   # `base` here is the /v1 API base
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.load(r)
    return body, time.time() - t


def cell(base: str, in_tok: int, out_tok: int, repeats: int = 3) -> dict:
    """Warm at the measured shape, then repeat. Returns the MEDIAN, not the mean.

    Median because a spot host occasionally gives one slow request and a
    three-sample mean cannot survive it; the cold/warm gap here is 4x, so an
    outlier that slips through would dominate.
    """
    prompt = (FILLER * max(1, in_tok)).strip()
    body = {"model": server.MODEL_NAME, "max_tokens": out_tok,
            "messages": [{"role": "user", "content": prompt}]}
    warm, _ = post(base, body)                     # compile this exact shape
    rates, walls = [], []
    for _ in range(repeats):
        got, wall = post(base, body)
        u = got.get("usage", {})
        comp = u.get("completion_tokens") or 0
        walls.append(wall)
        if wall > 0:
            rates.append(comp / wall)
    u = warm.get("usage", {})
    return {
        "input_tokens": u.get("prompt_tokens"),
        "output_tokens": out_tok,
        "pad_tokens": u.get("pad_tokens"),
        "bucket": u.get("bucket_size"),
        "end_to_end_tok_s": round(statistics.median(rates), 3) if rates else None,
        "wall_median_s": round(statistics.median(walls), 3),
        "samples": repeats,
    }


async def gauge() -> dict:
    """Decode gauge + weight bytes, straight off /metrics.

    tpu_jax_decode_tokens_per_second is the like-for-like figure both benchmark
    reports compare on: it times decode alone, where the end-to-end rate above
    also carries prefill and the HTTP round trip.
    """
    raw = await server.get_metrics()
    out = {}
    for line in raw.splitlines():
        for key in ("tpu_jax_decode_tokens_per_second", "tpu_jax_weight_bytes",
                    "tpu_jax_host_rss_bytes", "tpu_jax_prefill_milliseconds",
                    "tpu_jax_degenerate_responses_total", "tpu_jax_cold_requests_total"):
            if line.strip().startswith("|") and key in line:
                parts = [p.strip(" `|") for p in line.split("|") if p.strip(" `|")]
                if len(parts) >= 2:
                    try:
                        out[key] = float(parts[-1].replace(",", ""))
                    except ValueError:
                        pass
    return out


def profile_locally(outdir: str, steps: int, want_xprof: bool) -> None:
    """Run profile_decode.py here, with the serving process stopped.

    The cloud version of this was a shell script executed over SSM plus an S3
    round trip. Locally it is a subprocess, which removes two failure modes
    outright rather than mitigating them: SSM's silent 24,000-character output
    truncation (an xprof kernel table exceeds it, and a partial JSON is how you
    conclude a finding is not there), and the artifacts being stranded on an
    instance that gets reclaimed overnight.

    profile_decode.py is deliberately NOT part of the serving payload, so it is
    invoked from the rig root rather than from an installed copy.
    """
    root = os.path.dirname(os.path.abspath(__file__))
    logdir = os.path.join(outdir, "jaxtrace_raw")
    argv = [
        sys.executable, os.path.join(root, "profile_decode.py"),
        "--model", server.MODEL_NAME,
        "--ple-bits", str(server.PLE_BITS),
        "--max-model-len", str(server.MAX_MODEL_LEN),
        "--steps", str(steps), "--top", "25",
        "--logdir", logdir,
    ]
    if server.INT8_LM_HEAD:
        argv.append("--int8-lm-head")
    env = dict(os.environ, **server._serve_env())
    with open(f"{outdir}/profile_decode.txt", "w") as out, \
            open(f"{outdir}/profile_decode.err", "w") as err:
        subprocess.run(argv, stdout=out, stderr=err, env=env, cwd=root, check=False)

    # Keep the RAW trace, not only the derived rollups: the *.xplane.pb is what
    # a profile UI actually consumes, so keeping it means the run can be
    # reopened and re-analysed later. On the parent rig a trace was NOT kept and
    # the instance was reclaimed overnight, making it unrecoverable. Only the
    # xplane -- the per-op *.hlo_proto.pb files are hundreds of tiny artefacts
    # nothing consumes, and one run committed 241 files before this narrowed.
    keep = os.path.join(outdir, "jaxtrace")
    os.makedirs(keep, exist_ok=True)
    planes = sorted(
        glob.glob(os.path.join(logdir, "**", "*.xplane.pb"), recursive=True),
        key=os.path.getmtime,
    )
    if planes:
        shutil.copy2(planes[-1], os.path.join(keep, "profile.xplane.pb"))
    shutil.rmtree(logdir, ignore_errors=True)

    if want_xprof:
        extract_xprof(keep, outdir)


def extract_xprof(tracedir: str, outdir: str) -> None:
    """Turn the raw xplane into xprof's structured rollups, in this process.

    TWO TRAPS, both measured on the parent 2026-08-27 and both silent:

      * requirements-profiling.txt was never on the instance, so xprof
        "installed" with `Could not open requirements file` and the extraction
        died on ModuleNotFoundError. Here it is a plain local import, so the
        failure is an ImportError you can see -- reported, not swallowed.
      * `xspace_to_tool_data` returns BYTES, not str. Passing it through
        json.dumps raises "Object of type bytes is not JSON serializable", the
        handler catches it, and you get a 0-byte file plus a FAILED line: a
        profile that looks captured and is empty.

    WHETHER THE ROLLUPS SAY ANYTHING ON CPU IS UNVERIFIED. Every xprof finding
    in this rig's inherited prose -- the conversion share, the 0.0% TensorCore
    utilisation, the roofline -- came off a CUDA trace, and `kernel_stats` has
    no obvious analogue in an XLA:CPU xplane. profile_decode.py's own table does
    work here (VERIFIED 2026-08-29), so prefer it until this is checked.
    """
    planes = sorted(glob.glob(os.path.join(tracedir, "**", "*.xplane.pb"), recursive=True))
    log = open(f"{outdir}/xprof_extract.log", "w")
    print(f"xplane files: {planes}", file=log)
    if not planes:
        log.close()
        return
    try:
        from xprof.convert import raw_to_tool_data as R
    except ImportError as exc:
        print(f"xprof is not installed ({exc}); "
              f"pip install -r requirements-profiling.txt", file=log)
        log.close()
        return
    print(f"tools: {R.xspace_to_tool_names(planes)}", file=log)
    # No `^` suffix: current xprof maps the old `kernel_stats^` form and logs
    # "Received old tool format". docs/profiling-recipes.md still shows the old
    # names -- they work, but these are what the tool actually reports.
    for tool in ("kernel_stats", "memory_profile", "roofline_model"):
        try:
            data, _ = R.xspace_to_tool_data(planes, tool, {})
            with open(f"{outdir}/xprof_{tool}.json", "wb") as fh:
                if isinstance(data, bytes):
                    fh.write(data)
                elif isinstance(data, str):
                    fh.write(data.encode())
                else:
                    fh.write(json.dumps(data).encode())
            print(f"wrote {tool}", file=log)
        except Exception as exc:
            print(f"FAILED {tool} {type(exc).__name__} {exc}", file=log)
    log.close()


async def run(args) -> None:
    stamp = datetime.date.today().isoformat()
    # The `-cpu` suffix is the hardware slot, exactly as `-g5g` was on the
    # parent. It is also the least informative suffix in this monorepo, because
    # `cpu` names whatever machine the rig is checked out on rather than a SKU
    # the rig provisions -- so the host goes INTO the record below rather than
    # being carried by the directory name.
    outdir = args.outdir or f"benchmarks/runs/{stamp}-{args.label}-cpu"
    os.makedirs(outdir, exist_ok=True)
    rec: dict = {"label": args.label, "date": stamp, "host": host_facts()}

    # --- 1. run what we are about to measure --------------------------------
    # No deploy step, so the hazard is the opposite one: a process that is
    # ALREADY running is serving the code as it was when it started, and editing
    # the model port does not disturb it. Restarting is how you make the build
    # id below mean something.
    if not args.no_restart:
        os.system("make skill >/dev/null 2>&1")
        started = await server.start_jax_server(restart=True)
        if not started.startswith("✅"):
            raise SystemExit(f"start failed:\n{started}")
        rec["build_id"] = server._payload_digest()
        print(f"[start] build_id={rec['build_id']}", flush=True)

    # get_endpoint returns the OpenAI base, which ENDS IN /v1 -- but /health and
    # /metrics live at the ROOT. Polling {endpoint}/health therefore 404s forever
    # and reads as "never became ready" while the service is perfectly healthy.
    # Keep the two apart rather than reconstructing either by hand.
    raw = (await server.get_endpoint()).strip()
    api = next((w.strip("`") for w in raw.split() if w.strip("`").startswith("http")), raw)
    root = api[: -len("/v1")] if api.endswith("/v1") else api
    rec["endpoint"], rec["api_base"] = root, api

    # --- 2. wait for READY --------------------------------------------------
    # Longer than the parent's 600s. The load is the same ~10 GB read plus
    # host-side quantization, and there is no accelerator to hurry it; if the
    # weights are partly in swap it is slower again.
    t0 = time.time()
    while time.time() - t0 < args.ready_timeout:
        try:
            urllib.request.urlopen(f"{root}/health", timeout=10).read()
            break
        except Exception:
            time.sleep(5)
    else:
        raise SystemExit("never became ready")
    rec["time_to_ready_s"] = round(time.time() - t0, 1)
    print(f"[ready] {rec['time_to_ready_s']}s", flush=True)

    # --- 3. build-id assertion ---------------------------------------------
    health = json.load(urllib.request.urlopen(f"{root}/health", timeout=15))
    rec["served_build_id"] = health.get("build_id")
    if rec.get("build_id") and rec["served_build_id"] != rec["build_id"]:
        raise SystemExit(
            f"DIFFERENT PAYLOAD: the process is serving {rec['served_build_id']} "
            f"and this tree digests to {rec['build_id']}. It is running another "
            f"copy of the sources -- most likely the skill snapshot under "
            f".claude/skills/, which server.py resolves BEFORE the rig root.")

    # --- 4. sweep -----------------------------------------------------------
    cells = []
    for in_tok, out_tok in args.grid:
        c = cell(api, in_tok, out_tok, repeats=args.repeats)
        c["gauge_decode_tok_s"] = (await gauge()).get("tpu_jax_decode_tokens_per_second")
        cells.append(c)
        print(f"[sweep] in={c['input_tokens']} out={out_tok} "
              f"e2e={c['end_to_end_tok_s']} gauge={c['gauge_decode_tok_s']}", flush=True)
        json.dump(cells, open(f"{outdir}/sweep.json", "w"), indent=2)
    rec["cells"] = cells
    rec["metrics"] = await gauge()

    # --- 5. profile (wants the machine to itself) ---------------------------
    if not args.no_profile:
        print("[profile] stopping the serve and profiling locally", flush=True)
        await server.stop_jax_server()
        profile_locally(outdir, args.steps, args.xprof)
        if not args.no_restart:
            await server.start_jax_server()
        rec["kernel_table"] = summarize_kernels(f"{outdir}/profile_decode.txt")
        rec["xprof"] = summarize_xprof(f"{outdir}/xprof_kernel_stats.json")

    json.dump(rec, open(f"{outdir}/summary.json", "w"), indent=2)
    print(f"\n[done] {outdir}/summary.json")
    report(rec)
    if args.compare:
        compare(json.load(open(f"{args.compare}/summary.json")), rec)


def summarize_kernels(path: str) -> dict:
    """Pull the conversion / fp32-gemv shares out of profile_decode's table.

    These two numbers are the whole point of the loop: 87% of decode is dtype tax
    (54.4% conversion + 32.6% fp32 gemvx, measured 2026-08-25), so a tuning change
    that does not move THEM will not move throughput either.
    """
    if not os.path.exists(path):
        return {}
    out, total = {}, 0.0
    for line in open(path):
        low = line.lower()
        if "convert" in low and "%" in line:
            out["convert_pct"] = out.get("convert_pct", 0.0) + _pct(line)
        if "gemv" in low and "%" in line:
            out["gemv_pct"] = out.get("gemv_pct", 0.0) + _pct(line)
        total += _pct(line) if "%" in line else 0.0
    out["accounted_pct"] = round(total, 1)
    return {k: round(v, 2) for k, v in out.items()}


def summarize_xprof(path: str) -> dict:
    """Kernel-time shares and TensorCore use, from xprof's structured rollup.

    Preferred over the text table when available: xprof returns a
    {cols, rows} grid, so this reads named columns instead of scraping
    percentages out of formatted output.

    `is_kernel_using_tensor_core` is the column that settles the question --
    kernel NAMES like `gemvx::kernel<...float...>` already imply fp32, but the
    explicit flag is what the finding rests on (docs/profiling-recipes.md).
    """
    if not os.path.exists(path):
        return {}
    grid = json.load(open(path))
    cols = [c["id"] for c in grid.get("cols", [])]
    if not cols or "total_duration_us" not in cols:
        return {}
    # strict=False on purpose: xprof has added columns between versions, and a
    # row that is short or long should degrade to missing keys rather than
    # raising and losing the whole profile.
    rows = [dict(zip(cols, [c.get("v") for c in r["c"]], strict=False))
            for r in grid.get("rows", [])]
    total = sum(float(r.get("total_duration_us") or 0) for r in rows) or 1.0
    buckets: dict[str, float] = {}
    tensorcore = 0.0
    for r in rows:
        dur = float(r.get("total_duration_us") or 0)
        if r.get("is_kernel_using_tensor_core"):
            tensorcore += dur
        name = str(r.get("kernel_name", ""))
        key = ("convert" if "convert" in name else
               "gemv_fp32" if "gemv" in name else
               "fusion" if "fusion" in name else "other")
        buckets[key] = buckets.get(key, 0.0) + dur
    out = {f"{k}_pct": round(100 * v / total, 1) for k, v in buckets.items()}
    out["tensorcore_pct"] = round(100 * tensorcore / total, 1)
    out["total_kernel_ms"] = round(total / 1000, 1)
    out["kernels"] = len(rows)
    return out


def _pct(line: str) -> float:
    for tok in line.replace("|", " ").split():
        if tok.endswith("%"):
            try:
                return float(tok[:-1])
            except ValueError:
                return 0.0
    return 0.0


def report(rec: dict) -> None:
    print(f"\n=== {rec['label']} (build {rec.get('served_build_id')}) ===")
    for c in rec.get("cells", []):
        print(f"  in={c['input_tokens']:>5} out={c['output_tokens']:>4}  "
              f"gauge={c['gauge_decode_tok_s']}  e2e={c['end_to_end_tok_s']}")
    m = rec.get("metrics", {})
    # RSS, not HBM. A CPU JAX device exposes no allocator, so the process's
    # resident size is the only honest reading of what the serve is costing --
    # and on this rig it is also the number that decides whether it fits.
    print(f"  weights={m.get('tpu_jax_weight_bytes', 0)/1e9:.3f} GB  "
          f"rss={m.get('tpu_jax_host_rss_bytes', 0)/1e9:.3f} GB  "
          f"degenerate={m.get('tpu_jax_degenerate_responses_total')}")
    if rec.get("kernel_table"):
        print(f"  kernels: {rec['kernel_table']}")
    if rec.get("xprof"):
        x = rec["xprof"]
        print(f"  xprof:   convert={x.get('convert_pct')}% gemv_fp32={x.get('gemv_fp32_pct')}% "
              f"tensorcore={x.get('tensorcore_pct')}% over {x.get('total_kernel_ms')}ms "
              f"/ {x.get('kernels')} kernels")


def compare(old: dict, new: dict) -> None:
    print(f"\n=== {old['label']} -> {new['label']} ===")
    om = {(c["input_tokens"], c["output_tokens"]): c for c in old.get("cells", [])}
    for c in new.get("cells", []):
        k = (c["input_tokens"], c["output_tokens"])
        if k in om and om[k]["gauge_decode_tok_s"] and c["gauge_decode_tok_s"]:
            a, b = om[k]["gauge_decode_tok_s"], c["gauge_decode_tok_s"]
            print(f"  in={k[0]:>5} out={k[1]:>4}  {a:6.2f} -> {b:6.2f} tok/s  "
                  f"({(b/a - 1) * 100:+.1f}%)")
    ok, nk = old.get("kernel_table") or {}, new.get("kernel_table") or {}
    for key in ("convert_pct", "gemv_pct"):
        if key in ok and key in nk:
            print(f"  {key}: {ok[key]:.1f}% -> {nk[key]:.1f}% ({nk[key] - ok[key]:+.1f})")
    ox, nx = old.get("xprof") or {}, new.get("xprof") or {}
    for key in ("convert_pct", "gemv_fp32_pct", "tensorcore_pct"):
        if key in ox and key in nx:
            print(f"  xprof {key}: {ox[key]:.1f}% -> {nx[key]:.1f}% ({nx[key] - ox[key]:+.1f})")
    ow = (old.get("metrics") or {}).get("tpu_jax_weight_bytes")
    nw = (new.get("metrics") or {}).get("tpu_jax_weight_bytes")
    if ow and nw:
        print(f"  weights: {ow/1e9:.3f} -> {nw/1e9:.3f} GB ({(nw/ow - 1) * 100:+.1f}%)")


def host_facts() -> dict:
    """The host, into the record.

    Every sibling rig's hardware slot names a SKU it provisions, so the rig name
    carries the hardware and a report does not have to. `cpu` does not: it names
    whatever machine this checkout is on. Two runs of this rig on two machines
    are not comparable, and nothing else in the artifact would say so.
    """
    facts = server._host_facts()
    model = ""
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    model = line.partition(":")[2].strip()
                    break
    except OSError:
        pass
    return {
        "cpu": model,
        "cores": facts["cores"],
        "ram_total_bytes": facts["ram_total"],
        "swap_total_bytes": facts["swap_total"],
        "python": sys.version.split()[0],
    }


def grid(text: str):
    return [tuple(int(x) for x in pair.split("x")) for pair in text.split(",")]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--outdir")
    ap.add_argument("--compare", help="a previous run directory to diff against")
    ap.add_argument("--grid", type=grid, default=grid("32x32,512x32"),
                    help="in_tokensXout_tokens pairs, comma separated")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--steps", type=int, default=20, help="decode steps to profile")
    ap.add_argument("--xprof", action="store_true", help="also capture xprof rollups")
    ap.add_argument("--ready-timeout", type=int, default=2400,
                    help="seconds to wait for /health; the load is ~10 GB with "
                         "no accelerator, and slower again out of swap")
    ap.add_argument("--no-restart", action="store_true")
    ap.add_argument("--no-profile", action="store_true")
    asyncio.run(run(ap.parse_args()))
