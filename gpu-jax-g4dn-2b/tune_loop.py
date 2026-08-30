#!/usr/bin/env python3
"""One iteration of the measure -> profile -> tune -> re-measure loop.

Run it from the RIG DIRECTORY, against a live instance:

    python3 tune_loop.py --instance i-0123 --label baseline
    # ... change the model port ...
    python3 tune_loop.py --instance i-0123 --label bf16-bitshift \\
        --compare benchmarks/runs/2026-08-29-first-serve-g4dn

WHY THIS EXISTS. Every ingredient was already here and none of them composed:
`sweep.py` lived INSIDE a run directory and was copy-pasted per run, so each
iteration re-derived its own harness; `profile_decode.py` sat at the rig root and
had to be driven by hand; and the xprof extraction was prose in
docs/profiling-recipes.md. Three sources of drift between two numbers that are
supposed to be comparable.

The loop is deliberately opinionated about three things, each of which has cost
this rig a measurement before:

  * DEPLOY WHAT YOU MEASURE. `make skill` then deploy, and assert the build id
    the server reports equals the local payload digest. On 2026-08-24 a deploy
    shipped the previous skill snapshot and a full measure-and-conclude cycle was
    spent on stale code.
  * WARM AT THE SHAPE YOU MEASURE. max_new_tokens is a static_argnames entry, so
    (bucket, max_tokens) IS the compiled shape. Warming at a different max_tokens
    was previously a 4x error here (3.4 vs 13.5 tok/s).
  * ARTIFACTS COME BACK THROUGH S3, NOT SSM. SSM caps command output at 24,000
    characters and truncates silently; an xprof kernel table exceeds that, and a
    partial JSON is how you conclude a finding is not there.

The profiler needs the GPU to itself, so the service is stopped for that phase
and restarted after. That is why the sweep runs FIRST.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server

S3_RESULTS = "s3://vllm-models-bucket/benchmarks/gpu-jax-g4dn-2b"

# Slot 3 of the rig directory name (<platform>-<runtime>-<hardware>-<model>),
# which is what benchmarks/README.md calls <hw-short>. Derived, never a literal:
# a literal is what silently survives a fork, and here it would mislabel the
# hardware a measurement came off.
RIG_NAME = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
HW_SHORT = RIG_NAME.split("-")[2]
REMOTE_OUT = "/opt/jax-g4dn/loopout"
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


async def gauge(iid: str) -> dict:
    """Decode gauge + weight bytes, straight off /metrics.

    tpu_jax_decode_tokens_per_second is the like-for-like figure both benchmark
    reports compare on: it times decode alone, where the end-to-end rate above
    also carries prefill and the HTTP round trip.
    """
    raw = await server.get_metrics(iid)
    out = {}
    for line in raw.splitlines():
        for key in ("tpu_jax_decode_tokens_per_second", "tpu_jax_weight_bytes",
                    "tpu_jax_hbm_used_bytes", "tpu_jax_prefill_milliseconds",
                    "tpu_jax_degenerate_responses_total", "tpu_jax_cold_requests_total"):
            if line.strip().startswith("|") and key in line:
                parts = [p.strip(" `|") for p in line.split("|") if p.strip(" `|")]
                if len(parts) >= 2:
                    try:
                        out[key] = float(parts[-1].replace(",", ""))
                    except ValueError:
                        pass
    return out


PROFILE_SH = r"""
set -e
mkdir -p {out}
cd /opt/jax-g4dn/app
set -a; . /opt/jax-g4dn/env; set +a
systemctl stop jax-g4dn
# profile_decode.py is NOT part of the deploy payload -- it is a profiling tool,
# not a serving one -- so it is shipped by this driver alongside the run.
PYTHONPATH=/opt/jax-g4dn/app python3.14 {out}/profile_decode.py \
    --model "$MODEL_NAME" --ple-bits "${{PLE_BITS:-4}}" {int8flag} \
    --steps {steps} --top 25 > {out}/profile_decode.txt 2>{out}/profile_decode.err || true
{xprof}
systemctl start jax-g4dn
# The RAW trace comes back too, not just the derived rollups. jax.profiler.trace
# writes /tmp/jaxtrace/plugins/profile/<run>/*.xplane.pb, and that file is what
# any profile UI actually consumes -- so keeping it means the run can be opened
# and re-analysed later, on a laptop, long after the instance is gone. Yesterday's
# trace was NOT kept and the instance was reclaimed overnight, so it is
# unrecoverable; that is the whole reason this line exists. A decode-only trace is
# a few MB, which is noise next to the payload.
mkdir -p {out}/jaxtrace
# Only the xplane (and the chrome trace) -- that is what a profile UI reads. The
# per-op *.hlo_proto.pb files are hundreds of tiny artefacts nothing consumes,
# and copying the directory wholesale also drags in every EARLIER run still
# sitting in /tmp/jaxtrace. One run committed 241 files before this narrowed.
NEWEST=$(ls -1dt /tmp/jaxtrace/plugins/profile/*/ 2>/dev/null | head -1)
cp "$NEWEST"/*.xplane.pb {out}/jaxtrace/profile.xplane.pb 2>/dev/null || true
tar czf {out}.tgz -C {out} . 2>/dev/null || true
aws s3 cp {out}.tgz {s3}/{label}.tgz --only-show-errors
echo "UPLOADED {s3}/{label}.tgz"
"""

# Substituted with str.replace on a SENTINEL, not str.format. This block embeds
# Python that contains `{}` (an empty dict literal) and f-strings of its own, and
# str.format reads every one of those as a placeholder -- the same brace hazard
# the monorepo CLAUDE.md documents for startup_script_template.sh. A sentinel
# cannot collide with embedded code.
XPROF_SH = r"""
python3.14 -m pip install --break-system-packages -q -r __OUT__/requirements-profiling.txt \
    >__OUT__/xprof_install.log 2>&1 || echo "xprof install FAILED" >>__OUT__/xprof_install.log
PYTHONPATH=/opt/jax-g4dn/app python3.14 - <<'XP' >__OUT__/xprof_extract.log 2>&1 || true
import glob, json
from xprof.convert import raw_to_tool_data as R
xs = sorted(glob.glob("/tmp/jaxtrace/**/*.xplane.pb", recursive=True))
print("xplane files:", xs)
if xs:
    print("tools:", R.xspace_to_tool_names(xs))
    # No `^` suffix: current xprof maps the old `kernel_stats^` form and logs
    # "Received old tool format". docs/profiling-recipes.md still shows the old
    # names -- they work, but these are what the tool actually reports.
    for tool in ("kernel_stats", "memory_profile", "roofline_model"):
        try:
            data, _ = R.xspace_to_tool_data(xs, tool, {})
            # THIS RETURNS bytes, not str. Writing it through json.dumps raises
            # "Object of type bytes is not JSON serializable", the handler catches
            # it, and you are left with a 0-byte file and a FAILED line in a log
            # nobody reads -- measured 2026-08-27.
            with open("__OUT__/xprof_" + tool + ".json", "wb") as fh:
                if isinstance(data, bytes):
                    fh.write(data)
                elif isinstance(data, str):
                    fh.write(data.encode())
                else:
                    fh.write(json.dumps(data).encode())
            print("wrote", tool)
        except Exception as e:
            print("FAILED", tool, type(e).__name__, e)
XP
"""


async def run(args) -> None:
    iid = args.instance
    stamp = datetime.date.today().isoformat()
    # The <hw-short> suffix must equal the rig's HARDWARE SLOT, not a literal
    # carried over from whichever rig this harness was forked from. It was
    # hardcoded "g5g" here and produced `2026-08-29-first-serve-g5g` inside
    # gpu-jax-g4dn-2b -- a directory name asserting the run happened on hardware
    # it did not, which is exactly how benchmark JSON came to travel with the
    # forks in the first place. Derive it from the rig directory instead.
    outdir = args.outdir or f"benchmarks/runs/{stamp}-{args.label}-{HW_SHORT}"
    os.makedirs(outdir, exist_ok=True)
    rec: dict = {"label": args.label, "instance": iid, "date": stamp}

    # --- 1. deploy what we are about to measure -----------------------------
    if not args.no_deploy:
        os.system("make skill >/dev/null 2>&1")
        dep = await server.deploy_jax_server(iid)
        if not dep.startswith("✅"):
            raise SystemExit(f"deploy failed:\n{dep}")
        rec["build_id"] = next(
            (ln.split("`")[1] for ln in dep.splitlines() if ln.startswith("Build id")), None)
        print(f"[deploy] build_id={rec['build_id']}", flush=True)

    # get_endpoint returns the OpenAI base, which ENDS IN /v1 -- but /health and
    # /metrics live at the ROOT. Polling {endpoint}/health therefore 404s forever
    # and reads as "never became ready" while the service is perfectly healthy.
    # Keep the two apart rather than reconstructing either by hand.
    raw = (await server.get_endpoint(iid)).strip()
    api = next((w.strip("`") for w in raw.split() if w.strip("`").startswith("http")), raw)
    root = api[: -len("/v1")] if api.endswith("/v1") else api
    rec["endpoint"], rec["api_base"] = root, api

    # --- 2. wait for READY --------------------------------------------------
    t0 = time.time()
    while time.time() - t0 < 600:
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
            f"STALE DEPLOY: served {rec['served_build_id']} != local {rec['build_id']}")

    # --- 4. sweep -----------------------------------------------------------
    cells = []
    for in_tok, out_tok in args.grid:
        c = cell(api, in_tok, out_tok, repeats=args.repeats)
        c["gauge_decode_tok_s"] = (await gauge(iid)).get("tpu_jax_decode_tokens_per_second")
        cells.append(c)
        print(f"[sweep] in={c['input_tokens']} out={out_tok} "
              f"e2e={c['end_to_end_tok_s']} gauge={c['gauge_decode_tok_s']}", flush=True)
        json.dump(cells, open(f"{outdir}/sweep.json", "w"), indent=2)
    rec["cells"] = cells
    rec["metrics"] = await gauge(iid)

    # --- 5. profile (needs the GPU to itself) -------------------------------
    if not args.no_profile:
        await server._ssm(iid, f"mkdir -p {REMOTE_OUT}")
        # requirements-profiling.txt ships too. It is NOT in the deploy payload
        # (a serving image should not carry a profiler) and nothing else puts it
        # on the box -- so docs/profiling-recipes.md's
        # `pip install -r /opt/jax-g4dn/requirements-profiling.txt` referenced a
        # path that never existed, and xprof silently failed to install.
        for f in ("profile_decode.py", "requirements-profiling.txt"):
            import base64
            blob = base64.b64encode(open(f, "rb").read()).decode()
            await server._ssm(iid, f"echo '{blob}' | base64 -d > {REMOTE_OUT}/{f}")
        xprof = XPROF_SH.replace("__OUT__", REMOTE_OUT) if args.xprof else ""
        sh = PROFILE_SH.format(out=REMOTE_OUT, s3=S3_RESULTS, label=args.label,
                               steps=args.steps, xprof=xprof,
                               int8flag="--int8-lm-head" if server.INT8_LM_HEAD else "")
        print("[profile] running on the instance (service stopped)", flush=True)
        print((await server._ssm(iid, sh, timeout=1800)).splitlines()[-1])
        os.system(f"aws s3 cp {S3_RESULTS}/{args.label}.tgz {outdir}/artifacts.tgz "
                  f"--only-show-errors && tar xzf {outdir}/artifacts.tgz -C {outdir} "
                  f"&& rm -f {outdir}/artifacts.tgz")
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
    print(f"  weights={m.get('tpu_jax_weight_bytes', 0)/1e9:.3f} GB  "
          f"hbm={m.get('tpu_jax_hbm_used_bytes', 0)/1e9:.3f} GB  "
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


def grid(text: str):
    return [tuple(int(x) for x in pair.split("x")) for pair in text.split(",")]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--outdir")
    ap.add_argument("--compare", help="a previous run directory to diff against")
    ap.add_argument("--grid", type=grid, default=grid("32x64,512x64,2048x64"),
                    help="in_tokensXout_tokens pairs, comma separated")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--steps", type=int, default=20, help="decode steps to profile")
    ap.add_argument("--xprof", action="store_true", help="also capture xprof rollups")
    ap.add_argument("--no-deploy", action="store_true")
    ap.add_argument("--no-profile", action="store_true")
    asyncio.run(run(ap.parse_args()))
