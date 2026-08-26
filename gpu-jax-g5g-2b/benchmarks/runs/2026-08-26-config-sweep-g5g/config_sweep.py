#!/usr/bin/env python3
"""Full configuration sweep for gpu-jax-g5g-2b: the quantization levers.

The 2026-08-25 context sweep varied only the request shape. This varies the
SERVING CONFIG -- ple_bits and int8_lm_head -- which are the rig's actual
quantization knobs and which nothing here has ever measured.

What each lever is supposed to do, from ports/gemma4/jax_e_model.py:

  ple_bits=8   embed_tokens_per_layer 4.70 GB -> 2.35 GB, per-row scale
  ple_bits=4   4.70 GB -> 1.17 GB, needs group_size = hidden_size_per_layer_input
  int8_lm_head halves the largest single HBM read in a decode step, for ~0.8%
               logit error. The one lever that is NOT numerics-preserving.

The claim to test is memory, not speed: the PLE table is a gather, so decode
never streams it. Expect weight_bytes to fall and decode to move little.

Each config is a systemd ExecStart rewrite + restart + full model reload, so
this is slow. Results checkpoint after every config -- a spot reclamation keeps
everything already measured.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, "/home/xbill/gemma4-dev/gpu-jax-g5g-2b")
import server  # noqa: E402

APP = "/opt/jax-g5g/app"
UNIT = "/etc/systemd/system/jax-g5g.service"


def ssm(iid: str, cmd: str, timeout: int = 900) -> str:
    return asyncio.run(server._ssm(iid, cmd, timeout=timeout))


def set_config(iid: str, ple_bits: int, int8_lm_head: bool, max_model_len: int) -> str:
    """Rewrite the unit's ExecStart for this config and restart."""
    argv = (f"--model google/gemma-4-E2B-it --host 0.0.0.0 --port 8000 "
            f"--kv-cache-dtype auto --quant-mode fp16 "
            f"--max-model-len {max_model_len} --ple-bits {ple_bits}")
    if int8_lm_head:
        argv += " --int8-lm-head"
    exec_line = f"ExecStart=/usr/bin/python3.14 {APP}/jax_openai_server.py {argv}"
    cmd = (
        f"set -e; systemctl stop jax-g5g || true; "
        f"python3 - <<'PYEOF'\n"
        f"p = {UNIT!r}\n"
        f"lines = open(p).read().splitlines()\n"
        f"out = [({exec_line!r} if l.startswith('ExecStart=') else l) for l in lines]\n"
        f"open(p, 'w').write('\\n'.join(out) + '\\n')\n"
        f"PYEOF\n"
        f"grep '^ExecStart=' {UNIT}; "
        f"systemctl daemon-reload; systemctl start jax-g5g; "
        f"sleep 2; systemctl show jax-g5g -p ActiveState -p MainPID"
    )
    return ssm(iid, cmd, timeout=600)


def wait_ready(base: str, budget_s: int = 900) -> tuple[bool, dict]:
    deadline = time.time() + budget_s
    last = {}
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=20) as res:
                last = json.load(res)
                if last.get("status") == "ok":
                    return True, last
        except urllib.error.HTTPError as exc:
            try:
                last = json.load(exc)
            except Exception:
                last = {"http": exc.code}
        except Exception as exc:
            last = {"err": repr(exc)[:120]}
        time.sleep(15)
    return False, last


def metrics(base: str) -> dict:
    try:
        with urllib.request.urlopen(f"{base}/metrics", timeout=30) as res:
            text = res.read().decode()
    except Exception:
        return {}
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        series, _, raw = line.rpartition(" ")
        try:
            out[series.partition("{")[0]] = float(raw)
        except ValueError:
            pass
    return out


def run_sweep(base: str, tag: str, contexts: str, outputs: str,
              repeats: int, outdir: str) -> list:
    res = os.path.join(outdir, f"sweep_{tag}.json")
    jl = os.path.join(outdir, f"requests_{tag}.jsonl")
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "sweep.py"),
           "--base", base, "--out", res, "--jsonl", jl,
           "--contexts", contexts, "--outputs", outputs,
           "--repeats", str(repeats), "--timeout", "900"]
    log = os.path.join(outdir, f"console_{tag}.log")
    with open(log, "w") as fh:
        subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=False)
    try:
        return json.load(open(res))
    except Exception:
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--contexts", default="128,1024,4096")
    ap.add_argument("--outputs", default="64")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-model-len", type=int, default=8192)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    configs = [
        {"tag": "ple0",             "ple_bits": 0, "int8_lm_head": False},
        {"tag": "ple8",             "ple_bits": 8, "int8_lm_head": False},
        {"tag": "ple4",             "ple_bits": 4, "int8_lm_head": False},
        {"tag": "ple0_int8head",    "ple_bits": 0, "int8_lm_head": True},
        {"tag": "ple4_int8head",    "ple_bits": 4, "int8_lm_head": True},
    ]

    results = []
    out_json = os.path.join(a.outdir, "config_sweep.json")
    for cfg in configs:
        print(f"\n{'='*72}\n[config] {cfg['tag']}  ple_bits={cfg['ple_bits']} "
              f"int8_lm_head={cfg['int8_lm_head']}\n{'='*72}", flush=True)
        rec = dict(cfg)
        t0 = time.time()
        try:
            print(set_config(a.instance, cfg["ple_bits"], cfg["int8_lm_head"],
                             a.max_model_len)[:400], flush=True)
        except Exception as exc:
            rec.update(status="config_failed", error=repr(exc)[:300])
            results.append(rec); _save(out_json, results); continue

        ok, health = wait_ready(a.base)
        rec["load_s"] = round(time.time() - t0, 1)
        rec["health"] = health
        if not ok:
            rec["status"] = "never_ready"
            print(f"  NEVER READY after {rec['load_s']}s: {health}", flush=True)
            results.append(rec); _save(out_json, results); continue

        m0 = metrics(a.base)
        rec["weight_bytes"] = m0.get("tpu_jax_weight_bytes")
        rec["hbm_used_bytes"] = m0.get("tpu_jax_hbm_used_bytes")
        rec["hbm_limit_bytes"] = m0.get("tpu_jax_hbm_limit_bytes")
        print(f"  ready in {rec['load_s']}s  weights={rec['weight_bytes']}  "
              f"hbm={rec['hbm_used_bytes']}", flush=True)

        cells = run_sweep(a.base, cfg["tag"], a.contexts, a.outputs,
                          a.repeats, a.outdir)
        rec["cells"] = cells
        m1 = metrics(a.base)
        rec["hbm_used_after"] = m1.get("tpu_jax_hbm_used_bytes")
        rec["degenerate_total"] = m1.get("tpu_jax_degenerate_responses_total")
        rec["status"] = "ok" if cells and all(c.get("status") == "ok" for c in cells) else "partial"
        ok_n = sum(1 for c in cells if c.get("status") == "ok")
        print(f"  cells ok: {ok_n}/{len(cells)}   status={rec['status']}", flush=True)
        results.append(rec)
        _save(out_json, results)

    print(f"\nwrote {out_json}")
    return 0


def _save(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


if __name__ == "__main__":
    sys.exit(main())
