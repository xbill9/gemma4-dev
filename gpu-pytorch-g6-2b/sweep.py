#!/usr/bin/env python3
"""Context x output-length sweep against a running gpu-pytorch-g6-2b endpoint.

    python3 sweep.py --base http://<ip>:8000/v1 --out benchmarks/runs/<run>/

Lives at the rig root, NOT inside a run directory. The JAX sibling kept its
sweep inside `benchmarks/runs/<run>/` and copy-pasted it per run, so every
iteration re-derived its own harness and two numbers that were supposed to be
comparable had two sources of drift. One harness, many runs.

Three things it is opinionated about, each because it has cost a measurement
somewhere in this family:

* **Warm at the shape you measure.** The sibling recorded a 4x error from
  warming at one max_tokens and measuring at another. Cheap insurance even here,
  where nothing is compiled: cuBLAS autotune and allocator growth are per-shape.
* **Median, not mean.** A spot host gives the occasional slow request, and the
  cold/warm gap is several-fold.
* **Quote the decode gauge, not end-to-end.** `tpu_jax_decode_tokens_per_second`
  times decode alone; the end-to-end rate carries prefill and the HTTP round
  trip. Both are recorded, and they will not agree.
"""

import argparse
import json
import os
import statistics
import time
import urllib.error
import urllib.request

# Prompt lengths in TOKENS (approximate: built by repetition, then measured from
# the server's own usage.prompt_tokens, which is what the report records).
CONTEXTS = (64, 512, 1024, 2048)
OUTPUTS = (32, 128)
REPEATS = 3

# The filler is the actual prompt content, so it should describe THIS rig -- the
# G5g sibling's text named a Graviton2 host and a Turing GPU, neither of which is
# here. It is kept to a similar token density so context lengths stay comparable.
FILLER = (
    "The NVIDIA L4 is an Ada Lovelace generation GPU with compute capability 8.9. "
    "It is paired here with an x86_64 host on an EC2 G6 instance, and unlike the "
    "Turing parts it has a native bfloat16 datapath as well as fp8 support. "
)


def post(base: str, path: str, payload: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return json.loads(fh.read())


def get_text(url: str, timeout: int = 60) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as fh:
        return fh.read().decode()


def gauge(metrics_text: str, name: str) -> float:
    """Last value of a single-sample series, whatever its labels."""
    for line in metrics_text.splitlines():
        if line.startswith(name) and not line.startswith("#"):
            try:
                return float(line.rsplit(" ", 1)[1])
            except (IndexError, ValueError):
                return 0.0
    return 0.0


def prompt_for(approx_tokens: int) -> str:
    # ~1.3 tokens/word for this filler; overshoot then let the server report the
    # real count. The REPORTED prompt_tokens is what goes in the artifact.
    words = max(4, int(approx_tokens / 1.35))
    body = (FILLER * (words // len(FILLER.split()) + 2)).split()[:words]
    return " ".join(body) + "\n\nSummarize the text above."


def one(base: str, model: str, prompt: str, max_tokens: int) -> dict:
    t0 = time.perf_counter()
    body = post(base, "/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    })
    wall = time.perf_counter() - t0
    usage = body.get("usage", {})
    return {
        "wall_s": round(wall, 4),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "decode_tps": usage.get("decode_tokens_per_second", 0.0),
        "cold_shape": usage.get("cold_shape", None),
        "end_to_end_tps": round(usage.get("completion_tokens", 0) / wall, 3) if wall else 0.0,
        "text_head": (body.get("choices", [{}])[0].get("message", {}).get("content") or "")[:80],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="e.g. http://1.2.3.4:8000/v1")
    ap.add_argument("--model", default=os.getenv("MODEL_NAME", "google/gemma-4-E2B-it"))
    ap.add_argument("--out", required=True, help="run directory to write into")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    root = args.base.rstrip("/").removesuffix("/v1")
    health = json.loads(get_text(root + "/health"))
    print("health:", json.dumps(health), flush=True)

    cells, degen_before = [], gauge(get_text(root + "/metrics"),
                                   "tpu_jax_degenerate_responses_total")

    for ctx in CONTEXTS:
        prompt = prompt_for(ctx)
        for out_len in OUTPUTS:
            label = f"ctx~{ctx} out={out_len}"
            try:
                # Warm AT THIS SHAPE, then measure. The warm-up result is
                # recorded but never averaged in.
                warm = one(args.base, args.model, prompt, out_len)
                runs = [one(args.base, args.model, prompt, out_len)
                        for _ in range(args.repeats)]
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                detail = ""
                if isinstance(exc, urllib.error.HTTPError):
                    detail = exc.read().decode()[:300]
                print(f"{label}: FAILED {exc} {detail}", flush=True)
                cells.append({"context": ctx, "output_len": out_len,
                              "status": "infeasible", "error": f"{exc} {detail}"})
                continue

            decode = statistics.median(r["decode_tps"] for r in runs)
            e2e = statistics.median(r["end_to_end_tps"] for r in runs)
            cell = {
                "context": ctx,
                "input_len": runs[0]["prompt_tokens"],
                "output_len": out_len,
                "status": "ok",
                "completion_tokens": runs[0]["completion_tokens"],
                "decode_tps_median": round(decode, 3),
                "end_to_end_tps_median": round(e2e, 3),
                "wall_s_median": round(statistics.median(r["wall_s"] for r in runs), 3),
                "cold_first": warm["cold_shape"],
                "warmup_decode_tps": warm["decode_tps"],
                "runs": runs,
            }
            cells.append(cell)
            print(f"{label}: in={cell['input_len']} out={cell['completion_tokens']} "
                  f"decode={decode:.2f} tok/s  e2e={e2e:.2f} tok/s "
                  f"(warmup {warm['decode_tps']:.2f})", flush=True)

    metrics = get_text(root + "/metrics")
    degen_after = gauge(metrics, "tpu_jax_degenerate_responses_total")
    ok = [c for c in cells if c["status"] == "ok"]
    result = {
        "rig": health.get("rig"),
        "build_id": health.get("build_id"),
        "model": args.model,
        "health": health,
        "cells": cells,
        "summary": {
            "cells_ok": len(ok),
            "cells_total": len(cells),
            "decode_tps_min": round(min((c["decode_tps_median"] for c in ok), default=0), 3),
            "decode_tps_max": round(max((c["decode_tps_median"] for c in ok), default=0), 3),
            "decode_tps_median": round(
                statistics.median([c["decode_tps_median"] for c in ok]), 3) if ok else 0,
            "degenerate_responses": degen_after - degen_before,
        },
    }
    with open(os.path.join(args.out, "sweep.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    with open(os.path.join(args.out, "metrics.prom"), "w") as fh:
        fh.write(metrics)
    print("\n" + json.dumps(result["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
