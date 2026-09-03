#!/usr/bin/env python3
"""Context x output-length sweep against a running OpenAI-compatible endpoint.

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
* **Quote decode, not end-to-end.** Decode times the token loop alone; the
  end-to-end rate carries prefill and the HTTP round trip. Both are recorded,
  and they will not agree -- by up to 41% on a long-context row.

## Why there are two decode sources

`usage.decode_tokens_per_second` is a field OUR servers invent. vLLM does not
emit it, and neither does anything else -- so until 2026-08-31 this harness
structurally could not measure the vLLM sibling, and the three-rig comparison
was three harnesses computing three statistics. Re-running a rig does not fix
that; only a common statistic does.

The common statistic is **client-side inter-token latency off the SSE stream**,
which every OpenAI-compatible server provides. `--decode-source` selects it:

* `usage`  -- the server's own decode gauge. Excludes HTTP entirely; the number
              every report in this family before 2026-08-31 quotes.
* `stream` -- (output_tokens - 1) / (t_last_token - t_first_token) from the
              stream. **Deliberately identical to `vllm bench serve`'s TPOT**,
              which is `(latency - ttft) / (output_len - 1)`, so a number from
              here is directly comparable to that tool's published figures.
* `both`   -- both, per repeat, at the cost of two requests instead of one.
              `usage` stays primary so older runs remain comparable, and the
              ratio between them is recorded as `stream_over_usage`.
* `auto`   -- (default) probe the endpoint once: `both` where the server emits
              the gauge, `stream` where it does not.

**`stream` reads slightly LOW against `usage` by construction** -- it carries
per-token SSE framing and a socket read that the server-side gauge does not.
That is the point of `both`: it measures the offset rather than assuming it.

**On THIS rig `auto` always resolves to `stream`, and that is the right answer
rather than a degraded one.** `llama-server` does not emit
`usage.decode_tokens_per_second` -- that field is an invention of our own JAX and
PyTorch servers -- so the probe falls through, and the harness measures the
client-side inter-token statistic that every rig can produce. That is exactly the
common statistic the 2026-08-31 rework exists to provide; do not "fix" this by
teaching the harness to scrape /metrics, which would reintroduce a per-rig
statistic under a shared name.

The server-side gauge still exists here and is still worth reading -- llama.cpp
exposes `llamacpp:tokens_predicted_total` / `llamacpp:tokens_predicted_seconds_total`
and the MCP `get_metrics` tool derives decode tok/s from them. Use it to sanity-
check a sweep, not as the sweep's own number: the two are different statistics
and the whole point is not to mix them.

One assumption the stream path makes, stated because it is not universal: **one
content delta per chunk**. True for this rig, the JAX sibling and vLLM's default
chat stream. A server that batches deltas would understate token count, so the
recorded `stream_chunks` is cross-checked against `usage.completion_tokens`
whenever the stream carries a usage block.
"""

import argparse
import itertools
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

FILLER = (
    "The Graviton2 processor is a 64-bit Arm server CPU designed by Annapurna Labs. "
    "It is paired here with an NVIDIA T4G, a Turing-generation GPU with compute "
    "capability 7.5, which has no bfloat16 datapath and no fp8. "
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


_NONCE = itertools.count(1)


def prompt_for(approx_tokens: int, unique: bool = True) -> str:
    # ~1.3 tokens/word for this filler; overshoot then let the server report the
    # real count. The REPORTED prompt_tokens is what goes in the artifact.
    words = max(4, int(approx_tokens / 1.35))
    body = (FILLER * (words // len(FILLER.split()) + 2)).split()[:words]
    # A DIFFERENT prompt per request by default, and the leading position matters:
    # a shared prefix is exactly what a prefix cache keys on, so a nonce appended
    # at the end would not defeat it.
    head = f"Record {next(_NONCE):06d}. " if unique else ""
    return head + " ".join(body) + "\n\nSummarize the text above."


def _chat_body(model: str, prompt: str, max_tokens: int, stream: bool) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if stream:
        body["stream"] = True
        # vLLM only emits a usage block on a stream when asked; our servers send
        # one regardless. Harmless where unsupported.
        body["stream_options"] = {"include_usage": True}
    return body


def one_usage(base: str, model: str, prompt: str, max_tokens: int) -> dict:
    """Non-streaming request; decode comes from the server's own gauge."""
    t0 = time.perf_counter()
    body = post(base, "/chat/completions", _chat_body(model, prompt, max_tokens, False))
    wall = time.perf_counter() - t0
    usage = body.get("usage", {})
    completion = usage.get("completion_tokens", 0)
    return {
        "source": "usage",
        "wall_s": round(wall, 4),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": completion,
        "decode_tps": usage.get("decode_tokens_per_second", 0.0),
        "cold_shape": usage.get("cold_shape", None),
        "end_to_end_tps": round(completion / wall, 3) if wall else 0.0,
        "text_head": (body.get("choices", [{}])[0].get("message", {}).get("content") or "")[:80],
    }


def one_stream(base: str, model: str, prompt: str, max_tokens: int,
               timeout: int = 900) -> dict:
    """Streaming request; decode comes from inter-token gaps, client-side.

    Matches `vllm bench serve`'s TPOT definition exactly:
    (latency - ttft) / (output_len - 1), i.e. the mean gap between the first and
    last token. The median gap is recorded alongside it because a single stalled
    chunk moves the mean and not the median, and a spot host does stall.
    """
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(_chat_body(model, prompt, max_tokens, True)).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    stamps: list[float] = []
    pieces: list[str] = []
    usage: dict = {}
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        for raw in fh:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []) or []:
                delta = (choice.get("delta") or {}).get("content")
                if delta:
                    # Stamp on arrival, before any parsing of later chunks.
                    stamps.append(time.perf_counter())
                    pieces.append(delta)
    wall = time.perf_counter() - t0

    n = len(stamps)
    ttft_s = (stamps[0] - t0) if n else 0.0
    if n >= 2:
        span = stamps[-1] - stamps[0]
        tpot_mean_s = span / (n - 1)
        gaps = [stamps[i + 1] - stamps[i] for i in range(n - 1)]
        tpot_median_s = statistics.median(gaps)
    else:
        tpot_mean_s = tpot_median_s = 0.0

    completion = usage.get("completion_tokens") or n
    return {
        "source": "stream",
        "wall_s": round(wall, 4),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": completion,
        "stream_chunks": n,
        # One delta per chunk is an assumption; this is how you find out it broke.
        "chunks_match_usage": (usage.get("completion_tokens") in (None, n)),
        "decode_tps": round(1.0 / tpot_mean_s, 4) if tpot_mean_s else 0.0,
        "decode_tps_median_gap": round(1.0 / tpot_median_s, 4) if tpot_median_s else 0.0,
        "ttft_ms": round(ttft_s * 1000, 3),
        "tpot_ms": round(tpot_mean_s * 1000, 4),
        "cold_shape": usage.get("cold_shape", None),
        "end_to_end_tps": round(completion / wall, 3) if wall else 0.0,
        "text_head": "".join(pieces)[:80],
    }


def probe_source(base: str, model: str) -> str:
    """Does this server emit its own decode gauge? Decides `auto`."""
    try:
        body = post(base, "/chat/completions", _chat_body(model, "ok", 4, False), timeout=180)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return "stream"
    if (body.get("usage") or {}).get("decode_tokens_per_second") is not None:
        return "both"
    return "stream"


def measure(base: str, model: str, prompt: str, max_tokens: int, source: str) -> dict:
    """One repeat. Returns a run dict whose `decode_tps` is the primary figure."""
    if source == "usage":
        return one_usage(base, model, prompt, max_tokens)
    if source == "stream":
        return one_stream(base, model, prompt, max_tokens)
    # both: usage stays primary so pre-2026-08-31 runs remain comparable.
    u = one_usage(base, model, prompt, max_tokens)
    s = one_stream(base, model, prompt, max_tokens)
    merged = dict(u)
    merged["source"] = "both"
    merged["decode_tps_usage"] = u["decode_tps"]
    merged["decode_tps_stream"] = s["decode_tps"]
    merged["ttft_ms"] = s["ttft_ms"]
    merged["tpot_ms"] = s["tpot_ms"]
    merged["stream_chunks"] = s["stream_chunks"]
    merged["chunks_match_usage"] = s["chunks_match_usage"]
    merged["stream_over_usage"] = (
        round(s["decode_tps"] / u["decode_tps"], 4) if u["decode_tps"] else None
    )
    return merged


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="e.g. http://1.2.3.4:8000/v1")
    ap.add_argument("--model", default=os.getenv("MODEL_NAME", "google/gemma-4-E2B-it"))
    ap.add_argument("--out", required=True, help="run directory to write into")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    # Configurable because the first run swept only to 2,501 tokens against a
    # --seq of 4096 and still reported "8/8 cells" -- coverage overstated by not
    # going near the configured bound. Name the contexts you mean.
    ap.add_argument("--contexts", default=",".join(str(c) for c in CONTEXTS))
    ap.add_argument("--outputs", default=",".join(str(o) for o in OUTPUTS))
    ap.add_argument("--decode-source", choices=("auto", "usage", "stream", "both"),
                    default="auto",
                    help="where the decode figure comes from; see the module docstring")
    ap.add_argument("--rig", default=None,
                    help="rig name for the artifact when /health does not report one")
    # MEASURED 2026-08-31, and it invalidated a TTFT comparison before anyone
    # published it. The sweep sent ONE prompt per cell for the warm-up and all
    # repeats. vLLM ships `enable_prefix_caching=True`, so it answered from cache:
    # 97,440 hits of 102,898 queries, a 94.7% hit rate, and it genuinely prefilled
    # only 5.3% of the tokens it was sent. Its TTFT read 0.025 ms/token against
    # JAX's 1.403 -- a 56x "advantage" that is mostly the harness, since neither
    # sibling has a prefix cache to hit. `fixed` reproduces that older behaviour.
    ap.add_argument("--prompt-mode", choices=("unique", "fixed"), default="unique",
                    help="unique (default) puts a per-request nonce FIRST so a prefix "
                         "cache cannot hit; fixed reproduces pre-2026-08-31 runs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    root = args.base.rstrip("/").removesuffix("/v1")

    # vLLM's /health is an empty 200, not JSON. Tolerate it rather than making
    # the harness rig-specific again.
    try:
        health = json.loads(get_text(root + "/health"))
    except (json.JSONDecodeError, urllib.error.HTTPError, urllib.error.URLError,
            TimeoutError):
        health = {}
    print("health:", json.dumps(health), flush=True)

    source = probe_source(args.base, args.model) if args.decode_source == "auto" \
        else args.decode_source
    print(f"decode-source: {args.decode_source} -> {source}", flush=True)

    try:
        metrics_before = get_text(root + "/metrics")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        metrics_before = ""
    cells = []
    degen_before = gauge(metrics_before, "tpu_jax_degenerate_responses_total")

    unique = args.prompt_mode == "unique"
    for ctx in [int(c) for c in args.contexts.split(",")]:
        for out_len in [int(o) for o in args.outputs.split(",")]:
            prompt = prompt_for(ctx, unique)
            label = f"ctx~{ctx} out={out_len}"
            try:
                # Warm AT THIS SHAPE, then measure. The warm-up result is
                # recorded but never averaged in.
                warm = measure(args.base, args.model,
                               prompt_for(ctx, unique) if unique else prompt,
                               out_len, source)
                runs = [measure(args.base, args.model,
                                prompt_for(ctx, unique) if unique else prompt,
                                out_len, source)
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
                "decode_source": source,
        "prompt_mode": args.prompt_mode,
                "completion_tokens": runs[0]["completion_tokens"],
                "decode_tps_median": round(decode, 3),
                "end_to_end_tps_median": round(e2e, 3),
                "wall_s_median": round(statistics.median(r["wall_s"] for r in runs), 3),
                "cold_first": warm.get("cold_shape"),
                "warmup_decode_tps": warm["decode_tps"],
                "runs": runs,
            }
            for key in ("decode_tps_usage", "decode_tps_stream", "stream_over_usage",
                        "ttft_ms", "tpot_ms"):
                vals = [r[key] for r in runs if r.get(key) is not None]
                if vals:
                    cell[f"{key}_median"] = round(statistics.median(vals), 4)
            cells.append(cell)
            extra = ""
            if "stream_over_usage_median" in cell:
                extra = f"  stream/usage={cell['stream_over_usage_median']:.4f}"
            elif "tpot_ms_median" in cell:
                extra = f"  tpot={cell['tpot_ms_median']:.2f} ms"
            print(f"{label}: in={cell['input_len']} out={cell['completion_tokens']} "
                  f"decode={decode:.2f} tok/s  e2e={e2e:.2f} tok/s "
                  f"(warmup {warm['decode_tps']:.2f}){extra}", flush=True)

    try:
        metrics = get_text(root + "/metrics")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        metrics = ""
    degen_after = gauge(metrics, "tpu_jax_degenerate_responses_total")
    ok = [c for c in cells if c["status"] == "ok"]
    summary = {
        "cells_ok": len(ok),
        "cells_total": len(cells),
        "decode_source": source,
        "decode_tps_min": round(min((c["decode_tps_median"] for c in ok), default=0), 3),
        "decode_tps_max": round(max((c["decode_tps_median"] for c in ok), default=0), 3),
        "decode_tps_median": round(
            statistics.median([c["decode_tps_median"] for c in ok]), 3) if ok else 0,
        "degenerate_responses": degen_after - degen_before,
    }
    ratios = [c["stream_over_usage_median"] for c in ok if "stream_over_usage_median" in c]
    if ratios:
        # The calibration this whole flag exists for: how much lower the portable
        # client-side statistic reads than the server's own gauge, on a rig that
        # emits both. Apply it before comparing a `stream` number to a `usage` one.
        summary["stream_over_usage_median"] = round(statistics.median(ratios), 4)
    result = {
        "rig": health.get("rig") or args.rig,
        "build_id": health.get("build_id"),
        "model": args.model,
        "health": health,
        "cells": cells,
        "summary": summary,
    }
    with open(os.path.join(args.out, "sweep.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    with open(os.path.join(args.out, "metrics.prom"), "w") as fh:
        fh.write(metrics)
    print("\n" + json.dumps(result["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
