#!/usr/bin/env python3
"""Two-dimensional serving sweep for Gemma 4 E2B on one NVIDIA T4 (Turing, SM 7.5).

PORTED, NOT COPIED. The eleven `benchmarking_suite.py` copies in this tree
(`tpu-vllm-*`, `gce-vllm-*`, `gke-vllm-*`) are byte-identical to each other and
WRONG FOR THIS RIG IN FOUR WAYS. Each one is fixed here, and each fix is the kind
that fails silently rather than loudly:

  1. THEY POST TO `/v1/completions`. The monorepo CLAUDE.md records that raw
     `/v1/completions` returns an EMPTY completion on `-it` checkpoints -- it is
     a documented, expected emptiness, which is exactly what makes it dangerous
     in a benchmark. The inherited suite then does

         tokens = data.get("usage", {}).get("completion_tokens", max_tokens)

     so a cell that generated NOTHING still books `max_tokens` worth of
     throughput whenever `usage` is absent. It reports a number either way.
     This harness uses `/v1/chat/completions`, the endpoint `server.py` itself
     uses, and counts only tokens it actually received.

  2. THEY MEASURE NON-STREAMING WALL TIME, so there is no TTFT and no TPOT --
     only a single end-to-end latency. The report schema has `ttft_ms`,
     `tpot_ms` and `itl_ms`, and on a decode-bound chip TPOT is the number that
     matters. This harness streams and timestamps every chunk.

  3. THEY SWEEP CONCURRENCY ONLY (1-D). Schema 1.1 added `input_len`/`output_len`
     per point precisely so a context x concurrency grid fits one report, and
     every recent run in the tree is 2-D. This harness sweeps both and writes
     `status` per cell -- `infeasible` for cells that cannot exist at this
     `max_model_len` rather than dropping them, because an absent cell is
     indistinguishable from an untried one.

  4. THEY IMPORT pandas. The repo code style says not to assume pandas; more to
     the point this runs ON the instance under SSM, where adding a pip step adds
     a way for the measurement to fail after the GPU is already burning money.
     Stdlib only -- urllib, csv, json, threads.

One more thing this does that the inherited suite does not: it CHECKS WHAT CAME
BACK. `verify_model_health` in `server.py` carries a degeneracy check because a
broken deploy on this lineage answered `': ok: ok: ok...'` -- non-empty, 16
tokens, completely wrong, and it would have benchmarked beautifully. A sweep that
only times bytes cannot tell that apart from working inference.

Prompts are sized with vLLM's own `/tokenize`, so `input_len` is the model's
token count and not a word-count guess.
"""

import argparse
import csv
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from itertools import pairwise

SCHEMA_VERSION = "1.1"


def _drop_none(d):
    """Omit null-valued keys. The schema types these as integers, so a null is
    a validation failure, not a way of saying 'not applicable'."""
    return {k: v for k, v in d.items() if v is not None}

# Filler that tokenizes densely and carries no instruction the model might obey.
_FILLER = (
    "The system records throughput and latency for each configuration under test. "
    "Measurements are repeated and aggregated across requests. "
)


def _post(url, payload, timeout=600, stream=False):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    if stream:
        return resp
    with resp:
        return json.loads(resp.read().decode())


def _stats(values):
    """Schema `latency_ms`: mean/median/p90/p99, milliseconds."""
    if not values:
        return None
    s = sorted(values)

    def pct(p):
        if len(s) == 1:
            return s[0]
        idx = min(len(s) - 1, max(0, round((p / 100.0) * (len(s) - 1))))
        return s[idx]

    return {
        "mean": round(statistics.mean(s), 3),
        "median": round(statistics.median(s), 3),
        "p90": round(pct(90), 3),
        "p99": round(pct(99), 3),
    }


class Sweep:
    def __init__(self, base_url, model, out_dir, output_tokens, requests_per_cell, max_model_len,
                 prompts_per_conc=0):
        self.base = base_url.rstrip("/")
        self.model = model
        self.out_dir = out_dir
        self.output_tokens = output_tokens
        self.requests_per_cell = requests_per_cell
        self.prompts_per_conc = prompts_per_conc
        self.max_model_len = max_model_len
        self.cells = []
        self.issues = []
        self._prompt_cache = {}
        self._lock = threading.Lock()

    # ---------- prompt construction ----------

    def _token_count(self, text):
        """Exact model token count via vLLM's /tokenize. None if unavailable."""
        try:
            d = _post(f"{self.base}/tokenize", {"model": self.model, "prompt": text}, timeout=60)
            if isinstance(d.get("count"), int):
                return d["count"]
            if isinstance(d.get("tokens"), list):
                return len(d["tokens"])
        except Exception:
            return None
        return None

    def build_prompt(self, target_tokens):
        """Grow filler until /tokenize reports >= target, then trim by words."""
        if target_tokens in self._prompt_cache:
            return self._prompt_cache[target_tokens]

        text = _FILLER
        n = self._token_count(text)
        if n is None:
            # Fallback: ~0.75 tokens/word is the usual English ratio. Recorded as
            # an issue so nobody reads input_len as exact.
            words = max(1, int(target_tokens / 0.75))
            reps = (words // len(_FILLER.split())) + 1
            text = " ".join((_FILLER * reps).split()[:words])
            with self._lock:
                if not any(i.get("id") == "tokenize-unavailable" for i in self.issues):
                    self.issues.append(
                        {
                            "id": "tokenize-unavailable",
                            "summary": "/tokenize did not answer; input_len is a word-count estimate, not a model token count.",
                            "severity": "medium",
                        }
                    )
            self._prompt_cache[target_tokens] = text
            return text

        while n < target_tokens:
            grow = max(2, int(len(text) * (target_tokens / max(n, 1)) * 1.1))
            text = (_FILLER * ((grow // len(_FILLER)) + 2))[:grow]
            n = self._token_count(text) or n
            if len(text) > 4_000_000:
                break

        words = text.split()
        lo, hi = 1, len(words)
        while lo < hi:
            mid = (lo + hi) // 2
            c = self._token_count(" ".join(words[:mid]))
            if c is None:
                break
            if c < target_tokens:
                lo = mid + 1
            else:
                hi = mid
        text = " ".join(words[:lo])
        self._prompt_cache[target_tokens] = text
        return text

    # ---------- one request ----------

    def one_request(self, prompt):
        """Stream one chat completion, timestamping every token."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.output_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
            # Pin the output length so every cell generates the same work.
            # Without this an -it model stops early and the cell measures
            # a shorter decode than the one it claims.
            "ignore_eos": True,
        }
        t0 = time.perf_counter()
        stamps = []
        text_parts = []
        usage = None
        try:
            resp = _post(f"{self.base}/v1/chat/completions", payload, stream=True)
            with resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for ch in chunk.get("choices", []):
                        piece = (ch.get("delta") or {}).get("content")
                        if piece:
                            stamps.append(time.perf_counter())
                            text_parts.append(piece)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

        if not stamps:
            return {"ok": False, "error": "no tokens streamed"}

        total = time.perf_counter() - t0
        ttft = (stamps[0] - t0) * 1000.0
        itls = [(b - a) * 1000.0 for a, b in pairwise(stamps)]
        n_out = (usage or {}).get("completion_tokens") or len(stamps)
        tpot = ((total * 1000.0) - ttft) / max(1, n_out - 1)
        return {
            "ok": True,
            "ttft_ms": ttft,
            "tpot_ms": tpot,
            "itls_ms": itls,
            "total_s": total,
            "out_tokens": n_out,
            "text": "".join(text_parts),
        }

    # ---------- degeneracy ----------

    @staticmethod
    def degenerate(text):
        """Catch the ': ok: ok: ok...' failure shape this lineage actually served.

        NOT a quality metric. It answers one question: did the engine emit real
        text, or a short token looping? A benchmark that skips this can post a
        clean throughput curve for an engine producing garbage.
        """
        words = text.split()
        if len(words) < 12:
            return None
        uniq = len(set(words)) / len(words)
        if uniq < 0.12:
            top = statistics.mode(words) if words else "?"
            return f"degenerate output: {uniq:.0%} unique words, dominated by {top!r}"
        return None

    # ---------- one cell ----------

    def run_cell(self, input_len, concurrency):
        key = f"ctx{input_len}-c{concurrency}"

        if input_len + self.output_tokens > self.max_model_len:
            print(f"  ⏭️  {key}: infeasible (ctx+out > max_model_len={self.max_model_len})")
            self.cells.append(
                {
                    "concurrency": concurrency,
                    "input_len": input_len,
                    "output_len": self.output_tokens,
                    "status": "infeasible",
                    "error": f"input_len+output_len={input_len + self.output_tokens} exceeds max_model_len={self.max_model_len}",
                }
            )
            return

        prompt = self.build_prompt(input_len)
        # Match the g5g baseline's `--num-prompts $((C*4))` when a multiplier is
        # set, so the A/B compares the same amount of work per cell rather than a
        # fixed request count that would under-load the high-concurrency cells.
        n = concurrency * self.prompts_per_conc if self.prompts_per_conc else max(concurrency, self.requests_per_cell)

        print(f"  🏎️  {key}: {n} requests at concurrency {concurrency} ...", flush=True)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(lambda _: self.one_request(prompt), range(n)))
        wall = time.perf_counter() - t0

        good = [r for r in results if r["ok"]]
        bad = [r for r in results if not r["ok"]]

        if not good:
            err = bad[0]["error"] if bad else "unknown"
            print(f"    ❌ {key}: all {n} requests failed -- {err}")
            self.cells.append(
                {
                    "concurrency": concurrency,
                    "input_len": input_len,
                    "output_len": self.output_tokens,
                    "status": "failed",
                    "error": err,
                }
            )
            return

        deg = self.degenerate(good[0]["text"])
        if deg:
            with self._lock:
                self.issues.append(
                    {"id": f"degenerate-{key}", "summary": deg, "severity": "high"}
                )
            print(f"    ⚠️  {key}: {deg}")

        out_tokens = sum(r["out_tokens"] for r in good)
        tpots = [r["tpot_ms"] for r in good]
        cell = {
            "concurrency": concurrency,
            "input_len": input_len,
            "output_len": self.output_tokens,
            "status": "ok",
            "request_rate_rps": round(len(good) / wall, 4),
            "output_tok_per_s": round(out_tokens / wall, 2),
            "total_tok_per_s": round((out_tokens + input_len * len(good)) / wall, 2),
            "per_stream_tok_per_s": round(1000.0 / statistics.median(tpots), 2),
            "ttft_ms": _stats([r["ttft_ms"] for r in good]),
            "tpot_ms": _stats(tpots),
            "itl_ms": _stats([x for r in good for x in r["itls_ms"]]),
            "raw": {
                "requests_attempted": n,
                "requests_ok": len(good),
                "requests_failed": len(bad),
                "wall_s": round(wall, 3),
                "output_tokens_total": out_tokens,
                "sample_completion": good[0]["text"][:400],
                "errors": sorted({r["error"] for r in bad})[:5],
            },
        }
        if bad:
            cell["error"] = f"{len(bad)}/{n} requests failed"
        self.cells.append(cell)
        print(
            f"    ✅ {key}: {cell['output_tok_per_s']:.1f} out tok/s | "
            f"per-stream {cell['per_stream_tok_per_s']:.1f} tok/s | "
            f"TTFT p50 {cell['ttft_ms']['median']:.0f} ms | TPOT p50 {cell['tpot_ms']['median']:.1f} ms",
            flush=True,
        )

    # ---------- output ----------

    def write(self, meta):
        os.makedirs(self.out_dir, exist_ok=True)

        csv_path = os.path.join(self.out_dir, "results.csv")
        cols = [
            "input_len", "concurrency", "output_len", "status",
            "output_tok_per_s", "per_stream_tok_per_s", "request_rate_rps",
            "ttft_ms_median", "tpot_ms_median", "itl_ms_median", "error",
        ]
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for c in sorted(self.cells, key=lambda x: (x["input_len"], x["concurrency"])):
                w.writerow(
                    {
                        "input_len": c["input_len"],
                        "concurrency": c["concurrency"],
                        "output_len": c["output_len"],
                        "status": c["status"],
                        "output_tok_per_s": c.get("output_tok_per_s", ""),
                        "per_stream_tok_per_s": c.get("per_stream_tok_per_s", ""),
                        "request_rate_rps": c.get("request_rate_rps", ""),
                        "ttft_ms_median": (c.get("ttft_ms") or {}).get("median", ""),
                        "tpot_ms_median": (c.get("tpot_ms") or {}).get("median", ""),
                        "itl_ms_median": (c.get("itl_ms") or {}).get("median", ""),
                        "error": c.get("error", ""),
                    }
                )

        ok = [c for c in self.cells if c["status"] == "ok"]
        peak = max((c["output_tok_per_s"] for c in ok), default=None)

        report = {
            "schema_version": SCHEMA_VERSION,
            "run": {
                "id": meta["run_id"],
                "date": datetime.now(UTC).strftime("%Y-%m-%d"),
                "operator": meta.get("operator", "gpu-vllm-g4dn-2b"),
                "source": meta.get("source", "benchmarking_suite.py (on-instance, SSM)"),
                "notes": meta.get("notes", ""),
            },
            "hardware": meta["hardware"],
            "model": meta["model"],
            "software": meta["software"],
            "throughput": {
                "workload": _drop_none({
                    "tool": "benchmarking_suite.py (stdlib, streaming chat/completions)",
                    "dataset": "synthetic filler sized with vLLM /tokenize",
                    "output_len": self.output_tokens,
                    # Omitted entirely when the count scales with concurrency --
                    # the schema types num_prompts as an integer, and a null here
                    # would fail validation rather than read as "varies".
                    "num_prompts": None if self.prompts_per_conc else self.requests_per_cell,
                    "runs_per_point": 1,
                }),
                "sweep": sorted(self.cells, key=lambda x: (x["input_len"], x["concurrency"])),
            },
            "issues": self.issues,
        }
        if peak is not None:
            report["notes"] = [f"Peak measured output throughput: {peak:.2f} tok/s."]

        json_path = os.path.join(self.out_dir, f"{meta['run_id']}.json")
        with open(json_path, "w") as fh:
            json.dump(report, fh, indent=2)
            fh.write("\n")

        print(f"\n📊 {csv_path}\n📄 {json_path}")
        return json_path


def main():
    p = argparse.ArgumentParser(description="Gemma 4 E2B serving sweep (vLLM, NVIDIA T4)")
    p.add_argument("--url", required=True, help="vLLM base URL, e.g. http://127.0.0.1:8000")
    p.add_argument("--model", default=os.getenv("MODEL_NAME", "google/gemma-4-E2B-it"))
    p.add_argument("--contexts", default="128,1024,4096", help="comma-separated input_len values")
    p.add_argument("--concurrencies", default="1,2,4,8", help="comma-separated concurrency values")
    p.add_argument("--output-tokens", type=int, default=128)
    p.add_argument("--requests-per-cell", type=int, default=8)
    p.add_argument("--prompts-per-concurrency", type=int, default=0,
                   help="if set, requests per cell = concurrency * this (matches vllm bench serve --num-prompts $((C*N)))")
    p.add_argument("--max-model-len", type=int, default=int(os.getenv("MAX_MODEL_LEN", "16384")))
    p.add_argument("--out-dir", default=".")
    p.add_argument("--run-id", default=None)
    p.add_argument("--meta", default=None, help="path to JSON with hardware/model/software blocks")
    args = p.parse_args()

    contexts = [int(x) for x in args.contexts.split(",") if x.strip()]
    concurrencies = [int(x) for x in args.concurrencies.split(",") if x.strip()]
    run_id = args.run_id or f"{datetime.now(UTC):%Y-%m-%d}-sweep-g4dn"

    meta = {
        "run_id": run_id,
        "hardware": {"accelerator": "nvidia-t4", "chips": 1},
        "model": {"id": args.model},
        "software": {"engine": "vllm"},
    }
    if args.meta:
        with open(args.meta) as fh:
            meta.update(json.load(fh))
        meta["run_id"] = run_id

    sweep = Sweep(
        args.url, args.model, args.out_dir,
        args.output_tokens, args.requests_per_cell, args.max_model_len,
        args.prompts_per_concurrency,
    )

    print(f"🚀 {run_id} -> {args.url}")
    print(f"   grid: {len(contexts)} contexts x {len(concurrencies)} concurrencies "
          f"= {len(contexts) * len(concurrencies)} cells, {args.output_tokens} output tokens each")

    print("🔥 warmup ...", flush=True)
    warm = sweep.one_request(sweep.build_prompt(min(contexts)))
    if not warm["ok"]:
        print(f"❌ warmup failed: {warm['error']}", file=sys.stderr)
        return 1
    print(f"   warmup ok: {warm['out_tokens']} tokens, TTFT {warm['ttft_ms']:.0f} ms")
    deg = sweep.degenerate(warm["text"])
    if deg:
        print(f"❌ warmup produced {deg}", file=sys.stderr)
        print(f"   sample: {warm['text'][:200]!r}", file=sys.stderr)
        sweep.issues.append({"id": "degenerate-warmup", "summary": deg, "severity": "high"})

    for ctx in contexts:
        for conc in concurrencies:
            sweep.run_cell(ctx, conc)

    sweep.write(meta)
    ok = sum(1 for c in sweep.cells if c["status"] == "ok")
    print(f"\n{ok}/{len(sweep.cells)} cells measured")
    return 0


if __name__ == "__main__":
    sys.exit(main())
