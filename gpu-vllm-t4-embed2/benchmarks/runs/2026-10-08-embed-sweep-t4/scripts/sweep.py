"""Embedding sweep: input_len x inputs-per-request x concurrency, 3 repeats.

Inputs are random token-id arrays, so lengths are exact and no two requests share
a token sequence (seed is unique per cell, rep and request). One warmup request of
the cell's shape precedes each rep. Writes one JSON line per (cell, rep) to stdout.
"""
import json, math, os, random, statistics, subprocess, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

U = os.environ.get("URL", "http://127.0.0.1:8000/v1/embeddings")
MODEL = "google/embeddinggemma-2"
LENS = [int(x) for x in os.environ.get("LENS", "128,512,2048,8192").split(",")]
BATCHES = [int(x) for x in os.environ.get("BATCHES", "1,4,16").split(",")]
CONCS = [int(x) for x in os.environ.get("CONCS", "1,4").split(",")]
REPS = int(os.environ.get("REPS", "3"))
TOKEN_BUDGET = int(os.environ.get("TOKEN_BUDGET", "65536"))  # tokens per rep, before the floor
MIN_REQS = int(os.environ.get("MIN_REQS", "8"))
LO, HI = 1000, 200000  # clear of special tokens, inside the 262k vocab


def ids(seed, n):
    r = random.Random(seed)
    return [r.randrange(LO, HI) for _ in range(n)]


def post(body):
    req = urllib.request.Request(U, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=900))
    return time.time() - t, r["usage"]["prompt_tokens"], len(r["data"])


def vram():
    try:
        return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])
    except Exception:
        return None


def main():
    for L in LENS:
        for B in BATCHES:
            for C in CONCS:
                n_req = max(MIN_REQS, C, math.ceil(TOKEN_BUDGET / (L * B)))
                for rep in range(REPS):
                    base = (L * 1000003 + B * 10007 + C * 101 + rep) * 100000
                    row = dict(input_len=L, batch=B, concurrency=C, rep=rep + 1, requests=n_req)
                    try:
                        post({"model": MODEL, "input": [ids(base - 1 - k, L) for k in range(B)]})
                        bodies = [{"model": MODEL, "input": [ids(base + i * B + k, L) for k in range(B)]} for i in range(n_req)]
                        t0 = time.time()
                        with ThreadPoolExecutor(C) as ex:
                            out = list(ex.map(post, bodies))
                        wall = time.time() - t0
                        lat = sorted(o[0] for o in out)
                        toks = sum(o[1] for o in out)
                        assert all(o[2] == B for o in out), "embedding count mismatch"
                        assert toks == n_req * B * L, f"token count {toks} != {n_req * B * L}"
                        row.update(status="ok", wall_s=round(wall, 3), input_tok_per_s=round(toks / wall),
                                   inputs_per_s=round(n_req * B / wall, 2), req_per_s=round(n_req / wall, 3),
                                   p50_ms=round(1000 * statistics.median(lat)),
                                   p99_ms=round(1000 * lat[int(0.99 * (len(lat) - 1))]), vram_mib=vram())
                    except urllib.error.HTTPError as e:
                        row.update(status="failed", error=f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}")
                    except Exception as e:
                        row.update(status="failed", error=repr(e)[:300])
                    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
