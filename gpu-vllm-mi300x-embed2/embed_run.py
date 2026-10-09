"""EmbeddingGemma 2 on one AMD MI300X, alone and beside a Gemma 4 generator on the same card.

    python3 embed_run.py --ip <droplet ip> --run-id <date>-embed2-mi300x [--phases check,embed,gen,both]

Phases, each recorded under benchmarks/runs/<run-id>/:

  check  serve google/embeddinggemma-2 and compare its vectors with reference/reference.json
         (sentence-transformers on CPU): cosine per text at 768 dims and at 512/256/128 via the
         `dimensions` field, and whether each query retrieves the same document.
  embed  `vllm bench serve --backend openai-embeddings` at 128/1,024/4,096 input tokens by 1/16/128
         concurrent requests, the generator server loaded beside it but idle.
  gen    the generator (12B fp8, the sweep's pick) alone at 8 and 64 requests, 1,024-token prompts,
         512 output tokens, the embedding server loaded beside it but idle.
  both   each measured again while the other runs a steady load: the generator under 128-way
         embedding traffic at 1,024 tokens, and the embedder under 64-way generation. Each
         background load is sized from the alone run to outlast the measured one twice over.

Both servers share the card for the whole run: the generator at --gpu-memory-utilization 0.70
and the embedder at 0.15, so "alone" means "the other server idle", the same memory split as
"both". All commands go over ssh as one argument to the remote shell; nothing runs a local
shell. No GPU is used locally.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The nightly of 2026-10-09 (vLLM 0.31.1rc1.dev173, transformers 5.19.0). The sweep's digest
# (ec62abec..., transformers 5.18.0) registers EmbeddingGemma2Model in vLLM but its transformers cannot
# parse the `embedding_gemma2` config, so the server dies at startup. Both servers here use this one.
IMAGE = "vllm/vllm-openai-rocm@sha256:3b5af9b06c9bf2770f7efa433c0a63d163bc496a95f044fb5c4c706a7950a160"
HF = "/mnt/scratch/hf-cache"
GIDS = ("44", "991")
EMBED = {"name": "vllm-embed", "port": 8001, "model": "google/embeddinggemma-2", "gmu": "0.15",
         "extra": ["--max-model-len", "8192", "--limit-mm-per-prompt", '{"image": 0, "audio": 0}']}
GEN = {"name": "vllm-gen", "port": 8000, "model": "xbill9/gemma-4-12B-it-qat-q4_0-fp8-text", "gmu": "0.70",
       "extra": ["--max-model-len", "32768"]}
EMBED_GRID = [(c, n) for n in (128, 1024, 4096) for c in (1, 16, 128)]
GEN_GRID = [8, 64]
SEED = 70_000_000


def ssh(ip, cmd, timeout=1800):
    argv = ["ssh", "-i", os.path.expanduser("~/amd"), "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6", f"root@{ip}", cmd]
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s"


def q(s):
    return "'" + s.replace("'", "'\"'\"'") + "'"


def serve_cmd(srv):
    dev = " ".join(f"--group-add {g}" for g in GIDS)
    args = " ".join(q(a) for a in [srv["model"], "--host", "0.0.0.0", "--port", str(srv["port"]),
                                   "--gpu-memory-utilization", srv["gmu"], *srv["extra"]])
    return (f"docker rm -f {srv['name']} >/dev/null 2>&1; docker run -d --name {srv['name']} --ipc host "
            f"--shm-size 16g --device /dev/kfd --device /dev/dri {dev} --security-opt seccomp=unconfined "
            f"-v {HF}:/root/.cache/huggingface -p {srv['port']}:{srv['port']} {IMAGE} {args}")


def wait_ready(ip, srv, limit=1800):
    t = time.time()
    while time.time() - t < limit:
        code, out, _ = ssh(ip, f"curl -sf -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{srv['port']}/health", 30)
        if out.strip() == "200":
            return round(time.time() - t, 1)
        code, out, _ = ssh(ip, f"docker inspect -f '{{{{.State.Running}}}}' {srv['name']}", 30)
        if out.strip() == "false":
            return None
        time.sleep(15)
    return None


def bench(ip, srv, tag, extra, timeout=3600, detach=False):
    """One `vllm bench serve` in a GPU-less container. detach=True starts it and returns at once."""
    out = f"/dev/shm/bench-{tag}.json"
    cmd = (f"docker run --rm {'-d --name bench-' + tag if detach else ''} --net host -v /dev/shm:/dev/shm "
           f"-v {HF}:/root/.cache/huggingface --entrypoint vllm {IMAGE} bench serve "
           f"--base-url http://127.0.0.1:{srv['port']} --model {srv['model']} --dataset-name random "
           f"--save-result --result-dir /dev/shm --result-filename bench-{tag}.json {extra}")
    if detach:
        return ssh(ip, f"rm -f {out}; {cmd}", 120)
    code, o, e = ssh(ip, f"rm -f {out}; {cmd} > /dev/shm/bench-{tag}.log 2>&1; cat /dev/shm/bench-{tag}.log | tail -30; "
                         f"echo ---JSON---; cat {out}", timeout)
    log, _, js = o.partition("---JSON---")
    try:
        return json.loads(js), log
    except json.JSONDecodeError:
        return {"error": (e or log)[-800:]}, log


def wait_detached(ip, tag, timeout=7200):
    t = time.time()
    while time.time() - t < timeout:  # the result file, written as the bench exits, is the signal
        code, o, _ = ssh(ip, f"test -s /dev/shm/bench-{tag}.json && echo done", 30)
        if o.strip() == "done":
            time.sleep(5)
            break
        time.sleep(10)
    code, o, _ = ssh(ip, f"cat /dev/shm/bench-{tag}.json", 60)
    try:
        return json.loads(o)
    except json.JSONDecodeError:
        return {"error": "no result"}


def wait_loaded(ip, srv, limit=1200):  # a large random dataset takes minutes to prepare before the first request
    """Block until the server reports requests in flight, so a background load is running before
    the measured bench starts. A fixed sleep is not enough: the bench container takes tens of
    seconds to start."""
    # The request lines in the server's own log: the embedding server keeps
    # vllm:num_requests_running at 0 while serving ~220 req/s, so that metric cannot be the signal.
    path = "/v1/embeddings" if srv is EMBED else "/v1/completions"
    t = time.time()
    while time.time() - t < limit:
        code, o, _ = ssh(ip, f"docker logs --since 5s {srv['name']} 2>&1 | grep -c 'POST {path}'", 30)
        if o.strip().isdigit() and int(o.strip()) > 20:
            return round(time.time() - t, 1)
        time.sleep(3)
    return None


def overlap(ip, bg, fg):
    """True when the background bench ran across the whole measured one (from their result files)."""
    code, o, _ = ssh(ip, "python3 -c " + q(
        "import json,datetime as d\n"
        "def span(t):\n"
        "    j=json.load(open(f'/dev/shm/bench-{t}.json'));e=d.datetime.strptime(j['date'],'%Y%m%d-%H%M%S')\n"
        "    return e-d.timedelta(seconds=j['duration']),e\n"
        f"b=span('{bg}');f=span('{fg}');print(b[0]<=f[0] and b[1]>=f[1], b[0], b[1], f[0], f[1])"), 60)
    return o.strip()


def embed_args(c, n, prompts, seed):
    return (f"--backend openai-embeddings --endpoint /v1/embeddings --random-input-len {n} "
            f"--random-output-len 1 --num-prompts {prompts} --max-concurrency {c} --seed {seed}")


def gen_args(c, prompts, seed):
    return (f"--backend vllm --endpoint /v1/completions --random-input-len 1024 --random-output-len 512 "
            f"--ignore-eos --num-prompts {prompts} --max-concurrency {c} --seed {seed}")


def keep(j):
    keys = ("completed", "duration", "request_throughput", "total_input_tokens", "total_token_throughput",
            "input_throughput", "output_throughput", "mean_e2el_ms", "median_e2el_ms", "p99_e2el_ms",
            "median_ttft_ms", "median_tpot_ms", "error")
    return {k: j[k] for k in keys if k in j}


def check(ip, out):
    ref = json.load(open(HERE / "reference" / "reference.json"))
    res = {"reference": {k: ref[k] for k in ("model", "revision", "device", "torch", "transformers", "sentence_transformers")}}
    for dim in (None, 512, 256, 128):
        body = {"model": EMBED["model"], "input": ref["texts"]}
        if dim:
            body["dimensions"] = dim
        code, o, e = ssh(ip, f"curl -s http://127.0.0.1:{EMBED['port']}/v1/embeddings -H 'Content-Type: application/json' "
                             f"-d {q(json.dumps(body))}", 300)
        try:
            got = [d["embedding"] for d in json.loads(o)["data"]]
        except Exception:
            res[str(dim or 768)] = {"error": o[-600:] or e[-600:]}
            continue
        want = ref["embeddings" if not dim else f"embeddings_{dim}"]
        cos = []
        for a, b in zip(got, want):
            na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
            cos.append(sum(x * y for x, y in zip(a, b)) / (na * nb))
        nq, nd = ref["n_queries"], ref["n_docs"]
        top = []
        for i in range(nq):
            sims = [sum(x * y for x, y in zip(got[i], got[nq + j])) for j in range(nd)]
            top.append(max(range(nd), key=sims.__getitem__))
        res[str(dim or 768)] = {"dims": len(got[0]), "cos_min": round(min(cos), 6), "cos_mean": round(sum(cos) / len(cos), 6),
                                "cos_long_text": round(cos[-1], 6), "top_doc_agrees": sum(a == b for a, b in zip(top, ref["top_doc_per_query"])),
                                "queries": nq}
    (out / "check.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ip", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--phases", default="check,embed,gen,both")
    a = p.parse_args()
    out = HERE / "benchmarks" / "runs" / a.run_id
    out.mkdir(parents=True, exist_ok=True)
    phases = a.phases.split(",")
    results = json.loads((out / "results.json").read_text()) if (out / "results.json").exists() else {}
    def save():
        (out / "results.json").write_text(json.dumps(results, indent=1) + "\n")

    for srv in (GEN, EMBED):  # generator first: it claims the larger share of memory
        code, o, e = ssh(a.ip, f"docker ps -q -f name=^{srv['name']}$", 30)
        if not o.strip():
            ssh(a.ip, serve_cmd(srv), 300)
            t = wait_ready(a.ip, srv)
            print(f"{srv['name']} ready after {t}s", flush=True)
            results.setdefault("boot_seconds", {})[srv["name"]] = t
            if t is None:
                ssh_log = ssh(a.ip, f"docker logs {srv['name']} 2>&1 | tail -60", 60)[1]
                (out / f"boot-{srv['name']}.log").write_text(ssh_log)
                sys.exit(f"{srv['name']} did not start; log filed")
        log = ssh(a.ip, f"docker logs {srv['name']} 2>&1 | grep -vE 'GET /|POST /|Avg prompt|metrics'", 120)[1]
        (out / f"boot-{srv['name']}.log").write_text(log)
    save()

    seed = SEED
    if "check" in phases:
        check(a.ip, out)
    if "embed" in phases:
        for c, n in EMBED_GRID:
            seed += 1
            j, log = bench(a.ip, EMBED, f"embed-c{c}-in{n}", embed_args(c, n, max(64, c * 16), seed))
            results.setdefault("embed_alone", {})[f"c{c}-in{n}"] = keep(j)
            (out / f"embed-c{c}-in{n}.log").write_text(log)
            print(f"embed c{c} in{n}: {keep(j)}", flush=True)
            save()
    if "gen" in phases:
        for c in GEN_GRID:
            seed += 1
            j, log = bench(a.ip, GEN, f"gen-c{c}", gen_args(c, c * 4, seed))
            results.setdefault("gen_alone", {})[f"c{c}"] = keep(j)
            (out / f"gen-c{c}.log").write_text(log)
            print(f"gen c{c}: {keep(j)}", flush=True)
            save()
    if "both" in phases or "gen_under_embed" in phases or "embed_under_gen" in phases:
        ea, ga = results["embed_alone"]["c128-in1024"], results["gen_alone"]["c64"]
        # generator measured under a steady embedding load lasting about twice as long
    if "both" in phases or "gen_under_embed" in phases:
        n_bg = int(ea["request_throughput"] * (ga["duration"] + results["gen_alone"]["c8"]["duration"]) * 6) + 1000
        seed += 1
        bench(a.ip, EMBED, "bg-embed", embed_args(128, 1024, n_bg, seed), detach=True)
        results["bg_embed_load_after_s"] = wait_loaded(a.ip, EMBED)
        for c in GEN_GRID:
            seed += 1
            j, log = bench(a.ip, GEN, f"gen-c{c}-under-embed", gen_args(c, c * 4, seed))
            results.setdefault("gen_under_embed", {})[f"c{c}"] = keep(j)
            results.setdefault("overlap", {})[f"gen-c{c}-under-embed"] = "pending: checked after the load ends"
            (out / f"gen-c{c}-under-embed.log").write_text(log)
            print(f"gen c{c} under embed load: {keep(j)}", flush=True)
            save()
        results["bg_embed_during_gen"] = keep(wait_detached(a.ip, "bg-embed"))
        for c in GEN_GRID:
            results["overlap"][f"gen-c{c}-under-embed"] = overlap(a.ip, "bg-embed", f"gen-c{c}-under-embed")
        save()
    if "both" in phases or "embed_under_gen" in phases:
        # embedder measured under a steady 64-way generation load lasting about twice as long
        n_bg = int(ga["request_throughput"] * ea["duration"] * 2 * 3) + 256
        seed += 1
        bench(a.ip, GEN, "bg-gen", gen_args(64, n_bg, seed), detach=True)
        results["bg_gen_load_after_s"] = wait_loaded(a.ip, GEN)
        time.sleep(30)  # past the prefill wave, into steady decode
        for c, n in [(1, 1024), (16, 1024), (128, 1024)]:
            seed += 1
            j, log = bench(a.ip, EMBED, f"embed-c{c}-in{n}-under-gen", embed_args(c, n, max(64, c * 16), seed))
            results.setdefault("embed_under_gen", {})[f"c{c}-in{n}"] = keep(j)
            (out / f"embed-c{c}-in{n}-under-gen.log").write_text(log)
            print(f"embed c{c} in{n} under gen load: {keep(j)}", flush=True)
            save()
        results["bg_gen_during_embed"] = keep(wait_detached(a.ip, "bg-gen"))
        for c, n in [(1, 1024), (16, 1024), (128, 1024)]:
            results.setdefault("overlap", {})[f"embed-c{c}-in{n}-under-gen"] = overlap(a.ip, "bg-gen", f"embed-c{c}-in{n}-under-gen")
        save()
    print(f"wrote {out / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
