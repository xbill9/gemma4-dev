"""The v5e-1 ladder: every engine that serves Gemma 4 E2B on one TPU v5e chip, measured the way the G6 ladder
measured the L4 (gpu-vllm-g6-2b-q4w4a16emb4/flight/engines.py and suite.py, 2026-11-05/06).

    python3.13 ladder.py OUT_DIR JOB[,JOB...]       JOB = <kind>[+<kind>...]:<arm>, kind one | sweep | acc (one server start per JOB)

Run on the TPU VM, files under LADDER (default /opt/ladder). One server at a time on port 8000. Resumable: a
job whose DONE marker exists is skipped. driver.log in OUT_DIR.

Arms
  vllm            the rig's patched image (vllm-tpu-q4w4a16emb4:patched), xbill9/...-w4a16-ct-text-emb4 (int4
                  linears and vocabulary tables, the G6 ladder's checkpoint), the rig's serving flags plus
                  --max-num-seqs 64
  vllm-text       the same image and flags, xbill9/...-w4a16-ct-text (int4 linears, bf16 tables): the drafter
                  copies the target's bf16 embed_tokens and cannot pair with the -emb4 build (QUANTIZATION.md)
  vllm-text-dN    vllm-text with google/gemma-4-E2B-it-assistant as an MTP drafter, N drafted tokens (on this
                  chip the QAT drafter was slower than none at every load, 2026-09-29)
  jax             jax-dev/tpu-jax-v5e1-2b-q4w4a16 jax_openai_server.py, the -text checkpoint, --quant-mode w4a16
  jax-bf16        jax-dev/tpu-jax-v5e1-2b jax_openai_server.py, google/gemma-4-E2B-it, --quant-mode fp16
  jaxrust         jax-dev/tpu-jaxrust-v5e1-2b rust_openai_server.py, google/gemma-4-E2B-it (bf16 only)
  torchtpu        torchtpu-dev/tpu-pytorch-v5e1-2b-w4a16 tpu_openai_server.py, google/gemma-4-E2B-it-qat-w4a16-ct
  torchtpu-bf16   the same server file (identical in tpu-pytorch-v5e1-2b), google/gemma-4-E2B-it (bf16)
  torchtpu-rust   torchtpu-dev/tpu-pytorchrust-v5e1-2b-q4w4a16emb4: the torchtpu-rs systemd unit its startup
                  script installs (the -text linears with the -emb4 int4 tables), started and stopped here
  sglang          SGLang-JAX (sgl-project/sglang-jax, SGL_COMMIT in setup-sgl.sh), google/gemma-4-E2B-it bf16,
                  --reasoning-parser gemma4; it has no Gemma 4 tool-call parser, so no BFCL. Its acc job scores
                  the first 50 GSM8K questions first and runs the rest (and the sweep) only at accuracy >= 0.5

Jobs
  one     the G6 engines.py requests: six prompts, greedy, thinking off, 256 tokens; non-streaming (e2e) and
          streaming (decode tok/s, TTFT), 3 repeats each; then 4 requests at once, 3 repeats -> results.jsonl
  sweep   sweep.py, 512 in / 128 out and 128 in / 512 out, 1-64 requests, 5 repeats, unique prompts, decode
          from the stream (the G6 suite's arguments; --expect-device any)
  acc     GSM8K (1,319, 2,048 tokens) through jev-tpu-v5e1/gen_eval.py, greedy, one request at a time; BFCL simple
          (400) only for the vLLM arms: the other servers apply the chat template without the request's tools and
          return no tool_calls, so gen_eval would score their BFCL 0 for a missing server feature
"""

import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request

OUT, JOBS = sys.argv[1:3]
L = os.environ.get("LADDER", "/opt/ladder")
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
PORT = 8000
URL = f"http://127.0.0.1:{PORT}"
JEV = os.path.join(L, "gemma4-dev", "jev-tpu-v5e1")
IMAGE = "vllm-tpu-q4w4a16emb4:patched"
EMB4 = "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4"
TEXT = "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text"
BF16 = "google/gemma-4-E2B-it"
DRAFTER = "google/gemma-4-E2B-it-assistant"
LIMIT = ["--limit", os.environ["SUITE_LIMIT"]] if os.environ.get("SUITE_LIMIT") else []  # smoke tests only
# Arms too slow for the full GSM8K inside a 24-hour flex-start run score its first N questions (gen_eval --limit
# takes the first N in file order). TorchTPU Python, bf16: 1,000 s for the first 5 in the smoke test.
ARM_LIMIT = {"torchtpu-bf16": 200}
PROMPTS = {
    "code": "Write a Python function that checks whether a number is prime, with a docstring and two doctests.",
    "chat": "I just moved to a new city and feel a bit lonely. Any advice?",
    "story": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "explain": "Explain how a refrigerator works to a ten-year-old.",
    "facts": "List the planets of the solar system with one fact about each.",
    "summary": "Summarize the causes of the French Revolution in one paragraph.",
}
REPEATS = 3
signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # run the finally blocks: stop the server


def hf_token():
    p = os.path.join(L, "hf_token")
    return open(p).read().strip() if os.path.exists(p) else ""


class Server:
    """start() launches the arm's server; stop() ends it and waits for port 8000 to fall silent."""

    def __init__(self, arm):
        self.arm, self.proc, self.unit = arm, None, None
        self.env = dict(os.environ, HF_TOKEN=hf_token(), HF_HOME="/opt/hf")
        self.model, self.bfcl, self.cwd, self.broken = BF16, False, None, False
        a = arm
        if a.startswith("vllm"):
            self.model = EMB4 if a == "vllm" else TEXT
            self.bfcl = True
            flags = [
                "--host",
                "127.0.0.1",
                "--port",
                str(PORT),
                "--max-model-len",
                "16384",
                "--max-num-seqs",
                "64",
                "--tensor-parallel-size",
                "1",
                "--disable_chunked_mm_input",
                "--max_num_batched_tokens",
                "512",
                "--gpu-memory-utilization",
                # -text weights + the drafter ran out of HBM in warmup at 0.80 (smoke test, 2026-11-06)
                # 4 drafted tokens ran out at 0.70 (needed 512 MiB, 429.5 MiB free, 2026-11-07 01:47)
                "0.60" if a == "vllm-text-d4" else "0.70" if a.startswith("vllm-text-d") else "0.80",
                "--limit-mm-per-prompt",
                '{"image":0,"audio":0,"video":0}',
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "gemma4",
                "--reasoning-parser",
                "gemma4",
            ]
            if a.startswith("vllm-text-d"):
                n = int(a[len("vllm-text-d") :])
                flags += [
                    "--speculative-config",
                    json.dumps({"method": "mtp", "model": DRAFTER, "num_speculative_tokens": n}),
                ]
            self.cmd = [
                "sudo",
                "docker",
                "run",
                "--rm",
                "--name",
                "ladder-vllm",
                "--privileged",
                "--net=host",
                "--shm-size",
                "10gb",
                "-v",
                "/dev/shm:/dev/shm",
                "-v",
                "/opt/hf:/root/.cache/huggingface",
                "-e",
                "HF_TOKEN",
                IMAGE,
                "vllm",
                "serve",
                self.model,
            ] + flags
        elif a in ("jax", "jax-bf16"):
            rig = "tpu-jax-v5e1-2b-q4w4a16" if a == "jax" else "tpu-jax-v5e1-2b"
            self.model = TEXT if a == "jax" else BF16
            self.cwd = os.path.join(L, "jax-dev", rig)
            self.cmd = [
                PY,
                "jax_openai_server.py",
                "--model",
                self.model,
                "--quant-mode",
                "w4a16" if a == "jax" else "fp16",
                "--max-model-len",
                "4096",
                "--port",
                str(PORT),
            ]
        elif a == "jaxrust":
            self.cwd = os.path.join(L, "jax-dev", "tpu-jaxrust-v5e1-2b")
            self.env["PJRT_PLUGIN"] = "/usr/local/lib/python3.13/dist-packages/libtpu/libtpu.so"
            self.cmd = [PY, "rust_openai_server.py", "--context", "4096", "--port", str(PORT)]
        elif a == "torchtpu":
            self.model = "google/gemma-4-E2B-it-qat-w4a16-ct"
            self.cwd = os.path.join(L, "torchtpu-dev", "tpu-pytorch-v5e1-2b-w4a16")
            self.cmd = [PY, "tpu_openai_server.py", "--model", self.model, "--seq", "4096", "--port", str(PORT)]
        elif a == "torchtpu-bf16":
            self.model = "google/gemma-4-E2B-it"
            self.cwd = os.path.join(
                L, "torchtpu-dev", "tpu-pytorch-v5e1-2b-w4a16"
            )  # tpu_openai_server.py identical to tpu-pytorch-v5e1-2b's
            self.cmd = [PY, "tpu_openai_server.py", "--model", self.model, "--seq", "4096", "--port", str(PORT)]
        elif a == "sglang":
            self.cmd = [
                PY,
                "-m",
                "sgl_jax.launch_server",
                "--model-path",
                BF16,
                "--host",
                "127.0.0.1",
                "--port",
                str(PORT),
                "--device",
                "tpu",
                "--tp-size",
                "1",
                "--dtype",
                "bfloat16",
                "--context-length",
                "16384",
                "--max-running-requests",
                "64",
                "--mem-fraction-static",
                "0.8",
                "--reasoning-parser",
                "gemma4",
                # page size 1 caps running requests at 131072 // 16384 = 8 (FlashAttention backend's scalar-prefetch
                # budget); sglang-jax's Gemma 2 recipe requires 128. Lazy Pallas compiles of the 512-wide full-attention
                # kernel take minutes, longer than the default 300 s watchdog.
                "--page-size",
                "128",
                "--watchdog-timeout",
                "7200",
            ]
        elif a == "torchtpu-rust":
            self.model = EMB4
            self.unit = "torchtpu-rs"
            self.cmd = ["sudo", "systemctl", "start", self.unit]
        else:
            raise SystemExit(f"unknown arm {arm}")

    def start(self, log_path):
        self.log = open(log_path, "w")
        if self.unit:
            subprocess.run(self.cmd, check=True)
            self.journal = subprocess.Popen(
                ["sudo", "journalctl", "-fu", self.unit, "-n", "0"],
                stdout=self.log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        else:
            self.proc = subprocess.Popen(
                self.cmd, cwd=self.cwd, env=self.env, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True
            )

    def alive(self):
        if self.unit:
            return subprocess.run(["systemctl", "is-active", "--quiet", self.unit]).returncode == 0
        return self.proc.poll() is None

    def stop(self):
        if self.unit:
            subprocess.run(["sudo", "systemctl", "stop", self.unit])
            os.killpg(self.journal.pid, signal.SIGTERM)
        elif self.arm.startswith("vllm"):
            subprocess.run(["sudo", "docker", "rm", "-f", "ladder-vllm"], capture_output=True)
            self.proc.wait()
        else:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        for _ in range(120):
            try:
                urllib.request.urlopen(URL + "/v1/models", timeout=2)
                time.sleep(2)
            except Exception:
                break
        time.sleep(5)
        self.log.close()


def body(model, prompt, stream, max_tokens=256):
    b = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "max_completion_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if stream:
        b.update(stream=True, stream_options={"include_usage": True})
    return json.dumps(b).encode()


def post(data, timeout=900):
    return urllib.request.urlopen(
        urllib.request.Request(URL + "/v1/chat/completions", data, {"Content-Type": "application/json"}),
        timeout=timeout,
    )


def ready(model):
    try:
        with post(body(model, "ok", False, 2), timeout=120) as r:
            r.read()
        return True
    except Exception:
        return False


def e2e(model, prompt):
    t0 = time.perf_counter()
    with post(body(model, prompt, False)) as r:
        d = json.load(r)
    w = time.perf_counter() - t0
    return d["usage"]["completion_tokens"], w, d["choices"][0]["message"].get("content") or ""


def stream(model, prompt):
    stamps, text, n = [], [], 0
    t0 = time.perf_counter()
    with post(body(model, prompt, True)) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")
                if piece:
                    stamps.append(time.perf_counter())
                    text.append(piece)
            if d.get("usage"):
                n = d["usage"].get("completion_tokens", n)
    tps = (n - 1) / (stamps[-1] - stamps[0]) if n > 1 and len(stamps) > 1 else 0.0
    return n, tps, (stamps[0] - t0) if stamps else 0.0, "".join(text)


def log(msg):
    line = f"{time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    open(os.path.join(OUT, "driver.log"), "a").write(line + "\n")


def job_one(s, d, base):
    out = open(os.path.join(OUT, "results.jsonl"), "a")

    def emit(rec):
        out.write(json.dumps(rec) + "\n")
        out.flush()

    m = s.model
    stream(m, "Hello")
    for name, prompt in PROMPTS.items():
        for rep in range(REPEATS):
            n, w, text = e2e(m, prompt)
            emit(
                dict(
                    base,
                    kind="e2e",
                    prompt=name,
                    rep=rep,
                    tokens=n,
                    wall_s=w,
                    tps=n / w,
                    sha=hashlib.sha256(text.encode()).hexdigest()[:16],
                )
            )
    for name, prompt in PROMPTS.items():
        for rep in range(REPEATS):
            n, tps, ttft, text = stream(m, prompt)
            emit(
                dict(
                    base,
                    kind="stream",
                    prompt=name,
                    rep=rep,
                    tokens=n,
                    tps=tps,
                    ttft_s=ttft,
                    sha=hashlib.sha256(text.encode()).hexdigest()[:16],
                )
            )
    four = list(PROMPTS.values())[:4]
    for rep in range(REPEATS):
        res = [None] * 4

        def go(i):
            res[i] = e2e(m, four[i])

        ts = [threading.Thread(target=go, args=(i,)) for i in range(4)]
        t0 = time.perf_counter()
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        w = time.perf_counter() - t0
        tok = sum(r[0] for r in res if r)
        emit(dict(base, kind="batch4", rep=rep, tokens=tok, wall_s=w, tps=tok / w, ok=sum(r is not None for r in res)))
    out.close()
    return True


def job_sweep(s, d, base):
    if s.broken:
        log(f"  sweep:{s.arm} skipped: GSM8K gate failed")
        open(os.path.join(d, "SKIPPED"), "w").write("GSM8K gate failed\n")
        return True
    ok = True
    for i, o in ((512, 128), (128, 512)):
        rc = subprocess.run(
            [
                PY,
                os.path.join(HERE, "sweep.py"),
                "--base",
                URL + "/v1",
                "--model",
                s.model,
                "--out",
                os.path.join(d, f"in{i}-out{o}"),
                "--concurrency",
                "1,2,4,8,16,32,64",
                "--conc-input",
                str(i),
                "--conc-output",
                str(o),
                "--repeats",
                "5",
                "--prompt-mode",
                "unique",
                "--decode-source",
                "auto",
                "--expect-device",
                "any",
            ],
            stdout=open(os.path.join(d, f"sweep-in{i}-out{o}.log"), "w"),
            stderr=subprocess.STDOUT,
        ).returncode
        log(f"  sweep:{s.arm} in{i}-out{o} rc={rc}")
        ok = ok and rc == 0
    return ok


def job_acc(s, d, base):
    ok = True
    tasks = (["bfcl_simple"] if s.bfcl else []) + ["gsm8k"]
    if s.arm == "sglang" and not LIMIT:
        p = subprocess.run(
            [PY, "gen_eval.py", "run", URL, s.model, "gsm8k", os.path.join(d, "gsm8k-gate.jsonl"), "--concurrency", "1",
             "--limit", "50"],
            cwd=JEV,
            env=dict(os.environ, JEV_GEN_MAX_TOKENS="2048"),
            capture_output=True,
            text=True,
        )
        open(os.path.join(d, "gsm8k-gate.log"), "w").write(p.stdout + p.stderr)
        last = (p.stdout.strip().splitlines() or [""])[-1]
        log(f"  acc:{s.arm} gsm8k-gate rc={p.returncode} {last}")
        try:
            accu = json.loads(last)["accuracy"]
        except Exception:
            accu = 0.0
        if accu < 0.5:
            s.broken = True
            log(f"  acc:{s.arm} GATE accuracy {accu} < 0.5 on the first 50: full GSM8K and sweep not run")
            return True
    for t in tasks:
        p = subprocess.run(
            [PY, "gen_eval.py", "run", URL, s.model, t, os.path.join(d, f"{t}.jsonl"), "--concurrency", "1"]
            + (LIMIT or (["--limit", str(ARM_LIMIT[s.arm])] if s.arm in ARM_LIMIT else [])),
            cwd=JEV,
            env=dict(os.environ, JEV_GEN_MAX_TOKENS="2048"),
            capture_output=True,
            text=True,
        )
        open(os.path.join(d, f"{t}.log"), "w").write(p.stdout + p.stderr)
        last = (p.stdout.strip().splitlines() or [""])[-1]
        log(f"  acc:{s.arm} {t} rc={p.returncode} {last}")
        ok = ok and p.returncode == 0
    return ok


def run_job(job):
    """`kind:arm`, or `kind+kind+...:arm` to run several kinds against one server start (each still gets its
    own directory and DONE marker; the server's ready time is recorded in each)."""
    kinds, arm = job.split(":", 1)
    todo = []
    for kind in kinds.split("+"):
        d = os.path.join(OUT, f"{kind}-{arm}")
        if os.path.exists(os.path.join(d, "DONE")):
            log(f"SKIP {kind}:{arm} (done)")
        else:
            os.makedirs(d, exist_ok=True)
            todo.append((kind, d))
    if not todo:
        return
    s = Server(arm)
    d0 = todo[0][1]
    for _, d in todo:
        open(os.path.join(d, "command.txt"), "w").write(" ".join(s.cmd) + "\n")
    t0 = time.time()
    s.start(os.path.join(d0, "server.log"))
    try:
        while s.alive() and not ready(s.model) and time.time() - t0 < (10800 if arm == "sglang" else 3600):
            time.sleep(5)
        if not ready(s.model):
            log(f"FAIL {job}: server not ready after {time.time() - t0:.0f} s, see {d0}/server.log")
            return
        ready_s = round(time.time() - t0, 1)
        for kind, d in todo:
            log(f"START {kind}:{arm} ready_s={ready_s} server_log={d0}/server.log")
            base = {"arm": arm, "ready_s": ready_s}
            try:
                ok = {"one": job_one, "sweep": job_sweep, "acc": job_acc}[kind](s, d, base)
            except Exception as e:
                log(f"FAIL {kind}:{arm}: {type(e).__name__}: {e}")
                ok = False
            if ok:
                open(os.path.join(d, "DONE"), "w").write("")
            log(f"END {kind}:{arm} ok={ok}")
            if not s.alive() or not ready(s.model):
                log(f"FAIL {job}: server gone after {kind}")
                return
    finally:
        s.stop()


os.makedirs(OUT, exist_ok=True)
for j in JOBS.split(","):
    run_job(j)
log("SUITE-DONE")
