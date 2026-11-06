"""FLIGHT.md step 6 on the L4: concurrency sweeps and GSM8K/BFCL, each engine with the 3050 runs' settings.

    python3 suite.py OUT_DIR JOB[,JOB...]        JOB = sweep:<arm> | acc:<arm>

Run on the instance, monorepo at GEMMA4_DEV (default /opt/gemma4-dev). One server at a time, card cooled to
<= COOL_C first. Resumable: a job whose DONE marker exists is skipped.

sweep:<arm>  the 3050 concurrency runs' sweep.py arguments (2026-11-03/04-concurrency-3050oem): 512 in / 128 out
             and 128 in / 512 out, levels 1-64, 5 repeats, unique prompts, decode from the stream. sweep.py is
             gpu-vllm-g6-2b-q4w4a16emb4's (byte-identical to the SGLang and mistral.rs rigs'; llama.cpp's 3050
             copy lacks only vLLM's `reasoning` field, and thinking is off). GPU telemetry every second.
             arms: vllm, vllm-dN, sglang, sglang-dN (W4A16 drafter, N tokens), llama, llama-dN (exact Q4_0
             drafter; -c 65536 --parallel 64 as the 3050 sweep), mistralrs (3050 limits: 4 requests, 8,192
             tokens), mistralrs-card (32 requests, 16,384 tokens).
acc:<arm>    GSM8K (1,319, 2,048 tokens) and BFCL simple (400) through jev-tpu-v5e1/gen_eval.py, greedy, one
             request at a time, as the 3050 accuracy runs. Thinking off as each 3050 run set it
             (JEV_ENABLE_THINKING=0 for SGLang and mistral.rs; --reasoning off for llama.cpp; template default
             elsewhere). arms: vllm, vllm-dN, sglang, sglang-dN, llama, llama-dN (-c 131072 --parallel 1, the
             serving setting), mistralrs, pytorch, litert-int248, litert-q4 (through proxy.py, which moves
             max_tokens to max_completion_tokens: litert-lm 0.19.0 ignores max_tokens).
"""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

OUT, JOBS = sys.argv[1:3]
COOL_C = int(os.environ.get("COOL_C", "50"))
ROOT = os.environ.get("GEMMA4_DEV", "/opt/gemma4-dev")
VRIG = os.path.join(ROOT, "gpu-vllm-g6-2b-q4w4a16emb4")
LRIG = os.path.join(ROOT, "gpu-llamacpp-g6-2b-q4_0")
PRIG = os.path.join(ROOT, "gpu-pytorch-g6-2b-q4w4a16")
MRIG = os.path.join(ROOT, "gpu-mistralrs-g6-2b-q4_0")
SRIG = os.path.join(ROOT, "gpu-sglang-g6-2b-q4w4a16emb4")
LIRIG = os.path.join(ROOT, "gpu-litertlm-g6-2b-int248")
LQRIG = os.path.join(ROOT, "gpu-litertlm-g6-2b-q4w4a16emb4")
JEV = os.path.join(ROOT, "jev-tpu-v5e1")
HERE = os.path.dirname(os.path.abspath(__file__))
SYS_PY = "/usr/bin/python3"
LOCAL_BIN = os.path.expanduser("~/.local/bin")
LIMIT = ["--limit", os.environ["SUITE_LIMIT"]] if os.environ.get("SUITE_LIMIT") else []  # smoke tests only
signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # run the finally blocks: stop the server


def load_env(path):
    vals = {}
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            vals[k.strip()] = v.strip().strip('"')
    return vals


def gpu():
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,memory.used", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.strip()


def n_after(arm, prefix):
    return arm[len(prefix) :]


def server(kind, arm, d):
    """(cmd, env, url, model, extra gen_eval env)"""
    env = dict(os.environ)
    env["PATH"] = LOCAL_BIN + os.pathsep + env.get("PATH", "")  # pip --user scripts; SSM's PATH lacks it
    if arm.startswith("vllm"):
        v = load_env(os.path.join(VRIG, "tpu.env"))
        env["SPECULATIVE_MODEL"] = ""
        if arm.startswith("vllm-d"):
            env.update(SPECULATIVE_MODEL=v["DRAFTER_DIR"], NUM_SPECULATIVE_TOKENS=n_after(arm, "vllm-d"))
        return [SYS_PY, os.path.join(VRIG, "serve.py")], env, v["ENDPOINT"], v["MODEL_NAME"], {}
    if arm.startswith("sglang"):
        v = load_env(os.path.join(SRIG, "tpu.env"))
        env["SPECULATIVE_MODEL"] = ""
        if arm.startswith("sglang-d"):
            drafter = load_env(os.path.join(VRIG, "tpu.env"))["DRAFTER_DIR"]
            env.update(SPECULATIVE_MODEL=drafter, SPECULATIVE_NUM_STEPS=n_after(arm, "sglang-d"))
        return (
            [SYS_PY, os.path.join(SRIG, "serve.py")],
            env,
            v["ENDPOINT"],
            v["MODEL_NAME"],
            {"JEV_ENABLE_THINKING": "0"},
        )
    if arm.startswith("llama"):
        v = load_env(os.path.join(LRIG, "tpu.env"))
        ctx, par = ("65536", "64") if kind == "sweep" else ("131072", "1")
        cmd = [
            v["LLAMA_SERVER_BIN"],
            "-m",
            v["MODEL_PATH"],
            "--host",
            "127.0.0.1",
            "--port",
            "8099",
            "-ngl",
            "99",
            "-c",
            ctx,
            "--parallel",
            par,
            "-ctk",
            "f16",
            "-ctv",
            "f16",
            "-fa",
            "1",
            "-t",
            v["THREADS"],
            "-tb",
            v["THREADS_BATCH"],
            "--reasoning",
            "off",
            "-ot",
            v["OVERRIDE_TENSORS"],
            "--metrics",
        ]
        if arm.startswith("llama-d"):
            cmd += [
                "-md",
                v["SPEC_DRAFT_MODEL"],
                "--spec-type",
                "draft-mtp",
                "-ngld",
                "all",
                "--spec-draft-n-max",
                n_after(arm, "llama-d"),
            ]
        return cmd, env, "http://127.0.0.1:8099", v["MODEL_NAME"], {}
    if arm.startswith("mistralrs"):
        v = load_env(os.path.join(MRIG, "tpu.env"))
        env["MTP_MODEL"] = ""
        if arm == "mistralrs-card":  # 64 requests and 32 x 65,536 do not load on the L4 (2026-11-05-engines-g6)
            env.update(MAX_SEQS="32", KV_CACHE="16384tokens", MAX_NUM_BATCHED_TOKENS="4096")
        return [SYS_PY, os.path.join(MRIG, "serve.py")], env, v["ENDPOINT"], "default", {"JEV_ENABLE_THINKING": "0"}
    if arm == "pytorch":
        py = open("/opt/torch-g6-q4/PYTHON_BIN").read().strip()
        return (
            [py, os.path.join(PRIG, "openai_server.py"), "--port", "8100", "--max-ctx", "8192"],
            env,
            "http://127.0.0.1:8100",
            "google/gemma-4-E2B-it",
            {},
        )
    if arm in ("litert-int248", "litert-q4"):
        r = load_env(os.path.join({"litert-q4": LQRIG, "litert-int248": LIRIG}[arm], "tpu.env"))
        cfg = os.path.join(d, "config.json")
        json.dump(
            {"models": {r["MODEL_ID"]: {"backend": "gpu", "cache": "disk", "speculative_decoding": "off"}}},
            open(cfg, "w"),
        )
        env["LITERT_LM_DIR"] = r["LITERT_LM_DIR"]
        cmd = [
            "bash",
            "-c",
            f"{SYS_PY} {HERE}/proxy.py 9398 http://127.0.0.1:9399 {r['MODEL_ID']} & "
            f"exec {LOCAL_BIN}/litert-lm serve --config {cfg} --host 127.0.0.1 --port 9399",
        ]
        return cmd, env, "http://127.0.0.1:9398", "google/gemma-4-E2B-it", {}
    raise SystemExit(f"unknown arm {arm}")


def ready(url, model):
    """One real chat reply (mistral.rs answers /health before it admits requests; LiteRT has no /health)."""
    b = json.dumps(
        {"model": model, "messages": [{"role": "user", "content": "ok"}], "max_tokens": 2, "temperature": 0}
    ).encode()
    try:
        urllib.request.urlopen(
            urllib.request.Request(url + "/v1/chat/completions", data=b, headers={"Content-Type": "application/json"}),
            timeout=60,
        ).read()
        return True
    except Exception:
        return False


def log(msg):
    line = f"{time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    open(os.path.join(OUT, "driver.log"), "a").write(line + "\n")


def run_job(job):
    kind, arm = job.split(":", 1)
    d = os.path.join(OUT, f"{kind}-{arm}")
    if os.path.exists(os.path.join(d, "DONE")):
        log(f"SKIP {job} (done)")
        return
    os.makedirs(d, exist_ok=True)
    cmd, env, url, model, gen_env = server(kind, arm, d)
    while int(gpu().split(",")[0]) > COOL_C:
        time.sleep(5)
    open(os.path.join(d, "command.txt"), "w").write(" ".join(cmd) + "\n")
    slog = open(os.path.join(d, "server.log"), "w")
    t0 = time.time()
    srv = subprocess.Popen(cmd, stdout=slog, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    while srv.poll() is None and not ready(url, model):
        if time.time() - t0 > 1800:
            break
        time.sleep(3)
    if srv.poll() is not None or not ready(url, model):
        log(f"FAIL {job}: server not ready, see {d}/server.log")
        if srv.poll() is None:
            os.killpg(srv.pid, signal.SIGTERM)
            srv.wait()
        return
    log(f"START {job} ready_s={time.time() - t0:.1f} gpu={gpu()}")
    smi = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,memory.used,utilization.gpu",
            "--format=csv,noheader",
            "-l",
            "1",
        ],
        stdout=open(os.path.join(d, "gpu.csv"), "w"),
    )
    ok = True
    try:
        if kind == "sweep":
            for i, o in ((512, 128), (128, 512)):
                rc = subprocess.run(
                    [
                        SYS_PY,
                        os.path.join(VRIG, "sweep.py"),
                        "--base",
                        url + "/v1",
                        "--model",
                        model,
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
                        "gpu",
                    ],
                    stdout=open(os.path.join(d, f"sweep-in{i}-out{o}.log"), "w"),
                    stderr=subprocess.STDOUT,
                    env=env,
                ).returncode
                log(f"  {job} in{i}-out{o} rc={rc}")
                ok = ok and rc == 0
        else:
            genv = dict(env, JEV_GEN_MAX_TOKENS="2048", **gen_env)
            for t in ("bfcl_simple", "gsm8k"):
                p = subprocess.run(
                    [SYS_PY, "gen_eval.py", "run", url, model, t, os.path.join(d, f"{t}.jsonl"), "--concurrency", "1"] + LIMIT,
                    cwd=JEV,
                    env=genv,
                    capture_output=True,
                    text=True,
                )
                open(os.path.join(d, f"{t}.log"), "w").write(p.stdout + p.stderr)
                last = (p.stdout.strip().splitlines() or [""])[-1]
                log(f"  {job} {t} rc={p.returncode} {last} gpu={gpu()}")
                ok = ok and p.returncode == 0
    finally:
        smi.terminate()
        os.killpg(srv.pid, signal.SIGTERM)
        try:
            srv.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(srv.pid, signal.SIGKILL)
            srv.wait()
        slog.close()
    if ok:
        open(os.path.join(d, "DONE"), "w").write("")
    log(f"END {job} ok={ok}")
    while int(gpu().split(",")[2].split()[0]) > 100:
        time.sleep(2)


os.makedirs(OUT, exist_ok=True)
for j in JOBS.split(","):
    run_job(j)
log("SUITE-DONE")
