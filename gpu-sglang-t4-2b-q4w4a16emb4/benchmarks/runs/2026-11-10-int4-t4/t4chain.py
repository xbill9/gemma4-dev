"""Load sweeps and accuracy runs on the Tesla T4, one engine arm at a time, through engines.py's servers.

    python3 t4chain.py sweep    ARM OUT_DIR     # sweep.py, 512/128 and 128/512, 1-64 requests, 5 repeats
    python3 t4chain.py accuracy ARM OUT_DIR     # gen_eval.py, BFCL simple (400) then GSM8K (1,319)

The server for ARM comes from engines.server(), so the one-request run, the sweeps and the accuracy runs start
every engine the same way. The requests are the 3050 study's, unchanged:
 - sweep: each engine rig's own sweep.py with the arguments of the 2026-11-03/04 3050 concurrency runs
   (unique prompts, thinking on, decode from the stream, --expect-device gpu). llama.cpp at -c 65536
   --parallel 64, as there.
 - accuracy: jev-tpu-v5e1/gen_eval.py, one request at a time, greedy, thinking off, GSM8K at a 2,048-token
   limit. llama.cpp at -c 131072 --parallel 1, its serving setting.
SWEEP_LEVELS, SWEEP_REPEATS and EVAL_LIMIT shrink a job for a smoke test.
Writes OUT_DIR/{command.txt, server.log, driver.log, gpu.csv, ...}; the sweep writes OUT_DIR/in512-out128 and
OUT_DIR/in128-out512, accuracy OUT_DIR/{bfcl_simple,gsm8k}-ARM.jsonl."""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engines  # noqa: E402

JOB, ARM, OUT = sys.argv[1:4]
ROOT = engines.ROOT
SWEEP_RIG = {"vllm": engines.VRIG, "llama": engines.LRIG, "sglang": engines.SRIG, "sglang4": engines.SRIG4, "mistralrs": engines.MRIG}
JEV = os.path.join(ROOT, "jev-tpu-v5e1")


def log(msg):
    line = f"{time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    with open(os.path.join(OUT, "driver.log"), "a") as fh:
        fh.write(line + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    engines.OUT = OUT
    if ARM.startswith("llama"):
        ctx, par = ("65536", "64") if JOB == "sweep" else ("131072", "1")
        os.environ.setdefault("LLAMA_CTX", ctx)
        os.environ.setdefault("LLAMA_PARALLEL", par)
    cmd, env, url, model, health, _ = engines.server(ARM)
    with open(os.path.join(OUT, "command.txt"), "w") as fh:
        fh.write(" ".join(cmd) + "\n")
        for k in (
            "SPECULATIVE_MODEL",
            "NUM_SPECULATIVE_TOKENS",
            "SPECULATIVE_NUM_STEPS",
            "MTP_MODEL",
            "MTP_N_PREDICT",
            "KV_CACHE",
            "MAX_MODEL_LEN",
        ):
            if env.get(k):
                fh.write(f"{k}={env[k]}\n")
    while float(engines.gpu().split(",")[0]) > engines.COOL_C:
        time.sleep(5)
    srvlog = open(os.path.join(OUT, "server.log"), "w")
    t0 = time.time()
    srv = subprocess.Popen(cmd, stdout=srvlog, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    smi = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,memory.used,utilization.gpu",
            "--format=csv,noheader",
            "-l",
            "1",
        ],
        stdout=open(os.path.join(OUT, "gpu.csv"), "w"),
    )
    rc = 1
    try:
        while srv.poll() is None:
            try:
                urllib.request.urlopen(url + health, timeout=2)
                break
            except Exception:
                time.sleep(2)
        if srv.poll() is not None:
            log(f"{ARM}: server exited rc={srv.returncode}, see server.log")
            return 1
        for _ in range(300):  # ready = first real reply (mistral.rs answers /health before it admits requests)
            try:
                engines.e2e(url, model, "Hello")
                break
            except Exception:
                time.sleep(2)
        log(f"{ARM}: ready in {time.time() - t0:.1f} s, gpu {engines.gpu()}")
        tenv = dict(os.environ, PYTHONUSERBASE="")  # the harness scripts are stdlib only
        if JOB == "sweep":
            rig = SWEEP_RIG[ARM.split("-")[0]]
            for shape in ((512, 128), (128, 512)):
                name = f"in{shape[0]}-out{shape[1]}"
                log(f"START sweep {name}")
                r = subprocess.run(
                    [
                        "python3",
                        os.path.join(rig, "sweep.py"),
                        "--base",
                        url + "/v1",
                        "--model",
                        model,
                        "--out",
                        os.path.join(OUT, name),
                        "--concurrency",
                        os.environ.get("SWEEP_LEVELS", "1,2,4,8,16,32,64"),
                        "--conc-input",
                        str(shape[0]),
                        "--conc-output",
                        str(shape[1]),
                        "--repeats",
                        os.environ.get("SWEEP_REPEATS", "5"),
                        "--prompt-mode",
                        "unique",
                        "--decode-source",
                        "auto",
                        "--expect-device",
                        "gpu",
                    ],
                    cwd=rig,
                    env=tenv,
                    stdout=open(os.path.join(OUT, f"sweep-{name}.log"), "w"),
                    stderr=subprocess.STDOUT,
                )
                log(f"END sweep {name} rc={r.returncode}")
                rc = r.returncode
        elif JOB == "accuracy":
            rc = 0
            for task in ("bfcl_simple", "gsm8k"):
                log(f"START {task} gpu={engines.gpu()}")
                path = os.path.join(OUT, f"{task}-{ARM}.jsonl")
                with open(os.path.join(OUT, f"{task}-{ARM}.log"), "w") as fh:
                    r = subprocess.run(
                        ["python3", "gen_eval.py", "run", url, model, task, path, "--concurrency", "1"]
                        + (["--limit", os.environ["EVAL_LIMIT"]] if os.environ.get("EVAL_LIMIT") else []),
                        cwd=JEV,
                        env=dict(tenv, JEV_GEN_MAX_TOKENS="2048"),
                        stdout=fh,
                        stderr=subprocess.STDOUT,
                    )
                with open(os.path.join(OUT, f"{task}-{ARM}.log")) as fh:
                    last = (fh.read().strip().splitlines() or [""])[-1]
                log(f"END {task} rc={r.returncode} {last}")
                rc = rc or r.returncode
        else:
            raise SystemExit(f"unknown job {JOB}")
        with open(os.path.join(OUT, "vram.txt"), "w") as fh:
            fh.write(
                subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv"],
                    capture_output=True,
                    text=True,
                ).stdout
            )
    finally:
        os.killpg(srv.pid, signal.SIGTERM)
        try:
            srv.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(srv.pid, signal.SIGKILL)
            srv.wait()
        smi.terminate()
        while int(engines.gpu().split(",")[2].split()[0]) > 100:
            time.sleep(2)
        log(f"{ARM}: stopped, rc={rc}")
        with open(os.path.join(OUT, "status.json"), "w") as fh:
            json.dump({"arm": ARM, "job": JOB, "rc": rc}, fh)
    return rc


if __name__ == "__main__":
    sys.exit(main())
