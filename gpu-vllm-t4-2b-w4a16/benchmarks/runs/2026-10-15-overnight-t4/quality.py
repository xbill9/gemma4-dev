"""Does serving with the drafter change answers? And the first quality scores for the T4 builds.

    python3 quality.py RIG_DIR OUT_DIR TAG ARM[,ARM...]

Per arm one vLLM server (t4serve.py, the rig's serving flags plus --enable-auto-tool-choice
--tool-call-parser gemma4 --max-logprobs 32, which the harnesses need), then:
  GSM8K, all 1,319 test problems, jev-tpu-v5e1/gen_eval.py, JEV_GEN_MAX_TOKENS=2048, greedy, 8 at a time
  BFCL simple, 400 tool-calling records, gen_eval.py, greedy, 8 at a time
  on the `none` arm only, when SUITE_DATA holds the rebuilt public suite: the Nimble 13-subset
  suite (3,880 records), nimble_suite/run_suite.py, JEV_TOPK=32, 8 at a time. The suite reads one
  token's logprobs per record, which a drafter cannot change, so it runs without one.
Outputs: OUT_DIR/{gsm8k,bfcl}-TAG-ARM.jsonl, OUT_DIR/suite-TAG/ and logs. Scoring is gen_eval.py's
and score_suite.py's own code (compare.py).
"""
import os, shutil, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t4serve import load_env, parse_arm, start, stop, gpu

RIG, OUT, TAG, ARMLIST = sys.argv[1:5]
J = os.path.expanduser("~/gemma4-dev/jev-tpu-v5e1")
SUITE_DATA = os.environ.get("SUITE_DATA", "/opt1/data/nimble-suite/public")
env = load_env(RIG); URL = env["ENDPOINT"]; MODEL = env["MODEL_NAME"]
PY = [env["PYTHON_BIN"]]
cenv = dict(os.environ, PYTHONUSERBASE=env["PYTHONUSERBASE"], TMPDIR=env["TMPDIR"], HF_HUB_OFFLINE="1")
os.makedirs(OUT, exist_ok=True)

def run(name, cmd, extra_env=None):
    print(f"START {name} {gpu()}", flush=True)
    with open(os.path.join(OUT, name + ".log"), "w") as log:
        rc = subprocess.call(cmd, cwd=J, stdout=log, stderr=subprocess.STDOUT, env=dict(cenv, **(extra_env or {})))
    print(f"END {name} rc={rc} {gpu()}", flush=True)

for arm in ARMLIST.split(","):
    draft, n = parse_arm(env, arm)
    srv, log = start(env, os.path.join(OUT, f"server-{TAG}-{arm}.log"), draft, n,
                     ["--enable-auto-tool-choice", "--tool-call-parser", "gemma4", "--max-logprobs", "32"])
    try:
        TASKS = "" if os.environ.get("SUITE_ONLY") else os.environ.get("TASKS", "gsm8k,bfcl")
        if "gsm8k" in TASKS:
            run(f"gsm8k-{TAG}-{arm}", PY + ["gen_eval.py", "run", URL, MODEL, "gsm8k",
                os.path.join(OUT, f"gsm8k-{TAG}-{arm}.jsonl"), "--concurrency", "8"], {"JEV_GEN_MAX_TOKENS": "2048"})
        if "bfcl" in TASKS:
            run(f"bfcl-{TAG}-{arm}", PY + ["gen_eval.py", "run", URL, MODEL, "bfcl_simple",
                os.path.join(OUT, f"bfcl-{TAG}-{arm}.jsonl"), "--concurrency", "8"])
        if arm.startswith("none") and not os.path.isdir(os.path.join(OUT, f"suite-{TAG}")):
            if os.path.exists(os.path.join(os.path.dirname(SUITE_DATA), "BUILD_OK")):
                runname = f"2026-10-15-suite-t4-{TAG}"
                run(f"suite-{TAG}", PY + ["nimble_suite/run_suite.py", "--arm", "autoregressive", "--upstream", URL,
                    "--model", MODEL, "--records", SUITE_DATA, "--run", runname, "--concurrency", "8"],
                    {"JEV_TOPK": "32"})
                src = os.path.join(J, "results", runname)
                if os.path.isdir(src):
                    shutil.move(src, os.path.join(OUT, f"suite-{TAG}"))
            else:
                print(f"SKIP suite-{TAG}: no rebuilt suite at {SUITE_DATA}", flush=True)
    finally:
        stop(srv, log)
