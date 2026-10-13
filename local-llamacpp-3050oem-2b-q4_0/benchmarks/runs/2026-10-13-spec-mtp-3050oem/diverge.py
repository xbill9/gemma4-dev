"""Where do greedy replies diverge? Captures full text from the no-draft arm and the
draft-mtp n=2 arm (two runs each, to check each arm is self-consistent), same prompts as
spec_ab.py. Writes texts.json. Run from the rig root."""
import json, os, signal, subprocess, time, urllib.request
import importlib.util
D = "benchmarks/runs/2026-10-13-spec-mtp-3050oem"
spec = importlib.util.spec_from_file_location("ab", os.path.join(D, "spec_ab.py"))
src = open(os.path.join(D, "spec_ab.py")).read()
PROMPTS = eval(src.split("PROMPTS = ", 1)[1].split("\n}\n", 1)[0] + "\n}")
env = dict(l.strip().split("=", 1) for l in open("tpu.env") if l.strip() and not l.startswith("#") and "=" in l)
DRAFT = "/home/xbill/models/gemma-4-E2B-it-qat-assistant/gemma-4-E2B-it-qat-assistant-q8_0.gguf"
texts = {}
for arm, extra in (("none", []), ("mtp-n2", ["-md", DRAFT, "--spec-type", "draft-mtp", "-ngld", "all", "--spec-draft-n-max", "2"])):
    cmd = [env["LLAMA_SERVER_BIN"], "-m", env["MODEL_PATH"], "--host", "127.0.0.1", "--port", "8099", "-ngl", "99",
           "-c", "8192", "-fa", "1", "-t", "6", "-tb", "12", "--parallel", "1", "--reasoning", "off",
           "-ot", env["OVERRIDE_TENSORS"]] + extra
    srv = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try: urllib.request.urlopen("http://127.0.0.1:8099/health", timeout=2); break
        except Exception: time.sleep(1)
    for name, p in PROMPTS.items():
        for r in range(2):
            body = {"messages": [{"role": "user", "content": p}], "max_tokens": 256, "temperature": 0, "cache_prompt": False}
            req = urllib.request.Request("http://127.0.0.1:8099/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
            texts[f"{arm}|{name}|{r}"] = json.load(urllib.request.urlopen(req))["choices"][0]["message"]["content"]
    srv.send_signal(signal.SIGTERM); srv.wait()
json.dump(texts, open(os.path.join(D, "texts.json"), "w"), indent=1)
for name in PROMPTS:
    a, b = texts[f"none|{name}|0"], texts[f"mtp-n2|{name}|0"]
    i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
    print(f"{name}: none self-same={texts[f'none|{name}|0']==texts[f'none|{name}|1']} "
          f"mtp self-same={texts[f'mtp-n2|{name}|0']==texts[f'mtp-n2|{name}|1']} "
          f"first diff at char {i} of {len(a)}/{len(b)}")
    print("   none: ..." + repr(a[max(0,i-60):i+60]))
    print("   mtp : ..." + repr(b[max(0,i-60):i+60]))
