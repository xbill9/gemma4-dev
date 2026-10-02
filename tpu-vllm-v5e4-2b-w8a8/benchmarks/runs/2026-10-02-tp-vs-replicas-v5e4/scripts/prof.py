"""On the VM: profile one decode-heavy burst. prof.py <port> <container> <label>
16 parallel chats of 128 output tokens (ignore_eos, greedy) between /start_profile and /stop_profile."""
import json, sys, time, urllib.request, subprocess
from concurrent.futures import ThreadPoolExecutor
port, cont, label = sys.argv[1:]
base = f"http://localhost:{port}"
model = json.load(urllib.request.urlopen(base + "/v1/models"))["data"][0]["id"]
def chat(i):
    b = {"model": model, "messages": [{"role": "user", "content": f"Count upward from {i}."}], "max_tokens": 128, "temperature": 0, "ignore_eos": True}
    r = urllib.request.Request(base + "/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r))
with ThreadPoolExecutor(16) as ex:
    list(ex.map(chat, range(16)))  # warm
    urllib.request.urlopen(urllib.request.Request(base + "/start_profile", b"", method="POST"))
    t0 = time.time(); list(ex.map(chat, range(16))); wall = time.time() - t0
    urllib.request.urlopen(urllib.request.Request(base + "/stop_profile", b"", method="POST"))
print(json.dumps({"label": label, "wall_s": round(wall, 3), "tokens": 16 * 128}))
time.sleep(60)  # the trace is written after stop_profile returns
subprocess.run(["sudo", "rm", "-rf", f"/tmp/prof-{label}"]); subprocess.run(["sudo", "docker", "cp", f"{cont}:/tmp/prof", f"/tmp/prof-{label}"])
subprocess.run(["sudo", "docker", "exec", cont, "rm", "-rf", "/tmp/prof"])
