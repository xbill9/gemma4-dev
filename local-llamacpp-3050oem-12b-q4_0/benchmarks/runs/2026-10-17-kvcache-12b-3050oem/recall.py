"""Long-context recall by KV cache type: does a smaller cache lose facts deep in a conversation?

    python3 recall.py RIG_DIR OUT_DIR ARM:CTK:CTX:draft|none:DEPTH,DEPTH,... [...]

GSM8K and BFCL prompts are short, so they cannot show what the cache costs with a long
conversation held. Here each prompt is the first DEPTH tokens of the NVIDIA driver README
(doc.sha256) with 8 sentences "The access code for project <Name> is <6 digits>." inserted at
evenly spread random positions, followed by a request for all 8 codes as JSON. 10 trials per
depth, seeded, identical across arms. Greedy, max_tokens 256, prompt cache on. Score: codes
returned exactly, out of 80 per depth. One llama-server per arm on port 8099, the rig's serving
flags otherwise. Writes recall.jsonl.
"""
import json, os, random, re, signal, subprocess, sys, time, urllib.request

RIG, OUT, SPECS = sys.argv[1], sys.argv[2], sys.argv[3:]
PORT = 8099
URL = f"http://127.0.0.1:{PORT}"
env = {}
for line in open(os.path.join(RIG, "tpu.env")):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        env[k] = v.strip('"')
DOC = open("/usr/share/doc/NVIDIA_GLX-1.0/README.txt").read()
NAMES = ["Alder", "Basalt", "Cobalt", "Dune", "Ember", "Fjord", "Garnet", "Heron", "Indigo", "Juniper",
         "Kestrel", "Lumen", "Marlin", "Nimbus", "Onyx", "Pebble", "Quartz", "Raven", "Sorrel", "Tundra"]
TRIALS, NEEDLES = 10, 8


def post(path, body, timeout=3600):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def gpu():
    return subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,memory.used",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()


def start(ctk, ctx, draft, log_path):
    cmd = [env["LLAMA_SERVER_BIN"], "-m", env["MODEL_PATH"], "--host", "127.0.0.1", "--port", str(PORT),
           "-ngl", "99", "-c", str(ctx), "-ctk", ctk, "-ctv", ctk, "-fa", "1", "-t", "6", "-tb", "12",
           "--parallel", "1", "--reasoning", "off"]
    if draft:
        cmd += ["-md", env["SPEC_DRAFT_MODEL"], "--spec-type", "draft-mtp", "-ngld", "all", "--spec-draft-n-max", "2"]
    while int(gpu().split(",")[0]) > 40:
        time.sleep(5)
    log = open(log_path, "w")
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(300):
        if srv.poll() is not None:
            raise RuntimeError(f"server exited, see {log_path}")
        try:
            urllib.request.urlopen(URL + "/health", timeout=2)
            return srv, log
        except Exception:
            time.sleep(1)
    raise RuntimeError("server timeout")


def build(ids, depth, trial):
    """Document of `depth` tokens with NEEDLES facts spliced in at sentence-ish boundaries."""
    rng = random.Random(f"{depth}-{trial}")
    names = rng.sample(NAMES, NEEDLES)
    codes = {n: f"{rng.randrange(100000, 1000000)}" for n in names}
    text = post("/detokenize", {"tokens": ids[:depth]})["content"]
    cuts = sorted(rng.sample(range(1, NEEDLES * 4), NEEDLES))
    parts, prev = [], 0
    for i, n in enumerate(names):
        pos = len(text) * cuts[i] // (NEEDLES * 4)
        nl = text.rfind("\n", prev, pos)
        pos = nl + 1 if nl > prev else pos
        parts.append(text[prev:pos])
        parts.append(f"The access code for project {n} is {codes[n]}.\n")
        prev = pos
    parts.append(text[prev:])
    q = ("List the access code of every project mentioned in the document above, as a JSON object "
         "mapping project name to code. Output only the JSON.")
    return "Here is a document:\n\n" + "".join(parts) + "\n\n" + q, codes


out = open(os.path.join(OUT, "recall.jsonl"), "a")
ids = None
for spec in SPECS:
    arm, ctk, ctx, dd, depths = spec.split(":")
    srv, log = start(ctk, int(ctx), dd == "draft", os.path.join(OUT, f"server-recall-{arm}.log"))
    if ids is None:
        ids = post("/tokenize", {"content": DOC})["tokens"]
    for depth in map(int, depths.split(",")):
        right = 0
        for trial in range(TRIALS):
            content, codes = build(ids, depth, trial)
            d = post("/v1/chat/completions", {"messages": [{"role": "user", "content": content}],
                                              "max_tokens": 256, "temperature": 0, "cache_prompt": True})
            text = d["choices"][0]["message"]["content"]
            got = dict(re.findall(r'"?([A-Z][a-z]+)"?\s*:\s*"?(\d{6})', text))
            ok = sum(got.get(n) == c for n, c in codes.items())
            right += ok
            rec = {"arm": arm, "ctk": ctk, "ctx": int(ctx), "draft": dd == "draft", "depth": depth, "trial": trial,
                   "prompt_n": d["usage"]["prompt_tokens"], "right": ok, "of": NEEDLES,
                   "decode_tps": d["timings"]["predicted_per_second"], "text": text, "gpu": gpu()}
            out.write(json.dumps(rec) + "\n"); out.flush()
        print(arm, depth, f"{right}/{TRIALS * NEEDLES}", flush=True)
    srv.send_signal(signal.SIGTERM); srv.wait(); log.close()
