"""Gemma 4 E2B with a long document in context, on the RTX 3050 OEM: what it finds, how long it waits.

    python3 longctx.py OUT_DIR ARM [ARM ...]      (resumable: skips trials already in longctx.jsonl)

Each prompt is the first DEPTH tokens of the NVIDIA driver README (doc.sha256) with facts spliced in at
line breaks, then one question. Depths 16,384 / 32,768 / 64,000 / 98,304 / 126,976, as the 12B's recall test
(../../../../local-llamacpp-3050oem-12b-q4_0/benchmarks/runs/2026-10-17-kvcache-12b-3050oem), 10 seeded
trials per depth, identical across arms. Two tasks:
  recall:  8 sentences "The access code for project <P> is <6 digits>."; asked for all 8 codes.
  link:    8 sentences "Project <P> is led by <person>." and 8 "<person> works from the <city> office.",
           16 positions shuffled independently; asked for the city of each project's lead (two facts apart).
Scored by code: exact codes / cities out of 8 per trial. Prompt cache off on every request, so each one
reads its whole prompt and timings.prompt_ms is the time to read DEPTH tokens.
Arms (all the rig's serving flags, -c 131072, per-layer table on the card, one request at a time):
  f16-draft, q8-draft, q4-draft      greedy, thinking off, drafter at 2 tokens (the rig serves f16-draft)
  f16-sampled-off, f16-sampled-on    Google's generation_config (temperature 1.0, top_k 64, top_p 0.95, seed
                                     = trial), no drafter; thinking on gets max_tokens 16,384, depths to 98,304
"""
import json, os, random, re, signal, subprocess, sys, time, urllib.request

OUT, ARMS = sys.argv[1], sys.argv[2:]
RIG = "/home/xbill/gemma4-dev/local-llamacpp-3050oem-2b-q4_0"
PORT, URL = 8080, "http://127.0.0.1:8080"
env = {}
for line in open(os.path.join(RIG, "tpu.env")):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        env[k] = v.strip('"')
DOC = open("/usr/share/doc/NVIDIA_GLX-1.0/README.txt").read()
DEPTHS = [16384, 32768, 64000, 98304, 126976]
PROJECTS = ["Alder", "Basalt", "Cobalt", "Dune", "Ember", "Fjord", "Garnet", "Heron", "Indigo", "Juniper",
            "Kestrel", "Lumen", "Marlin", "Nimbus", "Onyx", "Pebble", "Quartz", "Raven", "Sorrel", "Tundra"]
PEOPLE = ["Abena Osei", "Bruno Keller", "Chiara Moretti", "Dmitri Volkov", "Elif Yilmaz", "Farah Haddad",
          "Goran Petrovic", "Hana Sato", "Ines Duarte", "Jonas Lindqvist", "Kavya Rao", "Liam Byrne",
          "Mei Chen", "Nils Andersen", "Olga Nowak", "Pablo Ruiz"]
CITIES = ["Lisbon", "Oslo", "Nairobi", "Osaka", "Quito", "Tallinn", "Hanoi", "Perth", "Tunis", "Calgary",
          "Seville", "Gdansk", "Cusco", "Bergen", "Dakar", "Adelaide"]
TRIALS, N = int(os.environ.get("LONGCTX_TRIALS", 10)), 8
DEPTHS = json.loads(os.environ.get("LONGCTX_DEPTHS", "null")) or DEPTHS
ARMDEF = {  # ctk, drafter, sampled, thinking
    "f16-draft": ("f16", True, False, False), "q8-draft": ("q8_0", True, False, False),
    "q4-draft": ("q4_0", True, False, False),
    "f16-sampled-off": ("f16", False, True, False), "f16-sampled-on": ("f16", False, True, True)}


def post(path, body, timeout=7200):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def gpu():
    return subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,memory.used,power.draw",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()


def start(arm):
    ctk, draft, _, _ = ARMDEF[arm]
    subprocess.run(["pkill", "-x", "llama-server"])
    while subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True).returncode == 0:
        time.sleep(1)
    while int(gpu().split(",")[0]) > 45:
        time.sleep(5)
    cmd = [env["LLAMA_SERVER_BIN"], "-m", env["MODEL_PATH"], "--host", "127.0.0.1", "--port", str(PORT),
           "-ngl", "99", "-c", "131072", "-ctk", ctk, "-ctv", ctk, "-fa", "1", "-t", "6", "-tb", "12",
           "--parallel", "1", "--reasoning", "off", "--metrics", "-ot", env["OVERRIDE_TENSORS"]]
    if draft:
        cmd += ["-md", env["SPEC_DRAFT_MODEL"], "--spec-type", "draft-mtp", "-ngld", "all", "--spec-draft-n-max", "2"]
    log = open(os.path.join(OUT, f"server-{arm}.log"), "w")
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(600):
        if srv.poll() is not None:
            raise RuntimeError(f"server exited, see server-{arm}.log")
        try:
            urllib.request.urlopen(URL + "/health", timeout=2)
            mem = subprocess.run(["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader"],
                                 capture_output=True, text=True).stdout.strip()
            print(f"{time.strftime('%F %T')} server up {arm}: {mem}", flush=True)
            return srv, log
        except Exception:
            time.sleep(1)
    raise RuntimeError("server timeout")


def splice(text, sentences, rng):
    """Insert sentences at evenly spread random line breaks, in the order given."""
    k = len(sentences)
    cuts = sorted(rng.sample(range(1, k * 4), k))
    parts, prev = [], 0
    for i, s in enumerate(sentences):
        pos = len(text) * cuts[i] // (k * 4)
        nl = text.rfind("\n", prev, pos)
        pos = nl + 1 if nl > prev else pos
        parts += [text[prev:pos], s + "\n"]
        prev = pos
    return "".join(parts) + text[prev:]


def build(task, text, depth, trial):
    rng = random.Random(f"{task}-{depth}-{trial}")
    names = rng.sample(PROJECTS, N)
    if task == "recall":
        truth = {n: f"{rng.randrange(100000, 1000000)}" for n in names}
        sents = [f"The access code for project {n} is {truth[n]}." for n in names]
        q = ("List the access code of every project mentioned in the document above, as a JSON object "
             "mapping project name to code. Output only the JSON.")
    else:
        people, cities = rng.sample(PEOPLE, N), rng.sample(CITIES, N)
        truth = dict(zip(names, cities))
        sents = [f"Project {n} is led by {p}." for n, p in zip(names, people)] + \
                [f"{p} works from the {c} office." for p, c in zip(people, cities)]
        rng.shuffle(sents)
        q = ("Every project mentioned in the document above has a lead, and every lead works from one office. "
             "For each project, give the city of its lead's office, as a JSON object mapping project name to "
             "city. Output only the JSON.")
    return "Here is a document:\n\n" + splice(text, sents, rng) + "\n\n" + q, truth


def score(task, reply, truth):
    pat = (r'"?(?:[Pp]roject )?([A-Z][a-z]+)"?\s*:\s*"?(\d{6})' if task == "recall"
           else r'"(?:[Pp]roject )?([A-Z][a-z]+)"\s*:\s*"(?:the )?([A-Z][a-z]+)')
    got = dict(re.findall(pat, reply))
    return sum(got.get(n) == v for n, v in truth.items())


path = os.path.join(OUT, "longctx.jsonl")
done = set()
if os.path.exists(path):
    for line in open(path):
        r = json.loads(line)
        done.add((r["arm"], r["task"], r["depth"], r["trial"]))
out = open(path, "a")
for arm in ARMS:
    ctk, draft, sampled, think = ARMDEF[arm]
    depths = [d for d in DEPTHS if not think or d <= 98304]
    todo = [(t, d, i) for d in depths for t in ("recall", "link") for i in range(TRIALS) if (arm, t, d, i) not in done]
    if not todo:
        print(f"{arm}: done", flush=True); continue
    srv, log = start(arm)
    ids = post("/tokenize", {"content": DOC})["tokens"]
    texts = {}
    for task, depth, trial in todo:
        if depth not in texts:
            texts[depth] = post("/detokenize", {"tokens": ids[:depth]})["content"]
        content, truth = build(task, texts[depth], depth, trial)
        body = {"messages": [{"role": "user", "content": content}], "cache_prompt": False,
                "max_tokens": 16384 if think else 256, "chat_template_kwargs": {"enable_thinking": think}}
        body.update({"temperature": 1.0, "top_k": 64, "top_p": 0.95, "seed": trial} if sampled else {"temperature": 0})
        t0 = time.time()
        d = post("/v1/chat/completions", body)
        msg = d["choices"][0]["message"]
        reply = msg.get("content") or ""
        tm = d["timings"]
        rec = {"arm": arm, "task": task, "depth": depth, "trial": trial, "right": score(task, reply, truth), "of": N,
               "prompt_n": d["usage"]["prompt_tokens"], "completion_n": d["usage"]["completion_tokens"],
               "finish": d["choices"][0]["finish_reason"], "prompt_ms": tm["prompt_ms"],
               "prompt_tps": tm["prompt_per_second"], "decode_tps": tm["predicted_per_second"],
               "draft_n": tm.get("draft_n"), "draft_accepted": tm.get("draft_n_accepted"),
               "wall_s": round(time.time() - t0, 2), "reply": reply,
               "reasoning_chars": len(msg.get("reasoning_content") or ""), "gpu": gpu()}
        out.write(json.dumps(rec) + "\n"); out.flush()
        print(f"{time.strftime('%T')} {arm} {task} {depth} t{trial}: {rec['right']}/8 read {tm['prompt_ms']/1000:.1f}s "
              f"{rec['completion_n']} tok", flush=True)
    srv.send_signal(signal.SIGTERM); srv.wait(); log.close()
