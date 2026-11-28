"""Gemma 4 12B as an agent on the RTX 3050 OEM: tool calls over a workspace, one long session at a time.

    python3 agent.py OUT_DIR ARM [ARM ...]      (resumable: skips sessions already in tasks.jsonl)

Each session builds a seeded workspace of plain-text files: a deploy manifest naming 12 services with their
config file and owner, 12 config files (each with a listen_port line), a team roster with pager numbers, and
one log per service with a single ERROR line. Padding lines come from the NVIDIA driver README (doc.sha256),
so every file reads like real text and reading one adds about a thousand tokens to the context.
Tools (OpenAI `tools`, tool_choice auto): list_files(dir), read_file(path, start_line) -> up to 60 lines,
search(text) -> up to 15 matching lines with path and line number, submit_answer(answer).
Ten tasks per session, four kinds, each about a different service:
  port   "What port does service S listen on?"                       manifest -> config
  pager  "What is the pager number of the owner of service S?"      manifest -> roster
  error  "What error code is in the log of service S?"               log
  owner  "What port does the service owned by P listen on?"         manifest -> config
Scored by code (score()): the digits of the submit_answer argument, or the last number in a final reply with no\ntool call, against the truth. 50 sessions (500 tasks) per arm, the same workspaces and questions in every arm.
Arms:
  served, served-fresh          temperature 0, thinking off, drafter at 2 tokens: the rig as it serves
  nodraft                       temperature 0, thinking off, no drafter
  sampled-off                   Google's generation_config (temperature 1.0, top_k 64, top_p 0.95), thinking off
  sampled-on, sampled-on-fresh  Google's settings, thinking on
"-fresh" arms ask each task in a new conversation (system prompt + one task) on the same workspace; the rest
ask all ten tasks in one conversation, so tool results pile up as in a coding-assistant session. Prompt cache
on (the server default), one request at a time. Context size, cache type and tensor override come from the
rig's tpu.env (CONTEXT_SIZE, KV_CACHE_TYPE, OVERRIDE_TENSORS). Forked from the 2026-11-26 E2B run; workspaces,
questions, seeds and scoring are identical. At most 12 tool calls per task; a session stops early past
CONTEXT_SIZE - 11,072 prompt tokens (120,000 at 131,072, the E2B run's margin).
"""
import json, os, random, re, signal, subprocess, sys, time, urllib.request

OUT, ARMS = sys.argv[1], sys.argv[2:]
RIG = "/home/xbill/gemma4-dev/local-llamacpp-3050oem-12b-q4_0"
URL = "http://127.0.0.1:8080"
SESSIONS = int(os.environ.get("AGENT_SESSIONS", 50)); TASKS = 10; MAX_CALLS = 12
env = {}
for line in open(os.path.join(RIG, "tpu.env")):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1); env[k] = v.strip('"')
CTX = int(env["CONTEXT_SIZE"]); CTX_STOP = CTX - 11072; KV = env["KV_CACHE_TYPE"]
PAD = [l.rstrip() for l in open("/usr/share/doc/NVIDIA_GLX-1.0/README.txt") if len(l.strip()) > 30]
SERVICES = ["billing", "ledger", "gateway", "search", "inventory", "checkout", "notify", "auth", "reports",
            "scheduler", "uploads", "metrics", "catalog", "payroll", "shipping", "pricing", "profiles", "audit",
            "mailer", "exports", "webhooks", "sessions", "quotas", "backups"]
PEOPLE = ["Abena Osei", "Bruno Keller", "Chiara Moretti", "Dmitri Volkov", "Elif Yilmaz", "Farah Haddad",
          "Goran Petrovic", "Hana Sato", "Ines Duarte", "Jonas Lindqvist", "Kavya Rao", "Liam Byrne",
          "Mei Chen", "Nils Andersen", "Olga Nowak", "Pablo Ruiz", "Rhea Kapoor", "Sven Holm"]
TEAMS = ["Atlas", "Beacon", "Comet", "Delta", "Echo", "Falcon"]
ARMDEF = {  # drafter, sampled, thinking, fresh
    "served": (True, False, False, False), "served-fresh": (True, False, False, True),
    "nodraft": (False, False, False, False), "sampled-off": (False, True, False, False),
    "sampled-on": (False, True, True, False), "sampled-on-fresh": (False, True, True, True)}
SYSTEM = ("You are an operations assistant. You can only learn about the systems through the tools, which read a "
          "workspace of plain-text files. Use the tools to find each answer, then call submit_answer with just the "
          "answer. Do not guess.")
TOOLS = [
    {"type": "function", "function": {"name": "list_files", "description": "List the files in a directory of the workspace.",
     "parameters": {"type": "object", "properties": {"dir": {"type": "string", "description": "Directory, e.g. '.' or 'configs'"}}, "required": ["dir"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read up to 60 lines of a file, starting at start_line (1-based).",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer", "description": "First line to read, default 1"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "search", "description": "Find lines containing the text in any file. Returns up to 15 matches as path:line: text.",
     "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "submit_answer", "description": "Submit the final answer to the current question.",
     "parameters": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}}},
]


def workspace(seed):
    rng = random.Random(f"ws-{seed}")
    pad = lambda n: [rng.choice(PAD) for _ in range(n)]
    svcs = rng.sample(SERVICES, 12); owners = rng.sample(PEOPLE, 12)
    files, truth = {}, {}
    man = ["# deploy manifest: one entry per service"] + pad(25)
    used = set()
    for s, o in zip(svcs, owners):
        cfg = f"configs/cfg-{rng.randrange(16**4):04x}.conf"
        port = rng.randrange(2000, 65000); used.add(port)
        lines = [f"# config {cfg}"] + pad(rng.randrange(60, 140))
        lines.insert(rng.randrange(5, len(lines)), f"listen_port = {port}")
        lines.insert(rng.randrange(5, len(lines)), f"max_connections = {rng.randrange(16, 4096)}")
        files[cfg] = lines
        man += [f"service={s} config={cfg} owner={o}"] + pad(rng.randrange(8, 20))
        err = f"E{rng.randrange(1000, 10000)}"
        log = [f"2026-11-{rng.randrange(1, 28):02d} {rng.randrange(24):02d}:{rng.randrange(60):02d} INFO {l}" for l in pad(rng.randrange(90, 180))]
        log.insert(rng.randrange(10, len(log)), f"2026-11-{rng.randrange(1, 28):02d} {rng.randrange(24):02d}:{rng.randrange(60):02d} ERROR code {err} in {s}")
        files[f"logs/{s}.log"] = log
        truth[s] = {"port": str(port), "owner": o, "error": err}
    files["deploy/manifest.txt"] = man
    roster = ["# on-call roster"] + pad(20)
    pagers = {}
    for o in rng.sample(PEOPLE, len(PEOPLE)):
        pagers[o] = str(rng.randrange(1000, 10000))
        roster += [f"{o}: team {rng.choice(TEAMS)}, pager {pagers[o]}"] + pad(rng.randrange(4, 10))
    files["people/roster.txt"] = roster
    for i in range(6):
        files[f"docs/notes-{i}.txt"] = pad(rng.randrange(80, 160))
    tasks = []
    kinds = ["port", "pager", "error", "owner"] * 3
    for k, s in zip(kinds[:TASKS], rng.sample(svcs, TASKS)):
        t = truth[s]
        if k == "port": tasks.append((k, f"What port does service {s} listen on?", t["port"]))
        elif k == "pager": tasks.append((k, f"What is the pager number of the owner of service {s}?", pagers[t["owner"]]))
        elif k == "error": tasks.append((k, f"What error code is in the log of service {s}?", t["error"]))
        else: tasks.append((k, f"What port does the service owned by {t['owner']} listen on?", t["port"]))
    return files, tasks


def run_tool(files, name, args):
    try:
        if name == "list_files":
            d = (args.get("dir") or ".").strip("/").strip(".").strip("/")
            names = sorted(p for p in files if not d or p.startswith(d + "/"))
            if not names: return f"No such directory: {args.get('dir')}"
            if not d: return "\n".join(sorted({p.split('/')[0] + "/" for p in files}))
            return "\n".join(names)
        if name == "read_file":
            p = str(args.get("path", "")).lstrip("./")
            if p not in files: return f"No such file: {args.get('path')}"
            s = max(1, int(args.get("start_line") or 1)); L = files[p]
            body = "\n".join(f"{i}: {L[i-1]}" for i in range(s, min(len(L), s + 59) + 1))
            more = f"\n[lines {s}-{min(len(L), s + 59)} of {len(L)}" + (f"; call read_file with start_line={s + 60} for more]" if s + 60 <= len(L) else "]")
            return body + more
        if name == "search":
            q = str(args.get("text", ""))
            if not q: return "Empty search."
            hits = [f"{p}:{i+1}: {l}" for p in sorted(files) for i, l in enumerate(files[p]) if q.lower() in l.lower()]
            return "\n".join(hits[:15]) + (f"\n[{len(hits)} matches, first 15 shown]" if len(hits) > 15 else "") if hits else "No matches."
        return f"Unknown tool: {name}"
    except Exception as e:
        return f"Tool error: {e}"


def digits(s): return re.sub(r"\D", "", str(s))


def score(answer, via, truth):
    """Tool answer: its digits equal the truth's. Prose answer: the last number in it (E-prefix allowed) is the truth."""
    if answer is None: return False
    if via == "tool": return digits(answer) == digits(truth) != ""
    nums = re.findall(r"E?\d{3,6}", answer)
    return bool(nums) and digits(nums[-1]) == digits(truth)


def post(body, timeout=3600):
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def gpu():
    return subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,memory.used,power.draw",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()


def start(arm):
    draft = ARMDEF[arm][0]
    subprocess.run(["pkill", "-x", "llama-server"])
    while subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True).returncode == 0: time.sleep(1)
    while int(gpu().split(",")[0]) > 45: time.sleep(5)
    cmd = [env["LLAMA_SERVER_BIN"], "-m", env["MODEL_PATH"], "--host", "127.0.0.1", "--port", "8080", "-ngl", "99",
           "-c", str(CTX), "-ctk", KV, "-ctv", KV, "-fa", "1", "-t", "6", "-tb", "12", "--parallel", "1",
           "--reasoning", "off", "--metrics"]
    if env.get("OVERRIDE_TENSORS"): cmd += ["-ot", env["OVERRIDE_TENSORS"]]
    if draft: cmd += ["-md", env["SPEC_DRAFT_MODEL"], "--spec-type", "draft-mtp", "-ngld", "all", "--spec-draft-n-max", "2"]
    log = open(os.path.join(OUT, f"server-{arm}.log"), "w")
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(600):
        if srv.poll() is not None: raise RuntimeError(f"server exited, see server-{arm}.log")
        try:
            urllib.request.urlopen(URL + "/health", timeout=2)
            mem = subprocess.run(["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
            print(f"{time.strftime('%F %T')} server up {arm}: {mem}", flush=True); return srv, log
        except Exception: time.sleep(1)
    raise RuntimeError("server timeout")


def session(arm, seed, treq, ttask):
    _, sampled, think, fresh = ARMDEF[arm]
    files, tasks = workspace(seed)
    msgs = [{"role": "system", "content": SYSTEM}]
    stopped = False
    for ti, (kind, q, truth) in enumerate(tasks):
        if fresh: msgs = [{"role": "system", "content": SYSTEM}]
        msgs.append({"role": "user", "content": q})
        calls = bad = 0; answer = None; via = None; t0 = time.time(); last_prompt = 0; tools_used = []
        while answer is None and calls < MAX_CALLS and not stopped:
            body = {"messages": msgs, "tools": TOOLS, "tool_choice": "auto", "cache_prompt": True,
                    "max_tokens": 8192 if think else 1024, "chat_template_kwargs": {"enable_thinking": think}}
            body.update({"temperature": 1.0, "top_k": 64, "top_p": 0.95, "seed": seed * 100 + ti} if sampled else {"temperature": 0})
            r0 = time.time(); d = post(body); wall = time.time() - r0
            ch = d["choices"][0]; m = ch["message"]; tm = d["timings"]; last_prompt = d["usage"]["prompt_tokens"]
            tcs = m.get("tool_calls") or []
            treq.write(json.dumps({"arm": arm, "session": seed, "task": ti, "kind": kind, "prompt_n": last_prompt,
                "new_prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n"), "prompt_ms": tm.get("prompt_ms"),
                "completion_n": d["usage"]["completion_tokens"], "decode_tps": tm.get("predicted_per_second"),
                "draft_n": tm.get("draft_n"), "draft_accepted": tm.get("draft_n_accepted"), "finish": ch["finish_reason"],
                "n_tool_calls": len(tcs), "thinking_chars": len(m.get("reasoning_content") or ""), "wall_s": round(wall, 3),
                "gpu": gpu()}) + "\n"); treq.flush()
            asst = {"role": "assistant", "content": m.get("content") or ""}
            if tcs: asst["tool_calls"] = tcs
            msgs.append(asst)
            if not tcs:
                answer, via = m.get("content") or "", "prose"; break
            for tc in tcs:
                calls += 1; name = tc["function"]["name"]
                try: args = json.loads(tc["function"].get("arguments") or "{}")
                except Exception: args = None
                tools_used.append([name, args])
                if name == "submit_answer" and isinstance(args, dict) and answer is None:
                    answer, via = str(args.get("answer", "")), "tool"; result = "Answer recorded."
                elif args is None or not isinstance(args, dict):
                    bad += 1; result = "Tool error: arguments are not valid JSON."
                else:
                    result = run_tool(files, name, args)
                    bad += result.startswith(("Unknown tool", "No such", "Tool error", "Empty search"))
                msgs.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": result})
            if last_prompt > CTX_STOP: stopped = True
        ok = score(answer, via, truth)
        ttask.write(json.dumps({"arm": arm, "session": seed, "task": ti, "kind": kind, "question": q, "truth": truth,
            "answer": answer, "via": via, "right": ok, "tool_calls": calls, "bad_calls": bad, "tools": tools_used,
            "prompt_n_end": last_prompt, "wall_s": round(time.time() - t0, 2), "stopped": stopped}) + "\n"); ttask.flush()
        print(f"{time.strftime('%T')} {arm} s{seed} t{ti} {kind}: {'ok ' if ok else 'BAD'} {answer!r:.40} calls {calls} ctx {last_prompt}", flush=True)
        if stopped: break


tpath = os.path.join(OUT, "tasks.jsonl")
done = set()
if os.path.exists(tpath):
    rows = [json.loads(l) for l in open(tpath)]
    per = {}
    for r in rows: per[(r["arm"], r["session"])] = per.get((r["arm"], r["session"]), 0) + 1
    done = {k for k, n in per.items() if n == TASKS or any(r["stopped"] for r in rows if (r["arm"], r["session"]) == k)}
treq, ttask = open(os.path.join(OUT, "requests.jsonl"), "a"), open(tpath, "a")
for arm in ARMS:
    todo = [s for s in range(SESSIONS) if (arm, s) not in done]
    if not todo: print(f"{arm}: done", flush=True); continue
    srv, log = start(arm)
    for s in todo: session(arm, s, treq, ttask)
    srv.send_signal(signal.SIGTERM); srv.wait(); log.close()
