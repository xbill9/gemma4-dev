"""Does the drafter keep paying with a long conversation in context, on the T4? The vLLM
counterpart of local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-drafter-gaps-3050oem/longctx.py,
capped by the rigs' --max-model-len 16384.

    python3 longctx.py RIG_DIR OUT_DIR ARM[,ARM...]

Context: the first D tokens of five licence texts from /usr/share/common-licenses (GPL-3, GPL-2,
Apache-2.0, MPL-2.0, AGPL-3.0, concatenated; SHA-256 in doc.sha256), D = 0, 4096, 8192, 15000,
then a story request and a code request unrelated to it and a request to summarize it. Greedy,
256 tokens, one request at a time, vLLM's prefix cache on (its default), so each depth's text is
read once per arm; decode tok/s is measured from the streamed tokens and excludes reading the prompt.
"""
import hashlib, json, os, sys, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t4serve import load_env, parse_arm, start, stop, spec_counters, stream_chat, gpu

RIG, OUT, ARMLIST = sys.argv[1:4]
env = load_env(RIG); URL = env["ENDPOINT"]; MODEL = env["MODEL_NAME"]
FILES = ["GPL-3", "GPL-2", "Apache-2.0", "MPL-2.0", "AGPL-3.0"]
DOC = "\n\n".join(open(f"/usr/share/common-licenses/{f}").read() for f in FILES)
DEPTHS = [0, 4096, 8192, 15000]
QUESTIONS = {
    "story": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "code": "Write a Python function that checks whether a number is prime, with a docstring and two doctests.",
    "summary": "Summarize the document above in one paragraph.",
}

def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(dict(body, model=MODEL)).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))

os.makedirs(OUT, exist_ok=True)
open(os.path.join(OUT, "doc.sha256"), "w").write(hashlib.sha256(DOC.encode()).hexdigest() + "  " + "+".join(FILES) + "\n")
out = open(os.path.join(OUT, "results.jsonl"), "a")
texts = {}
for arm in ARMLIST.split(","):
    draft, n = parse_arm(env, arm)
    srv, log = start(env, os.path.join(OUT, f"server-{arm}.log"), draft, n)
    try:
        if not texts:
            ids = post("/tokenize", {"prompt": DOC, "add_special_tokens": False})["tokens"]
            assert len(ids) >= DEPTHS[-1], len(ids)
            for depth in DEPTHS:
                texts[depth] = post("/detokenize", {"tokens": ids[:depth]})["prompt"] if depth else ""
        stream_chat(URL, MODEL, {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 16})
        for depth in DEPTHS:
            for q, question in QUESTIONS.items():
                if depth == 0 and q == "summary":
                    continue
                content = f"Here is a document:\n\n{texts[depth]}\n\n{question}" if depth else question
                d0, a0 = spec_counters(URL)
                text, ntok, ttft, dec, e2e = stream_chat(URL, MODEL, {
                    "messages": [{"role": "user", "content": content}], "max_tokens": 256, "temperature": 0})
                d1, a1 = spec_counters(URL)
                rec = {"arm": arm, "depth": depth, "question": q, "n": ntok, "ttft_s": ttft, "decode_tps": dec,
                       "draft_n": d1 - d0, "draft_accepted": a1 - a0,
                       "sha": hashlib.sha256(text.encode()).hexdigest()[:16], "gpu": gpu()}
                out.write(json.dumps(rec) + "\n"); out.flush()
                print(arm, depth, q, ntok, round(dec or 0, 1), int(a1 - a0), "/", int(d1 - d0), rec["gpu"], flush=True)
    finally:
        stop(srv, log)
