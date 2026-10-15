"""Two greedy requests at once: streaming on/off x prompt pair, tok/s and graphs-reused delta."""
import concurrent.futures, json, re, sys, time, urllib.request
sys.path.insert(0, ".")
BASE, LOG = sys.argv[1], sys.argv[2]
PROMPTS = [
    "Write a Python function that checks whether a number is prime, with a docstring and two doctests.",
    "I just moved to a new city and feel a bit lonely. Any advice?",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain how a refrigerator works to a ten-year-old.",
    "List the planets of the solar system with one fact about each.",
    "Summarize the causes of the French Revolution in one paragraph.",
]
def one(p, stream):
    body = {"messages": [{"role": "user", "content": p}], "max_tokens": 256, "cache_prompt": False, "temperature": 0}
    if stream: body.update(stream=True, stream_options={"include_usage": True})
    t0 = time.perf_counter()
    with urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                {"Content-Type": "application/json"}), timeout=900) as fh:
        if not stream:
            return json.load(fh)["timings"]["predicted_n"]
        n = 0
        for raw in fh:
            l = raw.decode().strip()
            if l.startswith("data:") and l != "data: [DONE]":
                t = json.loads(l[5:]).get("timings")
                if t: n = t["predicted_n"]
        return n
def graphs():
    g = re.findall(r"graphs reused =\s+(\d+)", open(LOG).read()); return int(g[-1]) if g else 0
for pair in ((2, 3), (0, 1), (4, 5), (1, 2)):
    for stream in (False, True):
        g0 = graphs(); t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(2) as ex:
            ns = list(ex.map(lambda i: one(PROMPTS[i], stream), pair))
        dt = time.perf_counter() - t0; time.sleep(0.3)
        print(f"prompts {pair} stream={stream!s:5} {sum(ns)/dt:6.1f} tok/s  tokens {ns}  graphs reused +{graphs()-g0}", flush=True)
