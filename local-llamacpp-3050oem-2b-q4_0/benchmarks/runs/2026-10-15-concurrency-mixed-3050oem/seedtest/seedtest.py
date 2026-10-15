"""Two requests at once, temperature 0.8, varying only the seed field: does the seed choice
decide whether llama-server keeps reusing CUDA graphs? Prints aggregate tok/s and the
server's graphs-reused delta per case."""
import concurrent.futures, json, re, sys, time, urllib.request
BASE = sys.argv[1]; LOG = sys.argv[2]
P = ["Write a short story about a lighthouse keeper who finds a message in a bottle.",
     "Explain how a refrigerator works to a ten-year-old."]
def one(p, extra):
    body = {"messages": [{"role": "user", "content": p}], "max_tokens": 256, "cache_prompt": False, "ignore_eos": True}
    body.update(extra)
    r = json.load(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
    return r["timings"]["predicted_n"]
def graphs():
    g = re.findall(r"graphs reused =\s+(\d+)", open(LOG).read())
    return int(g[-1]) if g else 0
CASES = [("greedy", lambda i: {"temperature": 0}),
         ("t0.8 no seed", lambda i: {"temperature": 0.8}),
         ("t0.8 same seed", lambda i: {"temperature": 0.8, "seed": 7}),
         ("t0.8 distinct seeds", lambda i: {"temperature": 0.8, "seed": 100 + i}),
         ("t0.8 top_k 1", lambda i: {"temperature": 0.8, "top_k": 1, "seed": 100 + i})]
for c in (1, 2):
    for name, f in CASES:
        for rep in range(2):
            g0 = graphs(); t0 = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(c) as ex:
                n = sum(ex.map(lambda i: one(P[i], f(i)), range(c)))
            dt = time.perf_counter() - t0; time.sleep(0.3)
            print(f"c={c} {name:22s} rep{rep} {n/dt:6.1f} tok/s  graphs reused +{graphs()-g0}", flush=True)
