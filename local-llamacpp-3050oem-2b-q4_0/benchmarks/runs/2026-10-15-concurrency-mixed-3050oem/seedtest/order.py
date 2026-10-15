"""Order effect: two greedy requests at once, before and after a burst of N requests, on a
server at the run's -c 32768 --parallel 64. Prints tok/s and graphs-reused delta."""
import concurrent.futures, json, re, sys, time, urllib.request
BASE, LOG = sys.argv[1], sys.argv[2]
P = ["Write a short story about a lighthouse keeper who finds a message in a bottle.",
     "Explain how a refrigerator works to a ten-year-old."]
def one(p, extra=None):
    body = {"messages": [{"role": "user", "content": p}], "max_tokens": 256, "cache_prompt": False, "temperature": 0}
    body.update(extra or {})
    r = json.load(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
    return r["timings"]["predicted_n"]
def graphs():
    g = re.findall(r"graphs reused =\s+(\d+)", open(LOG).read()); return int(g[-1]) if g else 0
def cell(label, c, extra=None):
    g0 = graphs(); t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(c) as ex:
        n = sum(ex.map(lambda i: one(P[i % 2], extra), range(c)))
    dt = time.perf_counter() - t0; time.sleep(0.3)
    print(f"{label:28s} c={c:<3} {n/dt:6.1f} tok/s  graphs reused +{graphs()-g0}", flush=True)
for b in map(int, sys.argv[3].split(",")):
    cell("before", 2); cell("burst", b); cell("after", 2); cell("after, sampled", 2, {"temperature": 0.8})
