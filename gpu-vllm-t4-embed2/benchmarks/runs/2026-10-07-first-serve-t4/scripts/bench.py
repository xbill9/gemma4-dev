import json, time, urllib.request, statistics, sys
from concurrent.futures import ThreadPoolExecutor
U = "http://127.0.0.1:8000/v1/embeddings"
def one(i, words):
    body = {"model": "google/embeddinggemma-2", "input": "title: none | text: " + " ".join(f"w{i}x{j}" for j in range(words))}
    t = time.time(); r = json.load(urllib.request.urlopen(urllib.request.Request(U, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})))
    return time.time() - t, r["usage"]["prompt_tokens"]
res = {}
for words, n_req in ((60, 64), (240, 32)):
    for c in (1, 4, 16):
        one(0, words)
        t0 = time.time()
        with ThreadPoolExecutor(c) as ex:
            out = list(ex.map(lambda i: one(i + 1000 * c, words), range(n_req)))
        wall = time.time() - t0; toks = sum(o[1] for o in out); lat = sorted(o[0] for o in out)
        row = dict(prompt_tokens=out[0][1], concurrency=c, requests=n_req, req_per_s=round(n_req / wall, 2), input_tok_per_s=round(toks / wall), p50_ms=round(1000 * statistics.median(lat)), p99_ms=round(1000 * lat[int(0.99 * (len(lat) - 1))]))
        print(json.dumps(row)); sys.stdout.flush()
