"""Paired score difference, engine minus llama.cpp no-drafter, with a 95% bootstrap range and right/wrong disagreements."""
import json, random, sys
A = sys.argv[1]
for t, ref in (("bfcl_simple", "bfcl_simple-e4b-none"), ("gsm8k", "gsm8k-e4b-none")):
    load = lambda p: {r["id"]: bool(r["right"]) for r in map(json.loads, open(p)) if "id" in r}
    e, l = load(f"{t}-pytorch.jsonl"), load(f"{A}/{ref}.jsonl")
    ids = sorted(set(e) & set(l)); d = [int(e[i]) - int(l[i]) for i in ids]; n = len(ids)
    random.seed(0); bs = sorted(sum(random.choices(d, k=n)) / n for _ in range(10000))
    print(json.dumps({"task": t, "n": n, "engine": round(sum(e[i] for i in ids) / n, 4), "llamacpp": round(sum(l[i] for i in ids) / n, 4),
                      "diff": round(sum(d) / n, 4), "range95": [round(bs[250], 4), round(bs[9749], 4)],
                      "engine_only": d.count(1), "llamacpp_only": d.count(-1)}))
