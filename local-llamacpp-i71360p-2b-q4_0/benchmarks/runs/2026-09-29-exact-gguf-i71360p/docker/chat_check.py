import json, sys, urllib.request
P = ["What is the capital of Australia? One sentence.",
     "What is 17 * 23? Answer with just the number.",
     "Write a Python function that reverses a string. Code only.",
     "Name three moons of Jupiter, comma separated.",
     "Translate to French: The cat is sleeping on the sofa."]
out = []
for p in P:
    body = json.dumps({"messages": [{"role": "user", "content": p}], "temperature": 0, "max_tokens": 200, "seed": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8099/v1/chat/completions", body, {"Content-Type": "application/json"})))
    out.append({"prompt": p, "text": r["choices"][0]["message"]["content"], "timings": r.get("timings", {})})
json.dump(out, open(sys.argv[1], "w"), indent=1)
for o in out: print(repr(o["text"][:160]), f'{o["timings"].get("predicted_per_second", 0):.1f} tok/s')
