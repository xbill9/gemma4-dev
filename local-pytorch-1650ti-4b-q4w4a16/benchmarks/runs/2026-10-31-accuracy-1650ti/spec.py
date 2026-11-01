import json, sys, urllib.request, statistics as st
P = json.load(open("../2026-10-31-first-light-1650ti/prompts-eight.json"))
out = []
for rep in range(2):
    for p in P:
        d = json.dumps({"messages": [{"role": "user", "content": p}], "max_tokens": 128, "temperature": 0, "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}).encode()
        r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8190/v1/chat/completions", d, {"Content-Type": "application/json"})))
        out.append((r["timings"]["predicted_n"], r["timings"]["predicted_ms"]))
print(json.dumps({"arm": sys.argv[1], "tok_s_mean_of_prompts": round(st.mean(n / m * 1e3 for n, m in out), 2), "tok_s_overall": round(sum(n for n, _ in out) / sum(m for _, m in out) * 1e3, 2), "min": round(min(n / m * 1e3 for n, m in out), 2), "max": round(max(n / m * 1e3 for n, m in out), 2)}))
