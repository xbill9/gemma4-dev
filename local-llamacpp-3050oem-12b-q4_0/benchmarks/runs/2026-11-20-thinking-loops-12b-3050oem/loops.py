"""12B thinking loops: resend GSM8K problems whose thinking-on greedy reply hit the 32,768-token
limit, with one change at a time, to the server the wave-3 run uses.
usage: loops.py ARM OUT.jsonl  (ARM: greedy | google, google = generation_config.json's
temperature 1.0, top_k 64, top_p 0.95)"""
import json, sys, time, urllib.request
sys.path.insert(0, "/home/xbill/gemma4-dev/benchmarks/gen-eval")
import gen_eval as g
SRC = "/home/xbill/gemma4-dev/local-llamacpp-3050oem-12b-q4_0/benchmarks/runs/2026-11-17-thinking-12b-3050oem/gsm8k-repack-none-thinkon.jsonl.partial"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 8
arm, out = sys.argv[1], sys.argv[2]
cut = [json.loads(l) for l in open(SRC)]
cut = [r["id"] for r in cut if r["finish"] == "length"][:N]
recs = {json.loads(l)["id"]: json.loads(l) for l in open("/home/xbill/gemma4-dev/benchmarks/gen-eval/data/gsm8k.jsonl")}
done = set()
try: done = {json.loads(l)["id"] for l in open(out)}
except FileNotFoundError: pass
for i in cut:
    if i in done: continue
    body = {"model": "google/gemma-4-12B-it", "max_tokens": 32768, "cache_prompt": False,
            "messages": [{"role": "user", "content": g.GSM8K_PROMPT.format(question=recs[i]["question"])}],
            "chat_template_kwargs": {"enable_thinking": True}}
    body.update({"temperature": 0} if arm == "greedy" else {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "seed": 1})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=7200))
    m = r["choices"][0]["message"]
    row = {"id": i, "arm": arm, "finish": r["choices"][0]["finish_reason"], "completion_tokens": r["usage"]["completion_tokens"],
           "seconds": round(time.time() - t0, 1), "right": g.gsm8k_right(m.get("content"), recs[i]["answer"]),
           "text": m.get("content"), "reasoning": m.get("reasoning_content"), "body": {k: v for k, v in body.items() if k != "messages"}}
    open(out, "a").write(json.dumps(row) + "\n")
    print(i, row["finish"], row["completion_tokens"], row["right"], row["seconds"], flush=True)
