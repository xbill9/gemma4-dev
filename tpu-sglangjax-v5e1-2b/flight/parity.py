"""Greedy parity: the six ladder prompts (256 tokens, thinking off) against the reply hashes the bf16 JAX engine
(jax-dev/tpu-jax-v5e1-2b) and the Rust-driven one recorded on the 2026-11-06 v5e-1 ladder (parity-ref.json).
Same request body as ladder.py. Usage: python3.13 parity.py [OUT.jsonl]"""
import ast, hashlib, json, os, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "ladder.py")).read()
PROMPTS = next(ast.literal_eval(n.value) for n in ast.parse(src).body
               if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "PROMPTS")
MODEL, URL = "google/gemma-4-E2B-it", "http://127.0.0.1:8000"
ref = json.load(open(os.path.join(HERE, "parity-ref.json")))
out = open(sys.argv[1], "w") if len(sys.argv) > 1 else None
match = 0
for name, prompt in PROMPTS.items():
    b = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 256,
         "max_completion_tokens": 256, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter()
    with urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions", json.dumps(b).encode(),
                                {"Content-Type": "application/json"}), timeout=3000) as r:
        d = json.load(r)
    w = time.perf_counter() - t0
    text = d["choices"][0]["message"].get("content") or ""
    n = d["usage"]["completion_tokens"]
    sha = hashlib.sha256(text.encode()).hexdigest()[:16]
    want = ref[name][0]
    ok = sha == want
    match += ok
    rec = dict(prompt=name, tokens=n, wall_s=round(w, 3), sha=sha, jax_bf16_sha=want, jax_bf16_tokens=ref[name][1],
               jaxrust_sha=ref["_jaxrust"].get(name), match_jax_bf16=ok, text=text)
    print(f"{name:8s} tokens={n} wall={w:.1f}s sha={sha} jax-bf16={want} {'MATCH' if ok else 'differ'}  {text[:160]!r}",
          flush=True)
    if out:
        out.write(json.dumps(rec) + "\n")
print(f"matches jax-bf16: {match}/{len(PROMPTS)}")
