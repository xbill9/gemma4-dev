"""n-gram speculative decoding vs none, single user, on this rig's demo serving flags.

    python3 spec_ab.py OUT_DIR

For each config (llama-server --spec-type), starts the server on PORT with the rig's
`make serve` flags plus the spec flags, sends every prompt REPEATS times at temperature 0
with thinking off, records the server's own timings (predicted_per_second and the draft
counters when present) and the text, then stops the server. Configs run in ABBA order
(forward, then reversed), the card cooled to <= COOL_C before each. Outputs are compared
to the no-speculation text: greedy speculative decoding must not change a token.
"""
import json, os, statistics as st, subprocess, sys, time, urllib.request

OUT = sys.argv[1]
BIN = "/home/xbill/llama.cpp/build/bin/llama-server"
MODEL = "/home/xbill/models/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf"
PORT, REPEATS, COOL_C = 8099, 3, 50
BASE = [BIN, "-m", MODEL, "--host", "127.0.0.1", "--port", str(PORT), "-ngl", "99", "-c", "8192",
        "-ctk", "f16", "-ctv", "f16", "-fa", "1", "-t", "6", "-tb", "12", "--parallel", "1",
        "--reasoning", "off"]
CONFIGS = {"none": [], "ngram-simple": ["--spec-type", "ngram-simple"], "ngram-map-k": ["--spec-type", "ngram-map-k"],
           "ngram-mod": ["--spec-type", "ngram-mod"], "ngram-cache": ["--spec-type", "ngram-cache"]}

CODE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_code.py")).read()
PARA = ("The quick brown fox jumps over the lazy dog. This sentance contains every letter of the alphabet, "
        "which is why typists and designers have used it for decades to test keyboards and fonts. It is short, "
        "memorable, and easy to type, though it is not the shortest such sentance. Shorter pangrams exist, but "
        "they tend to use obscure words that few people recognise. For everyday testing, the fox remains the "
        "favourite, and it apears in manuals, software samples and typing tutors around the world.")
PROMPTS = {
    "edit-code": "Add type hints to every function in this Python file. Return the complete file and nothing else; "
                 "change nothing else.\n\n" + CODE,
    "edit-text": "Fix the spelling mistakes in this paragraph. Return the full corrected paragraph and nothing else.\n\n" + PARA,
    "open-explain": "Explain what quantizing a model means, in three sentences, for an engineer.",
    "open-story": "Write a short story, about 200 words, about a lighthouse keeper who finds a message in a bottle.",
}


def gpu_temp():
    return int(subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"]).strip())


def ask(prompt):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 1024,
                       "seed": 0, "cache_prompt": False}).encode()
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=300))
    return time.time() - t0, r


results, texts = [], {}
order = list(CONFIGS) + list(reversed(CONFIGS))
log = open(os.path.join(OUT, "spec_ab.log"), "w")
for i, name in enumerate(order):
    while gpu_temp() > COOL_C:
        time.sleep(5)
    t_start = gpu_temp()
    srv = subprocess.Popen(BASE + CONFIGS[name], stdout=open(os.path.join(OUT, f"server-{i}-{name}.log"), "w"),
                           stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            break
        except Exception:
            time.sleep(1)
    ask("hi")  # warm-up, not recorded
    for pname, prompt in PROMPTS.items():
        for rep in range(REPEATS):
            wall, r = ask(prompt)
            tm = r.get("timings", {})
            text = r["choices"][0]["message"]["content"]
            texts.setdefault(name, {}).setdefault(pname, []).append(text)
            row = {"pass": i, "config": name, "prompt": pname, "rep": rep, "wall_s": round(wall, 3),
                   "predicted_n": tm.get("predicted_n"), "predicted_per_second": tm.get("predicted_per_second"),
                   "prompt_ms": tm.get("prompt_ms"), "draft_n": tm.get("draft_n"),
                   "draft_n_accepted": tm.get("draft_n_accepted"), "finish": r["choices"][0]["finish_reason"]}
            results.append(row)
            print(json.dumps(row), file=log, flush=True)
    srv.terminate()
    srv.wait()
    print(f"pass {i} {name}: gpu {t_start} -> {gpu_temp()} C", file=log, flush=True)

json.dump({"results": results, "texts": texts}, open(os.path.join(OUT, "spec_ab.json"), "w"), indent=1)

base = {p: v[0] for p, v in texts["none"].items()}
print("| config | prompt | tok/s (median of 6) | vs none | tokens | drafted | accepted | same text as none |")
print("|---|---|---:|---:|---:|---:|---:|---|")
med = {}
for name in CONFIGS:
    for pname in PROMPTS:
        rows = [r for r in results if r["config"] == name and r["prompt"] == pname]
        med[name, pname] = st.median(r["predicted_per_second"] for r in rows)
for name in CONFIGS:
    for pname in PROMPTS:
        rows = [r for r in results if r["config"] == name and r["prompt"] == pname]
        dn = sum(r["draft_n"] or 0 for r in rows); da = sum(r["draft_n_accepted"] or 0 for r in rows)
        same = all(t == base[pname] for t in texts[name][pname])
        print(f"| {name} | {pname} | {med[name, pname]:.2f} | {med[name, pname] / med['none', pname]:.3f} | "
              f"{st.median(r['predicted_n'] for r in rows):.0f} | {dn} | {da} | {'yes' if same else 'NO'} |")
