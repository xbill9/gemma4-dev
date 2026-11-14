"""The answers cut at the token limit in the 2026-11-14-accuracy-api run where thinking took the limit,
and the same problems sent again with the GSM8K limit raised from 2,048 to 16,384 (same model, same
reasoning_effort none, temperature 0). A diagnostic: the article's scores are the 2,048-token run."""
import json, os, sys
os.environ.update(JEV_CHAT_URL="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                  JEV_API_KEY_FILE=os.path.expanduser("~/gemini.key"), JEV_RETRIES="8", JEV_TIMEOUT_S="600",
                  JEV_GEN_MAX_TOKENS="16384", JEV_REASONING_EFFORT="none")
H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "jev-tpu-v5e1"))
import gen_eval
recs = {r["id"]: r for r in map(json.loads, open(os.path.join(gen_eval.HERE, "data", "gsm8k.jsonl")))}
out = []
for slot, model in (("flash38", "gemini-3.8-flash"), ("carbon4", "gemini-4-carbon")):
    rows = [json.loads(l) for l in open(f"{ROOT}/gemini-api-google-{slot}/benchmarks/runs/2026-11-14-accuracy-api/gsm8k.jsonl")]
    for r in rows:
        if r.get("finish") == "length" and (r.get("reasoning_tokens") or 0) > 0:
            n = gen_eval.one("x", model, "gsm8k", recs[r["id"]])
            out.append(f"{model:18s} {r['id']:11s} at 2,048: answer {r['completion_tokens']} thinking {r['reasoning_tokens']} right {r['right']}"
                       f"  | at 16,384: finish {n.get('finish')} answer {n.get('completion_tokens')} thinking {n.get('reasoning_tokens', 0)} right {n.get('right')} {n.get('error', '')}")
            print(out[-1], flush=True)
open(f"{H}/cut-rows.txt", "w").write("\n".join(out) + "\n")
