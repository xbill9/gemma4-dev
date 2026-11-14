#!/usr/bin/env python3
"""resend_errors.py <out.jsonl> <model> <task> -- re-send only the rows that ended in a request error
(HTTP 429/5xx after every retry, timeouts), with the same settings gen_eval.py used (same env), and
replace those rows in place. Rows that got an answer are never touched. Prints what it replaced."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_eval

out, model, task = sys.argv[1:4]
rows = [json.loads(line) for line in open(out)]
recs = {r["id"]: r for r in map(json.loads, open(os.path.join(gen_eval.HERE, "data", task + ".jsonl")))}
for i, r in enumerate(rows):
    if "error" in r:
        new = gen_eval.one("x", model, task, recs[r["id"]])
        new["model"] = model
        new["resent_after"] = r["error"]
        rows[i] = new
        print(json.dumps({"id": r["id"], "was": r["error"][:60], "now_right": new["right"], "error": new.get("error")}))
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(json.dumps(gen_eval.summary(rows)))
