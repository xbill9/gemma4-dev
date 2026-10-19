#!/usr/bin/env python3
"""resumable_eval.py -- jev-tpu-v5e1/gen_eval.py's `run`, one request at a time, appending each scored
record to the output as it lands and skipping ids already there, so a run that takes a day or more
survives a restart. Requests, prompts and scoring are gen_eval.py's own functions, unchanged.

  resumable_eval.py <upstream> <model> <task> <out.jsonl>

The final summary line matches gen_eval.py's, except that wall_s is the sum of the records' own
`seconds` (time spent in requests, across restarts) and `segments` counts the starts."""

import json
import os
import sys
import time

sys.path.insert(0, "/home/xbill/gemma4-dev/jev-tpu-v5e1")
import gen_eval  # noqa: E402


def main():
    upstream, model, task, out = sys.argv[1:5]
    recs = [json.loads(line) for line in open(os.path.join(gen_eval.HERE, "data", task + ".jsonl"))]
    done = {}
    if os.path.exists(out):
        for line in open(out):
            try:
                r = json.loads(line)
            except ValueError:
                continue  # a line cut short by a stop
            if "error" not in r:
                done[r["id"]] = r
    seg_file = out + ".segments"
    segs = int(open(seg_file).read()) + 1 if os.path.exists(seg_file) else 1
    open(seg_file, "w").write(str(segs))
    with open(out, "w") as f:  # rewrite without errored or truncated lines, then append
        for r in done.values():
            f.write(json.dumps(r) + "\n")
    with open(out, "a") as f:
        for i, rec in enumerate(recs):
            if rec["id"] in done:
                continue
            r = gen_eval.one(upstream, model, task, rec)
            r["model"] = model
            f.write(json.dumps(r) + "\n")
            f.flush()
            done[r["id"]] = r
            print(f"{time.strftime('%T')} {len(done)}/{len(recs)} {r['id']} right={r.get('right')} "
                  f"tokens={r.get('completion_tokens')} s={r.get('seconds')}", flush=True)
    rows = [done[rec["id"]] for rec in recs]
    s = gen_eval.summary(rows)
    s["wall_s"] = round(sum(r.get("seconds") or 0 for r in rows), 1)
    s["segments"] = segs
    print(json.dumps(s))


if __name__ == "__main__":
    main()
