#!/usr/bin/env python3
"""build_bfcl_sets.py -- write the BFCL v3 "multiple" and "irrelevance" records gen_eval.py scores.

  data/bfcl_multiple.jsonl     all 200 records: several functions offered, one call to the right one expected;
                               {"id", "question", "functions", "possible"} as data/bfcl_simple.jsonl.
  data/bfcl_irrelevance.jsonl  all 240 records: no offered function fits the request, so the right answer
                               is no call; {"id", "question", "functions"} (BFCL ships no answer file).

Fetched from the same source as build_gen_set.py's bfcl_simple and pinned by sha256 in
data/bfcl_manifest.json. build_gen_set.py and its files are left as they are.
"""

import hashlib
import json
import os

from build_gen_set import BFCL, HERE, fetch, jsonl


def main():
    out = os.path.join(HERE, "data")
    manifest = {"bfcl_source": BFCL}
    for cat in ("multiple", "irrelevance"):
        q_raw = fetch(BFCL + f"BFCL_v3_{cat}.json")
        manifest[f"{cat}_questions_sha256"] = hashlib.sha256(q_raw).hexdigest()
        answers = {}
        if cat != "irrelevance":
            a_raw = fetch(BFCL + f"possible_answer/BFCL_v3_{cat}.json")
            manifest[f"{cat}_answers_sha256"] = hashlib.sha256(a_raw).hexdigest()
            answers = {r["id"]: r["ground_truth"] for r in jsonl(a_raw)}
        rows = []
        for r in jsonl(q_raw):
            row = {"id": r["id"], "question": r["question"][0], "functions": r["function"]}
            if cat != "irrelevance":
                row["possible"] = answers[r["id"]]
            rows.append(row)
        name = f"bfcl_{cat}.jsonl"
        with open(os.path.join(out, name), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        manifest[f"bfcl_{cat}"] = len(rows)
        manifest[name + ".sha256"] = hashlib.sha256(open(os.path.join(out, name), "rb").read()).hexdigest()
    json.dump(manifest, open(os.path.join(out, "bfcl_manifest.json"), "w"), indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
