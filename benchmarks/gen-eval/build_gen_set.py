#!/usr/bin/env python3
"""build_gen_set.py -- write the generation and tool-calling records gen_eval.py scores.

  data/gsm8k.jsonl        GSM8K test split, all 1,319 problems: {"id", "question", "answer"} where
                          answer is the number after "####" in the reference solution.
  data/bfcl_simple.jsonl  BFCL v3 "simple" (one function, one call), all 400 records:
                          {"id", "question", "functions", "possible"} with the schema as BFCL ships it
                          and the accepted values per parameter from possible_answer/.

Both are fetched from their public sources and pinned by sha256 in data/gen_manifest.json, so a
bundle carries the exact records. Run once on a machine with network access; the TPU VM only reads
the files.
"""

import hashlib
import json
import os
import urllib.request

GSM8K = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
BFCL = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/"
HERE = os.path.dirname(os.path.abspath(__file__))


def fetch(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def jsonl(raw):
    return [json.loads(line) for line in raw.decode().splitlines() if line.strip()]


def gsm8k_answer(solution):
    return solution.split("####")[-1].strip().replace(",", "")


def main():
    out = os.path.join(HERE, "data")
    manifest = {}
    raw = fetch(GSM8K)
    manifest["gsm8k_source"] = {"url": GSM8K, "sha256": hashlib.sha256(raw).hexdigest()}
    rows = [
        {"id": f"gsm8k-{i}", "question": r["question"], "answer": gsm8k_answer(r["answer"])}
        for i, r in enumerate(jsonl(raw))
    ]
    with open(os.path.join(out, "gsm8k.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    manifest["gsm8k"] = len(rows)

    q_raw, a_raw = fetch(BFCL + "BFCL_v3_simple.json"), fetch(BFCL + "possible_answer/BFCL_v3_simple.json")
    manifest["bfcl_source"] = {
        "url": BFCL,
        "questions_sha256": hashlib.sha256(q_raw).hexdigest(),
        "answers_sha256": hashlib.sha256(a_raw).hexdigest(),
    }
    answers = {r["id"]: r["ground_truth"] for r in jsonl(a_raw)}
    rows = []
    for r in jsonl(q_raw):
        turn = r["question"][0]  # simple: one turn
        rows.append({"id": r["id"], "question": turn, "functions": r["function"], "possible": answers[r["id"]]})
    with open(os.path.join(out, "bfcl_simple.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    manifest["bfcl_simple"] = len(rows)
    for name in ("gsm8k.jsonl", "bfcl_simple.jsonl"):
        manifest[name + ".sha256"] = hashlib.sha256(open(os.path.join(out, name), "rb").read()).hexdigest()
    json.dump(manifest, open(os.path.join(out, "gen_manifest.json"), "w"), indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
