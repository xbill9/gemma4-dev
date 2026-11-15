#!/usr/bin/env python3
"""build_hard_set.py -- write the two harder question sets gen_eval.py can score.

  python3 build_hard_set.py aime                         # data/aime2025.jsonl (30, public)
  python3 build_hard_set.py gpqa --hf-token-file FILE    # data/gpqa_diamond.jsonl (198, gated)

aime2025      AIME 2025 I and II from opencompass/AIME2025 at a pinned revision; answers are
              integers 0-999, scored like GSM8K (the number after "Answer:").
gpqa_diamond  GPQA Diamond from Idavidrein/gpqa (gated: accept the terms on Hugging Face first).
              The four options are shuffled once per question with a seed taken from the question's
              own hash, so every model sees the same order; scored by the letter after "Answer:".
              GPQA asks that its questions not be published in plain text, so data/gpqa_diamond.jsonl
              and every run output holding GPQA text stay out of git (.gitignore); only scores,
              token counts and hashes are committed.

Each file's SHA-256 goes into data/hard_manifest.json.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
AIME_REPO, AIME_REV = "opencompass/AIME2025", "a6ad95f611d72cf628a80b58bd0432ef6638f958"
GPQA_REPO = "Idavidrein/gpqa"


def fetch(repo, name, revision=None, token=None):
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(repo, name, repo_type="dataset", revision=revision, token=token)
    return open(p, "rb").read()


def write(name, recs, meta):
    path = os.path.join(DATA, name + ".jsonl")
    body = "".join(json.dumps(r) + "\n" for r in recs)
    open(path, "w").write(body)
    mpath = os.path.join(DATA, "hard_manifest.json")
    man = json.load(open(mpath)) if os.path.exists(mpath) else {}
    man[name] = dict(meta, n=len(recs), sha256=hashlib.sha256(body.encode()).hexdigest())
    open(mpath, "w").write(json.dumps(man, indent=1) + "\n")
    print(f"wrote {path}: {len(recs)} records")


def aime():
    recs = []
    for part in ("I", "II"):
        raw = fetch(AIME_REPO, f"aime2025-{part}.jsonl", AIME_REV)
        for k, line in enumerate(raw.decode().splitlines(), 1):
            r = json.loads(line)
            # One answer is stored as "336^\\circ" (degrees); take the leading integer, keep the original.
            m = re.match(r"\s*(\d+)", r["answer"])
            ans = int(m.group(1))
            assert 0 <= ans <= 999, r
            rec = {"id": f"aime2025-{part}-{k:02d}", "question": r["question"], "answer": str(ans)}
            if r["answer"].strip() != str(ans):
                rec["answer_as_published"] = r["answer"]
            recs.append(rec)
    write("aime2025", recs, {"source": f"https://huggingface.co/datasets/{AIME_REPO}", "revision": AIME_REV})


def gpqa(token):
    raw = fetch(GPQA_REPO, "gpqa_diamond.csv", token=token).decode()
    recs = []
    for row in csv.DictReader(io.StringIO(raw)):
        q = row["Question"].strip()
        qid = hashlib.sha256(q.encode()).hexdigest()[:12]
        opts = [row["Correct Answer"].strip()] + [row[f"Incorrect Answer {i}"].strip() for i in (1, 2, 3)]
        order = list(range(4))
        random.Random(qid).shuffle(order)
        letter = "ABCD"[order.index(0)]
        recs.append({"id": f"gpqa-{qid}", "question": q, "options": [opts[i] for i in order], "answer": letter})
    write("gpqa_diamond", recs, {"source": f"https://huggingface.co/datasets/{GPQA_REPO}", "file": "gpqa_diamond.csv"})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("set", choices=["aime", "gpqa"])
    p.add_argument("--hf-token-file")
    a = p.parse_args()
    os.makedirs(DATA, exist_ok=True)
    if a.set == "aime":
        aime()
    else:
        if not a.hf_token_file:
            raise SystemExit("gpqa is gated: pass --hf-token-file after accepting the terms on Hugging Face")
        gpqa(open(os.path.expanduser(a.hf_token_file)).read().strip())


if __name__ == "__main__":
    main()
