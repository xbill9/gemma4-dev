"""Fetch each task's labelled split and write a fixed, class-stratified sample.

    python3 build_eval_set.py --per-task 300

Rows come from the Hugging Face datasets-server, so there is no `datasets`
dependency. The sample is drawn with a fixed seed and written to
data/<task>.jsonl, one {"id", "text", "gold"} per line, so every arm of the
comparison reads the same examples in the same order.
"""

import argparse
import json
import os
import random
import time
import urllib.parse
import urllib.request

from tasks import TASKS

ROWS_URL = "https://datasets-server.huggingface.co/rows"
PAGE = 100  # the server's maximum page length


def fetch(url, tries=8):
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(min(60, 2 ** (attempt + 1)))  # the server answers 429 to bursts


def fetch_split(task):
    rows = []
    offset = 0
    while True:
        q = urllib.parse.urlencode(
            {
                "dataset": task["dataset"],
                "config": task["config"],
                "split": task["split"],
                "offset": offset,
                "length": PAGE,
            }
        )
        page = fetch(f"{ROWS_URL}?{q}")
        time.sleep(0.5)
        batch = page.get("rows", [])
        rows += [(r["row_idx"], r["row"]) for r in batch]
        offset += len(batch)
        if not batch or offset >= page.get("num_rows_total", 0):
            return rows


def stratified(rows, task, n, seed):
    """n examples with each class represented in proportion to the split,
    rounding so the total is exactly n."""
    by_class = {}
    for idx, row in rows:
        by_class.setdefault(row[task["label_field"]], []).append((idx, row))
    rng = random.Random(seed)
    total = len(rows)
    quotas = {c: n * len(v) // total for c, v in by_class.items()}
    # hand out the rounding remainder to the largest classes first
    for c in sorted(by_class, key=lambda c: -len(by_class[c]))[: n - sum(quotas.values())]:
        quotas[c] += 1
    picked = []
    for c in sorted(by_class):
        picked += rng.sample(by_class[c], quotas[c])
    rng.shuffle(picked)
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-task", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data"))
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name in args.tasks:
        task = TASKS[name]
        rows = fetch_split(task)
        labels = {row[task["label_field"]] for _, row in rows}
        unknown = labels - set(task["gold"])
        if unknown:
            raise SystemExit(f"{name}: labels {sorted(unknown)} have no option name")
        picked = stratified(rows, task, args.per_task, args.seed)
        path = os.path.join(args.out, f"{name}.jsonl")
        with open(path, "w") as f:
            for idx, row in picked:
                f.write(
                    json.dumps(
                        {
                            "id": f"{name}-{idx}",
                            "text": row[task["text_field"]],
                            "gold": task["gold"][row[task["label_field"]]],
                        }
                    )
                    + "\n"
                )
        counts = {}
        for _, row in picked:
            g = task["gold"][row[task["label_field"]]]
            counts[g] = counts.get(g, 0) + 1
        print(f"{name}: {len(rows)} rows in split -> {len(picked)} written {counts}")


if __name__ == "__main__":
    main()
