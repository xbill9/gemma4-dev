#!/usr/bin/env python3
"""budget_run.py -- one task across several models, question by question, under a dollar budget.

  python3 budget_run.py <models.json> <task> <out_dir> --budget 40 [--limit N] [--wave 3]

models.json is a list of {"name", "model", "reasoning_effort", "price_in", "price_out"} with
prices in USD per million tokens (0 for free), plus optional "chat_url" / "key_file" (defaults:
GEN_EVAL_CHAT_URL / GEN_EVAL_API_KEY_FILE). Every model gets the same questions, prompt,
temperature 0 and output limit (GEN_EVAL_HARD_MAX_TOKENS, default 32768).

Spending is bounded before it happens, not after:
  * before starting, the worst case for the whole run -- every reply using its whole output
    limit, input counted at one token per two characters of prompt (an over-count) -- is printed
    and the run refuses if it exceeds --budget;
  * questions go out in waves of --wave, every model on the same questions; before each wave the
    spend so far (from the token counts each reply reports, thinking included) plus that wave's
    worst case is checked against --budget, so a stop leaves every model with the same questions
    answered.
Rows are appended to <out_dir>/<name>.jsonl as they finish; a re-run skips ids already there.
"""

import argparse
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import gen_eval as g


def cost(row, m):
    tin = row.get("prompt_tokens", 0)
    tout = row.get("completion_tokens", 0) + row.get("reasoning_tokens", 0)
    return (tin * m["price_in"] + tout * m["price_out"]) / 1e6


def prompt_chars(task, rec):
    if task == "gpqa_diamond":
        return len(g.GPQA_PROMPT) + len(rec["question"]) + sum(len(o) + 4 for o in rec["options"])
    return len(g.AIME_PROMPT) + len(rec["question"])


def worst(task, recs, models):
    return sum(
        (prompt_chars(task, r) / 2 * m["price_in"] + g.HARD_MAX_TOKENS * m["price_out"]) / 1e6
        for r in recs
        for m in models
    )


def load(path):
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(path)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models")
    p.add_argument("task", choices=["aime2025", "gpqa_diamond"])
    p.add_argument("out_dir")
    p.add_argument("--budget", type=float, required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--wave", type=int, default=3)
    p.add_argument("--dry-run", action="store_true", help="print the worst case and stop")
    a = p.parse_args()

    models = json.load(open(a.models))
    recs = [json.loads(line) for line in open(os.path.join(g.HERE, "data", a.task + ".jsonl"))]
    if a.limit:
        recs = recs[: a.limit]
    os.makedirs(a.out_dir, exist_ok=True)
    paths = {m["name"]: os.path.join(a.out_dir, m["name"] + ".jsonl") for m in models}
    done = {m["name"]: {r["id"]: r for r in load(paths[m["name"]])} for m in models}

    ceiling = worst(a.task, recs, models)
    print(
        f"{len(recs)} questions x {len(models)} models, output limit {g.HARD_MAX_TOKENS:,}: "
        f"worst case ${ceiling:.2f}, budget ${a.budget:.2f}",
        flush=True,
    )
    if a.dry_run:
        return
    if ceiling > a.budget:
        raise SystemExit("refusing: the worst case exceeds the budget; lower --limit or the output limit")

    spent = sum(cost(r, m) for m in models for r in done[m["name"]].values())
    t0 = time.time()
    for i in range(0, len(recs), a.wave):
        wave = recs[i : i + a.wave]
        jobs = [(m, r) for r in wave for m in models if r["id"] not in done[m["name"]]]
        if not jobs:
            continue
        wave_worst = sum(
            (prompt_chars(a.task, r) / 2 * m["price_in"] + g.HARD_MAX_TOKENS * m["price_out"]) / 1e6 for m, r in jobs
        )
        if spent + wave_worst > a.budget:
            print(
                f"STOP before questions {i + 1}-{i + len(wave)}: spent ${spent:.2f} + worst ${wave_worst:.2f} "
                f"> budget ${a.budget:.2f}",
                flush=True,
            )
            break

        def go(job):
            m, r = job
            opts = {k: m[k] for k in ("chat_url", "key_file", "reasoning_effort") if k in m}
            row = g.one("x", m["model"], a.task, r, opts)
            row.update(model=m["model"], name=m["name"], reasoning_effort=m.get("reasoning_effort"))
            row["cost_usd"] = round(cost(row, m), 6)
            return m, row

        # Rows are written as each reply finishes, so one slow request holds back nothing.
        with ThreadPoolExecutor(len(jobs)) as ex:
            for fut in as_completed([ex.submit(go, j) for j in jobs]):
                m, row = fut.result()
                with open(paths[m["name"]], "a") as f:
                    f.write(json.dumps(row) + "\n")
                done[m["name"]][row["id"]] = row
                spent += row["cost_usd"]
        print(
            f"{time.strftime('%T')} questions {i + 1}-{i + len(wave)} done, spent ${spent:.2f} "
            f"({time.time() - t0:.0f} s)",
            flush=True,
        )

    print(
        json.dumps(
            {
                "task": a.task,
                "spent_usd": round(spent, 4),
                "budget_usd": a.budget,
                "worst_case_usd": round(ceiling, 2),
                "wall_s": round(time.time() - t0, 1),
            }
        )
    )
    for m in models:
        rows = list(done[m["name"]].values())
        if not rows:
            continue
        print(
            json.dumps(
                {
                    "name": m["name"],
                    "reasoning_effort": m.get("reasoning_effort"),
                    "n": len(rows),
                    "right": sum(r["right"] for r in rows),
                    "errors": sum("error" in r for r in rows),
                    "cut": sum(r.get("finish") == "length" for r in rows),
                    "thinking_mean": round(statistics.mean(r.get("reasoning_tokens", 0) for r in rows), 1),
                    "answer_mean": round(statistics.mean(r.get("completion_tokens", 0) for r in rows), 1),
                    "median_s": statistics.median(r["seconds"] for r in rows),
                    "cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
                }
            )
        )


if __name__ == "__main__":
    main()
