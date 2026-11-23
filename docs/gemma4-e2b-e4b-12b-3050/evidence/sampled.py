#!/usr/bin/env python3
"""Google's sampling settings (temperature 1.0, top-k 64, top-p 0.95) against temperature 0, per model.

Paired by problem, exact two-sided McNemar. Sampled runs: 2026-11-21 (12B, thinking on, no seed) and
2026-11-22 (all three; seed 1, and seed 2 for the 12B AIME thinking-on repeat). Greedy runs: the waves.
All without a drafter, one request at a time, 32,768-token limit. Writes sampled.txt.
"""

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RUNS = "local-llamacpp-3050oem-{}-q4_0/benchmarks/runs/"
W = {
    "E2B": RUNS.format("2b") + "2026-11-16-thinking-e2b-3050oem/",
    "E4B": RUNS.format("4b") + "2026-11-17-thinking-e4b-3050oem/",
    "12B": RUNS.format("12b") + "2026-11-17-thinking-12b-3050oem/",
}
S22 = RUNS.format("12b") + "2026-11-22-sampling-e2b-e4b-12b-3050oem/"
S21 = RUNS.format("12b") + "2026-11-21-drafter-sampling-12b-3050oem/"


def load(rel):
    p = os.path.join(R, rel)
    if not os.path.exists(p) and os.path.exists(p + ".partial"):
        p += ".partial"  # the 12B GSM8K thinking-on wave stopped at 157 problems
    return {r["id"]: r for r in map(json.loads, open(p)) if "id" in r}


def cell(m, task, mode, think, seed=1):
    lo = m.lower()
    if mode == "greedy":
        if m == "12B" and task == "gsm8k" and think == "on":
            return load(S21 + "gsm8k-greedy-q4x-thinkon.jsonl")  # not used: wave run is the 157
        return load(W[m] + f"{task}-repack-none-think{think}.jsonl")
    if m == "12B" and think == "on" and (task == "gsm8k" or seed == 0):
        return load(S21 + f"{task}-sampled-none-thinkon.jsonl")
    return load(S22 + f"{task}-{lo}-sampled-seed{seed}-think{think}.jsonl")


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


out = []


def say(s=""):
    out.append(s)
    print(s)


def stats(d, ids):
    rs = [d[i] for i in ids]
    return (
        sum(r["right"] is True for r in rs),
        sum(r["finish"] == "length" for r in rs),
        sum(r["seconds"] for r in rs) / 3600,
    )


def pair(label, A, an, B, bn):
    ids = sorted(set(A) & set(B))
    ka, ca, ha = stats(A, ids)
    kb, cb, hb = stats(B, ids)
    b = sum(A[i]["right"] is True and B[i]["right"] is not True for i in ids)
    c = sum(B[i]["right"] is True and A[i]["right"] is not True for i in ids)
    say(
        f"{label}, n={len(ids)}: {an} {ka} right, {ca} cut, {ha:.2f} h | {bn} {kb} right, {cb} cut, {hb:.2f} h"
        f" | {an} only {b}, {bn} only {c}, McNemar p={mcnemar(b, c):.3g} | time {bn}/{an} {hb / ha:.2f}x"
    )


g157 = set(load(W["12B"] + "gsm8k-repack-none-thinkon.jsonl"))
say("# GSM8K, first 157 problems (the 12B thinking-on wave stopped there)")
for m in ("E2B", "E4B", "12B"):
    d = {k: {i: v[i] for i in g157} for k, v in [("off", cell(m, "gsm8k", "greedy", "off")),
                                                  ("on", cell(m, "gsm8k", "greedy", "on") if m != "12B"
                                                   else load(W["12B"] + "gsm8k-repack-none-thinkon.jsonl")),
                                                  ("son", cell(m, "gsm8k", "sampled", "on"))]}
    for k, v in d.items():
        kk, cc, hh = stats(v, sorted(g157))
        say(f"{m} gsm8k {k}: {kk}/157 right, {cc} cut, {hh:.2f} h")
    pair(f"{m} gsm8k greedy on -> sampled on", d["on"], "greedy-on", d["son"], "sampled-on")
    pair(f"{m} gsm8k greedy off -> sampled on", d["off"], "greedy-off", d["son"], "sampled-on")
s12off = {i: v for i, v in cell("12B", "gsm8k", "sampled", "off").items() if i in g157}
kk, cc, hh = stats(s12off, sorted(g157))
say(f"12B gsm8k soff: {kk}/157 right, {cc} cut, {hh:.2f} h")
pair("12B gsm8k sampled off -> sampled on", s12off, "sampled-off",
     {i: v for i, v in cell("12B", "gsm8k", "sampled", "on").items() if i in g157}, "sampled-on")
pair("12B gsm8k greedy off -> sampled off", {i: v for i, v in cell("12B", "gsm8k", "greedy", "off").items()
                                             if i in g157}, "greedy-off", s12off, "sampled-off")
say()
say("# AIME 2025, 30 problems")
for m in ("E2B", "E4B", "12B"):
    gon, goff = cell(m, "aime2025", "greedy", "on"), cell(m, "aime2025", "greedy", "off")
    soff = cell(m, "aime2025", "sampled", "off")
    ons = [("sampled-on", cell(m, "aime2025", "sampled", "on"))] if m != "12B" else [
        ("sampled-on-11-21", cell(m, "aime2025", "sampled", "on", seed=0)),
        ("sampled-on-seed2", cell(m, "aime2025", "sampled", "on", seed=2))]
    pair(f"{m} aime greedy off -> sampled off", goff, "greedy-off", soff, "sampled-off")
    for n_, son in ons:
        pair(f"{m} aime sampled off -> {n_}", soff, "sampled-off", son, n_)
        pair(f"{m} aime greedy on -> {n_}", gon, "greedy-on", son, n_)
say()
say("# Cross-model, Google's settings")
e4on = {i: v for i, v in cell("E4B", "gsm8k", "sampled", "on").items() if i in g157}
pair("gsm8k E4B sampled on vs 12B sampled off", e4on, "E4B-on", s12off, "12B-off")
pair("gsm8k E4B sampled on vs 12B sampled on", e4on, "E4B-on",
     {i: v for i, v in cell("12B", "gsm8k", "sampled", "on").items() if i in g157}, "12B-on")
pair("gsm8k E4B sampled on vs 12B greedy off", e4on, "E4B-on",
     {i: v for i, v in cell("12B", "gsm8k", "greedy", "off").items() if i in g157}, "12B-greedy-off")
pair("aime E4B sampled on vs 12B sampled off", cell("E4B", "aime2025", "sampled", "on"), "E4B-on",
     cell("12B", "aime2025", "sampled", "off"), "12B-off")
pair("aime E4B sampled on vs 12B sampled on seed2", cell("E4B", "aime2025", "sampled", "on"), "E4B-on",
     cell("12B", "aime2025", "sampled", "on", seed=2), "12B-on")
pair("aime E4B sampled off vs 12B sampled off", cell("E4B", "aime2025", "sampled", "off"), "E4B-off",
     cell("12B", "aime2025", "sampled", "off"), "12B-off")
# Longest reply and cut-offs across every sampled run
say()
for m in ("E2B", "E4B", "12B"):
    rows = []
    for f in sorted(os.listdir(os.path.join(R, S22))):
        if f.endswith(".jsonl") and f"-{m.lower()}-" in f:
            rows += list(load(S22 + f).values())
    if m == "12B":
        rows += list(load(S21 + "gsm8k-sampled-none-thinkon.jsonl").values())
        rows += list(load(S21 + "aime2025-sampled-none-thinkon.jsonl").values())
    say(f"{m} all sampled replies: {len(rows)}, cut {sum(r['finish'] == 'length' for r in rows)}")
open(os.path.join(HERE, "sampled.txt"), "w").write("\n".join(out) + "\n")

# Totals quoted in the article (arithmetic on figures above and in rollup.txt / memory.txt)
E4B_GREEDY_REPLIES, E4B_GREEDY_CUT = 10792, 0  # rollup.txt: E4B replies at the limit, all temperature-0 runs
CARD_MIB, E4B_MIB = 8192, 6430  # e4b-host.txt: the card (8192 MiB); memory.txt: E4B serving at 131,072 tokens
more = [
    f"E4B replies at the limit, temperature 0 and Google's settings: {E4B_GREEDY_CUT} of {E4B_GREEDY_REPLIES + 217:,}",
    f"E4B card memory to spare: {CARD_MIB:,} - {E4B_MIB:,} = {CARD_MIB - E4B_MIB:,} MiB",
]
for s in more:
    print(s)
open(os.path.join(HERE, "sampled.txt"), "a").write("\n".join(more) + "\n")
