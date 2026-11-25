#!/usr/bin/env python3
"""Same cell on two cards: how many problems get the same right/wrong verdict, the same extracted
answer, and byte-identical reply text. Reads the two run directories' .jsonl files."""
import json
from scipy.stats import binomtest
G = "/home/xbill/gemma4-dev"
R1 = f"{G}/local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-11-23-thinking-e2b-1650ti"
R3 = f"{G}/local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-11-16-thinking-e2b-3050oem"
A = ["repack-none", "repack-q4x", "google-none", "google-q4x"]
same_g = []
for t in ("gsm8k", "aime2025"):
    for a in A:
        for th in ("off", "on"):
            f = f"{t}-{a}-think{th}.jsonl"
            x = {r["id"]: r for r in map(json.loads, open(f"{R1}/{f}"))}; y = {r["id"]: r for r in map(json.loads, open(f"{R3}/{f}"))}
            ids = sorted(x); n = len(ids)
            same_v = sum(x[i]["right"] == y[i]["right"] for i in ids)
            same_t = sum(x[i]["text"] == y[i]["text"] and x[i].get("reasoning") == y[i].get("reasoning") for i in ids)
            only1 = sum(x[i]["right"] and not y[i]["right"] for i in ids); only3 = sum(y[i]["right"] and not x[i]["right"] for i in ids)
            p = binomtest(only1, only1 + only3).pvalue if only1 + only3 else 1.0
            same_g += [same_t] if t == "gsm8k" else []
            print(f"{t} {a} {th}: same verdict {same_v}/{n}, identical text {same_t}/{n}, right only on 1650 {only1}, only on 3050 {only3}, McNemar p {p:.4f}")
print(f"gsm8k identical text: at most {max(same_g)} of 1,319, at least {min(same_g)} of 1,319")
