#!/usr/bin/env python3
"""Thinking and answer tokens per reply, counted with Gemma 4 E4B's tokenizer.json (pass its path).
Control: on thinking-off replies, completion_tokens - tokens(text) must be the same constant on every
reply (the end-of-turn token); on thinking-on replies, completion_tokens - tokens(text) - tokens(reasoning)
likewise (the thinking delimiters). Also a Wilcoxon signed-rank test on each problem's generated tokens,
thinking on against off. Writes tokens.md and tokens.json."""
import json, os, statistics as st, sys
from collections import Counter
from tokenizers import Tokenizer
from scipy.stats import wilcoxon
HERE = os.path.dirname(os.path.abspath(__file__))
T = Tokenizer.from_file(sys.argv[1])
n = lambda s: len(T.encode(s or "", add_special_tokens=False).ids)
ARMS = ["repack-none", "repack-q4x", "google-none", "google-q4x"]
out, L = {}, ["# Thinking and answer tokens, wave 2", "", "Counted by `tokens.py` with Gemma 4 E4B's tokenizer.", ""]
ctrl = Counter()
for t in ("gsm8k", "aime2025"):
    L += [f"## {t}", "", "| arm | thinking | mean thinking tokens | mean answer tokens | median answer tokens | generated tokens, on vs off: Wilcoxon p | problems with more tokens when on |", "|---|---|---|---|---|---|---|"]
    for a in ARMS:
        rows = {th: [json.loads(l) for l in open(os.path.join(HERE, f"{t}-{a}-think{th}.jsonl"))] for th in ("off", "on")}
        for th, rs in rows.items():
            for r in rs:
                if r.get("finish") == "stop":
                    ctrl[(th, r["completion_tokens"] - n(r["text"]) - (n(r["reasoning"]) if th == "on" else 0))] += 1
        off = {r["id"]: r["completion_tokens"] for r in rows["off"]}; on = {r["id"]: r["completion_tokens"] for r in rows["on"]}
        ids = sorted(off); p = wilcoxon([on[i] for i in ids], [off[i] for i in ids]).pvalue
        more = sum(on[i] > off[i] for i in ids)
        for th in ("off", "on"):
            think = [n(r.get("reasoning")) for r in rows[th]]; ans = [n(r["text"]) for r in rows[th]]
            out[f"{t} {a} {th}"] = dict(think_mean=st.mean(think), answer_mean=st.mean(ans), answer_median=st.median(ans))
            c = out[f"{t} {a} {th}"]
            L.append(f"| {a} | {th} | {c['think_mean']:.0f} | {c['answer_mean']:.0f} | {c['answer_median']:.0f} | {'' if th == 'off' else f'{p:.2e}'} | {'' if th == 'off' else f'{more}/{len(ids)}'} |")
        out[f"{t} {a} wilcoxon_p"] = p; out[f"{t} {a} more_when_on"] = [more, len(ids)]
    L.append("")
L += ["## Control", "", "completion_tokens minus counted tokens, on replies that stopped normally: " + ", ".join(f"thinking {th}: {d} on {k} replies" for (th, d), k in sorted(ctrl.items())), ""]
out["control"] = {f"{th} {d}": k for (th, d), k in ctrl.items()}
json.dump(out, open(os.path.join(HERE, "tokens.json"), "w"), indent=1)
open(os.path.join(HERE, "tokens.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
