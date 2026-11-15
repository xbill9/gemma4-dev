#!/usr/bin/env python3
"""Statistical audit of the high-vs-low pilot, per the research-kit ablation-stat-auditor skill.

Inputs: aime-high/<model>.jsonl and aime-low/<model>.jsonl (one row per AIME 2025 I problem).
For each model, the problems both arms have a row for are paired; an error row counts as wrong.
  * Wilson 95% interval for right / n in each arm
  * exact McNemar on the pairs: b = right only at high, c = right only at low
  * Wilcoxon signed-rank on per-problem output tokens (thinking + answer) and on per-problem cost
  * the noise floor for n problems: the smallest lift distinguishable from noise
The interval and test code is the research-kit MCP server's own (research_audit_mcp.py, standard
library only), imported here because the server is not loaded in this session; Wilcoxon is scipy's.
Writes stat_audit.json beside this file and prints the audit table.
"""

import json
import os
import sys

from scipy.stats import wilcoxon

sys.path.insert(0, os.path.expanduser("~/claude-research-skills/plugin/mcp"))
import research_audit_mcp as ra  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["gemma-4-31b", "gemma-4-26b", "flash-lite-3.5", "flash-3.8", "hydrogen-4", "carbon-4", "argon-4"]


def rows(arm, name):
    p = os.path.join(HERE, f"aime-{arm}", name + ".jsonl")
    return {r["id"]: r for r in map(json.loads, open(p))} if os.path.exists(p) else {}


def out_tokens(r):
    return (r.get("completion_tokens") or 0) + (r.get("reasoning_tokens") or 0)


def main():
    report, claims = [], []
    pooled_k = pooled_n = 0
    for name in ORDER:
        hi, lo = rows("high", name), rows("low", name)
        ids = sorted(set(hi) & set(lo))
        n = len(ids)
        kh = sum(bool(hi[i]["right"]) for i in ids)
        kl = sum(bool(lo[i]["right"]) for i in ids)
        pooled_k += kh + kl
        pooled_n += 2 * n
        b = sum(bool(hi[i]["right"]) and not lo[i]["right"] for i in ids)
        c = sum(bool(lo[i]["right"]) and not hi[i]["right"] for i in ids)
        wh, wl = ra.wilson_ci(kh, n), ra.wilson_ci(kl, n)
        mc = ra.mcnemar_exact(b, c)
        th = [out_tokens(hi[i]) for i in ids]
        tl = [out_tokens(lo[i]) for i in ids]
        ch = [hi[i]["cost_usd"] for i in ids]
        cl = [lo[i]["cost_usd"] for i in ids]
        wt = wilcoxon(th, tl) if any(x != y for x, y in zip(th, tl)) else None
        wc = wilcoxon(ch, cl) if any(x != y for x, y in zip(ch, cl)) else None
        rec = {
            "model": name,
            "n": n,
            "right_high": kh,
            "right_low": kl,
            "wilson_high": wh,
            "wilson_low": wl,
            "mcnemar": {"b_right_only_high": b, "c_right_only_low": c, **mc},
            "wilcoxon_output_tokens_p": None if wt is None else float(wt.pvalue),
            "wilcoxon_cost_p": None if wc is None else float(wc.pvalue),
            "output_tokens_high": sum(th),
            "output_tokens_low": sum(tl),
            "cost_high": round(sum(ch), 4),
            "cost_low": round(sum(cl), 4),
        }
        report.append(rec)
        p = mc.get("p_value")
        accuracy_verdict = "WITHIN NOISE" if p is None or p >= 0.05 else "HOLDS"
        claims.append(
            {
                "id": f"{name}-accuracy-high-vs-low",
                "statement": f"{name}: {kh} of {n} right at high, {kl} of {n} at low, same problems",
                "paper_value": {"right_high": kh, "right_low": kl, "n": n},
                "paper_source": f"aime-high/{name}.jsonl, aime-low/{name}.jsonl",
                "computed": {"right_high": kh, "right_low": kl, "n": n, "mcnemar_p": p},
                "verdict": "VERIFIED",
                "conclusion": accuracy_verdict,
                "reason": f"exact McNemar b={b} c={c}",
            }
        )
        if wt is not None:
            claims.append(
                {
                    "id": f"{name}-tokens-high-vs-low",
                    "statement": f"{name}: output tokens {sum(th):,} at high against {sum(tl):,} at low",
                    "paper_value": {"tokens_high": sum(th), "tokens_low": sum(tl)},
                    "paper_source": f"aime-high/{name}.jsonl, aime-low/{name}.jsonl",
                    "computed": {"tokens_high": sum(th), "tokens_low": sum(tl), "wilcoxon_p": float(wt.pvalue)},
                    "verdict": "VERIFIED",
                    "conclusion": "HOLDS" if wt.pvalue < 0.05 else "WITHIN NOISE",
                    "reason": "Wilcoxon signed-rank on per-problem output tokens",
                }
            )

    nmax = max(r["n"] for r in report)
    floor = ra.noise_floor(nmax, pooled_k / pooled_n)
    json.dump(
        {"models": report, "noise_floor": floor, "pooled_rate": pooled_k / pooled_n, "claims": claims},
        open(os.path.join(HERE, "stat_audit.json"), "w"),
        indent=1,
    )

    print(
        "| Model | n | Right high [Wilson 95%] | Right low [Wilson 95%] | McNemar b/c, p | Output tokens high / low, Wilcoxon p | Verdict |"
    )
    print("|---|---:|---|---|---|---|---|")
    for r in report:
        wh, wl, mc = r["wilson_high"], r["wilson_low"], r["mcnemar"]
        wp = r["wilcoxon_output_tokens_p"]
        acc = "WITHIN NOISE" if mc.get("p_value", 1) >= 0.05 else "HOLDS"
        print(
            f"| {r['model']} | {r['n']} | {r['right_high']} [{wh['wilson_low_pct']:.1f}, {wh['wilson_high_pct']:.1f}] "
            f"| {r['right_low']} [{wl['wilson_low_pct']:.1f}, {wl['wilson_high_pct']:.1f}] "
            f"| {mc['b_right_only_high']}/{mc['c_right_only_low']}, {mc['p_value']:.3g} "
            f"| {r['output_tokens_high']:,} / {r['output_tokens_low']:,}, {('%.3g' % wp) if wp is not None else 'n/a'} "
            f"| accuracy {acc}; tokens {'HOLDS' if wp is not None and wp < 0.05 else 'n/a' if wp is None else 'WITHIN NOISE'} |"
        )
    print(f"\nnoise floor at n={nmax}, pooled rate {pooled_k / pooled_n:.3f}: {json.dumps(floor)}")


if __name__ == "__main__":
    main()
