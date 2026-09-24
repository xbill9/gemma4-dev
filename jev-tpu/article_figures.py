"""Figures the v6e-1 article derives from committed results, computed here rather than by hand.

    python3 article_figures.py > results/2026-09-24-v6e1-DERIVED.md
"""

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
JEV = os.path.join(HERE, "..", "jev", "results")
PREFIX = "2026-09-24-v6e1"
ARMS = ("e2b", "e4b", "12b", "26b-fp8")
RATE = {"on-demand": 2.97, "flex-start": 1.35}  # europe-west4 $/chip-hour, ../TPU.md


def main():
    out = ["# Derived figures for the v6e-1 article (computed by article_figures.py)", "",
           "Rates, europe-west4, ct6e-standard-1t per chip-hour (../TPU.md): " + ", ".join(f"{k} ${v:.2f}" for k, v in RATE.items()), ""]
    log = open(os.path.join(R, f"{PREFIX}-evidence", "run.log")).read()
    for arm in ARMS:
        m = re.search(rf"{re.escape(arm)} four tasks: (\d+)s for (\d+) decisions", log)
        s = re.search(rf"{re.escape(arm)} suite: (\d+)s for (\d+) records", log)
        t, n = int(m.group(1)), int(m.group(2))
        rate = n / t
        costs = ", ".join(f"${v / 3600 / rate * 1e6:.2f} per million decisions {k}" for k, v in RATE.items())
        out.append(f"- {arm}: {n} decisions in {t} s at concurrency 8 = {rate:.1f} a second ({costs}); suite {s.group(2)} records in {s.group(1)} s = {int(s.group(2)) / int(s.group(1)):.1f} a second")

    # Throughput from the requests themselves: eight in flight, so decisions a second is
    # 8 / mean request time. The wall-clock figures above include the client's start-up.
    out.append("")
    for arm in ARMS:
        ms = [json.loads(line)["latency"]["first_ms"] for t in ("sst2", "ag_news", "emotion", "irony")
              for line in open(os.path.join(R, f"{PREFIX}-{arm}", f"autoregressive-{t}.jsonl"))]
        rate = 8000 / (sum(ms) / len(ms))
        costs = ", ".join(f"${v / 3600 / rate * 1e6:.2f} per million {k}" for k, v in RATE.items())
        out.append(f"- {arm}: mean request {sum(ms) / len(ms):.1f} ms at eight in flight = about {rate:.0f} decisions a second ({costs})")

    # Label mass and non-label top tokens, per arm and task.
    out.append("")
    for arm in ARMS:
        cells = []
        for t in ("sst2", "ag_news", "emotion", "irony"):
            rs = [json.loads(line)["reads"][0] for line in open(os.path.join(R, f"{PREFIX}-{arm}", f"autoregressive-{t}.jsonl"))]
            lm = sorted(r["label_mass"] for r in rs)
            cells.append(f"{t} median label mass {100 * lm[len(lm) // 2]:.1f}%, top token a label on {sum(r['argmax_is_label'] for r in rs)} of {len(rs)}")
        out.append(f"- {arm}: " + "; ".join(cells))

    # Suite records keep no labels_returned. Two or more labels sharing one probability below
    # 1e-8 is the fallback value's signature, so this counts a lower bound of records using it.
    out.append("")
    for arm in ARMS:
        d = os.path.join(R, f"{PREFIX}-{arm}-suite")
        n = fb = uni = 0
        per = {}
        for f in sorted(x for x in os.listdir(d) if x.endswith(".jsonl")):
            for line in open(os.path.join(d, f)):
                r = json.loads(line)
                p = r["reads"][0]["probs"]
                n += 1
                low = [x for x in p if x < 1e-8]
                hit = len(low) >= 2 and max(low) - min(low) < 1e-15
                fb += hit
                uni += len(set(round(x, 12) for x in p)) == 1
                per.setdefault(r["subset"], [0, 0])
                per[r["subset"]][0] += hit
                per[r["subset"]][1] += 1
        top = sorted(per.items(), key=lambda kv: -kv[1][0] / kv[1][1])[:3]
        out.append(f"- {arm} suite: at least {fb} of {n} records ({100 * fb / n:.1f}%) used the fallback value; uniform probabilities on {uni}; most affected: " + ", ".join(f"{k} {v[0]} of {v[1]}" for k, v in top))

    sweep = json.load(open(os.path.join(R, f"{PREFIX}-SWEEP.json")))
    small = open(os.path.join(JEV, "SMALL-MODELS.md")).read()
    l4 = {}
    for m in re.finditer(r"\| (26B AWQ|E4B bf16|E2B bf16) \| (\w+) \| ([\d.]+)% \| [\d.]+% \| ([\d.]+) \| ([\d.]+) \|", small):
        l4.setdefault(m.group(1), {})[m.group(2)] = {"acc": float(m.group(3)), "ece": float(m.group(4)), "ece50": float(m.group(5))}
    pairs = {"26b-fp8": "26B AWQ", "e4b": "E4B bf16", "e2b": "E2B bf16"}
    out.append("")
    for arm, key in pairs.items():
        d = {t: round(100 * sweep[arm][t]["accuracy"] - l4[key][t]["acc"], 1) for t in l4[key]}
        out.append(f"- TPU {arm} minus L4 {key}, accuracy points: {d}; largest gap {max(abs(v) for v in d.values())}")
    for arm in ("e2b", "e4b"):
        a = [sweep[arm][t]["agree_l4"] for t in sweep[arm]]
        out.append(f"- {arm} same predicted label as L4: {min(x['same'] for x in a)} to {max(x['same'] for x in a)} of 300 per task, {sum(x['same'] for x in a)} of {sum(x['n'] for x in a)} overall")

    suite = json.load(open(os.path.join(R, f"{PREFIX}-SUITE.json")))
    l4s = open(os.path.join(JEV, "2026-09-24-l4-suite", "STATS.md")).read()
    m = re.search(r"\| all \| 3880 \| [^|]+\| ([\d.]+)% [^|]+\| [^|]+\| ([\d.]+)% ", l4s)
    l4_26, l4_e4b = float(m.group(1)), float(m.group(2))
    out.append(f"- Suite, all records: TPU 26b-fp8 {100 * suite['26b-fp8']['groups']['all']['acc']:.1f}% vs L4 26B AWQ {l4_26:.1f}% ({100 * suite['26b-fp8']['groups']['all']['acc'] - l4_26:+.1f}); TPU e4b {100 * suite['e4b']['groups']['all']['acc']:.1f}% vs L4 E4B {l4_e4b:.1f}% ({100 * suite['e4b']['groups']['all']['acc'] - l4_e4b:+.1f})")

    bound = []
    for arm in ARMS:
        for t, v in sweep[arm].items():
            bound.append((abs(v["at_bound"]["ece"] - v["ece"]), abs(v["at_bound"]["ece_fit50_mean20"] - v["ece_fit50_mean20"])))
    out.append(f"- Top-32 bound vs floor, largest change over all arms and tasks: raw ECE {max(b[0] for b in bound):.3f}, ECE after 50 labels {max(b[1] for b in bound):.3f}")
    ret = {arm: (min(v["all_labels_returned"] for v in sweep[arm].values()), max(v["all_labels_returned"] for v in sweep[arm].values())) for arm in ARMS}
    out.append("- Share of reads with every label in the top 32, lowest and highest task: " + ", ".join(f"{a} {100 * lo:.1f}%–{100 * hi:.1f}%" for a, (lo, hi) in ret.items()))
    # TPU v6e-1 against the L4 run in ../jev, 26B-A4B only (the one size timed on both).
    out.append("")
    lat_l4 = [int(m) for m in re.findall(r"\| \w+ \| (\d+) \| \d+ \| \d+ \| \d+ \| \d+ \|", open(os.path.join(JEV, "2026-09-23-l4-latency", "LATENCY.md")).read())]
    lat_tpu = [v["latency"]["median_ms"] for v in sweep["26b-fp8"].values()]
    out.append(f"- 26B median ms per decision, one at a time on the host: L4 {min(lat_l4)}-{max(lat_l4)}, v6e-1 {min(lat_tpu):.0f}-{max(lat_tpu):.0f}; L4 / v6e-1 = {min(lat_l4) / max(lat_tpu):.1f} to {max(lat_l4) / min(lat_tpu):.1f} times")
    g6 = float(re.search(r"g6.xlarge ([\d.]+)", open(os.path.join(JEV, "2026-09-23-l4-awq", "evidence", "pricing.txt")).read()).group(1))
    out.append(f"- Hourly price: g6.xlarge ${g6:.4f} on demand; v6e-1 ${RATE['on-demand']:.2f} on demand ({RATE['on-demand'] / g6:.1f} times), ${RATE['flex-start']:.2f} flex-start ({RATE['flex-start'] / g6:.1f} times)")
    print("\n".join(out))


if __name__ == "__main__":
    main()
