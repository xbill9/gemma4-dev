"""EmbeddingGemma 2 on jev's four tasks; see ../PLAN.md, which fixes every choice here.

Writes evidence/predictions.jsonl (one row per example), evidence/summary.json and prints a
markdown table. Stdlib only.
"""
import json, os, random, statistics, sys, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
DATA = os.path.join(ROOT, "jev-tpu", "data")
JEV = os.path.join(ROOT, "jev-tpu", "results")
ARMS = {"e2b": "2026-09-24-v6e1-e2b", "e4b": "2026-09-24-v6e1-e4b", "12b": "2026-09-24-v6e1-12b",
        "26b-fp8": "2026-09-24-v6e1-26b-fp8"}
OUT = os.path.join(os.path.dirname(__file__), "..", "evidence")
U = os.environ.get("URL", "http://127.0.0.1:8000/v1/embeddings")
MODEL = "google/embeddinggemma-2"
CLS, QRY, DOC = "task: classification | query: ", "task: search result | query: ", "title: none | text: "
TASKS = ["sst2", "ag_news", "emotion", "irony"]
LABELS = {
    "sst2": {"negative": "A negative movie review.", "positive": "A positive movie review."},
    "ag_news": {"world": "World news about international politics and conflicts.", "sports": "Sports news.",
                "business": "Business and financial news.", "scitech": "Science and technology news."},
    "emotion": {k: f"The writer feels {k}." for k in ("sadness", "joy", "love", "anger", "fear", "surprise")},
    "irony": {"yes": "An ironic or sarcastic tweet.", "no": "A sincere tweet that is not ironic."},
}
DRAWS, N_TRAIN, BOOT = 200, 50, 2000


def embed(texts, batch=16):
    out = []
    for i in range(0, len(texts), batch):
        body = json.dumps({"model": MODEL, "input": texts[i:i + batch]}).encode()
        r = json.load(urllib.request.urlopen(urllib.request.Request(U, data=body, headers={"Content-Type": "application/json"}), timeout=600))
        out += [d["embedding"] for d in sorted(r["data"], key=lambda d: d["index"])]
    return [unit(v) for v in out]


def unit(v):
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def nearest(v, cents):
    return max(cents, key=lambda k: dot(v, cents[k]))


def centroids(vecs, golds, idx):
    sums = {}
    for i in idx:
        s = sums.setdefault(golds[i], [0.0] * len(vecs[i]))
        for j, x in enumerate(vecs[i]):
            s[j] += x
    return {k: unit(s) for k, s in sums.items()}


def boot_diff(a, b, seed):
    r, n = random.Random(seed), len(a)
    ds = sorted(sum(a[i] - b[i] for i in (r.randrange(n) for _ in range(n))) / n for _ in range(BOOT))
    return ds[int(0.025 * BOOT)], ds[int(0.975 * BOOT) - 1]


def main():
    os.makedirs(OUT, exist_ok=True)
    summary, rows = {}, []
    for t in TASKS:
        recs = [json.loads(l) for l in open(os.path.join(DATA, f"{t}.jsonl"))]
        ids, golds = [r["id"] for r in recs], [r["gold"] for r in recs]
        names = list(LABELS[t])
        v_cls = embed([CLS + r["text"] for r in recs])
        v_qry = embed([QRY + r["text"] for r in recs])
        l_cls = dict(zip(names, embed([CLS + LABELS[t][k] for k in names])))
        l_doc = dict(zip(names, embed([DOC + LABELS[t][k] for k in names])))
        pred = {"A": [nearest(v, l_cls) for v in v_cls], "A2": [nearest(v, l_doc) for v in v_qry]}
        # C: leave-one-out centroid over the other 299
        sums = {}
        for v, g in zip(v_cls, golds):
            s = sums.setdefault(g, [0.0] * len(v))
            for j, x in enumerate(v):
                s[j] += x
        pred["C"] = []
        for v, g in zip(v_cls, golds):
            cents = {k: unit([s - x for s, x in zip(sums[k], v)] if k == g else sums[k]) for k in sums}
            pred["C"].append(nearest(v, cents))
        correct = {k: [int(p == g) for p, g in zip(ps, golds)] for k, ps in pred.items()}
        # B: 50 random labels, nearest centroid, scored on the other 250
        rb, accs = random.Random(f"B-{t}"), []
        for _ in range(DRAWS):
            train = rb.sample(range(len(recs)), N_TRAIN)
            cents, held = centroids(v_cls, golds, train), set(range(len(recs))) - set(train)
            accs.append(sum(nearest(v_cls[i], cents) == golds[i] for i in held) / len(held))
        accs.sort()
        # jev arms, non-reversed reads, argmax over names
        jev = {}
        for arm, d in ARMS.items():
            by = {}
            for l in open(os.path.join(JEV, d, f"autoregressive-{t}.jsonl")):
                r = json.loads(l)
                p = r["reads"][0]["probs"]
                by[r["id"]] = int(r["names"][max(range(len(p)), key=p.__getitem__)] == r["gold"])
            assert set(by) == set(ids), f"{arm}/{t}: id set differs"
            jev[arm] = [by[i] for i in ids]
        s = {"n": len(recs), **{k: sum(c) / len(c) for k, c in correct.items()},
             "B_mean": statistics.mean(accs), "B_lo": accs[int(0.025 * DRAWS)], "B_hi": accs[int(0.975 * DRAWS) - 1],
             "jev": {a: sum(c) / len(c) for a, c in jev.items()},
             "diff": {f"{k}-{a}": [sum(correct[k][i] - jev[a][i] for i in range(len(ids))) / len(ids),
                                   *boot_diff(correct[k], jev[a], f"{t}-{k}-{a}")]
                      for k in ("A", "A2", "C") for a in ARMS}}
        summary[t] = s
        for i, rid in enumerate(ids):
            rows.append({"id": rid, "task": t, "gold": golds[i], **{f"pred_{k}": pred[k][i] for k in pred},
                         **{f"jev_{a}_correct": jev[a][i] for a in ARMS}})
        print(f"{t}: A {s['A']:.3f}  A2 {s['A2']:.3f}  B {s['B_mean']:.3f}  C {s['C']:.3f}", file=sys.stderr, flush=True)
    with open(os.path.join(OUT, "predictions.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    pc = lambda x: f"{100 * x:.1f}%"
    print("| | " + " | ".join(TASKS) + " |\n|---|" + "---|" * len(TASKS))
    for k, lab in (("A", "embed2 A, zero-shot"), ("A2", "embed2 A2, zero-shot, retrieval prompts"),
                   ("B", "embed2 B, 50 labels, centroid (mean, 95% of draws)"), ("C", "embed2 C, 299 labels, LOO centroid")):
        cells = [f"{pc(summary[t]['B_mean'])} ({pc(summary[t]['B_lo'])}–{pc(summary[t]['B_hi'])})" if k == "B" else pc(summary[t][k]) for t in TASKS]
        print(f"| {lab} | " + " | ".join(cells) + " |")
    for a in ARMS:
        print(f"| jev {a}, v6e-1 | " + " | ".join(pc(summary[t]["jev"][a]) for t in TASKS) + " |")
    print("\nPaired difference, embed2 − jev (95% range):\n\n| | " + " | ".join(TASKS) + " |\n|---|" + "---|" * len(TASKS))
    for k in ("A", "A2", "C"):
        for a in ARMS:
            print(f"| {k} − {a} | " + " | ".join("{:+.1f} ({:+.1f} to {:+.1f})".format(*(100 * x for x in summary[t]["diff"][f"{k}-{a}"])) for t in TASKS) + " |")


if __name__ == "__main__":
    main()
