"""Recompute nDCG@10 with linear gain (BEIR / pytrec_eval convention) for embed2 768-d and BM25.
usage: python3 -P ndcg_linear.py <beir_root> <evidence_dir>"""
import collections, json, math, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from beir_eval import bm25_rank  # BM25 is CPU-only; no model call

def ndcg(r, rel):
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(r[:10]))
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(sorted(rel.values(), reverse=True)[:10]))
    return dcg / idcg if idcg else 0.0

root, ev = sys.argv[1], sys.argv[2]
out = {}
for s in ("scifact", "nfcorpus"):
    d = os.path.join(root, s)
    q = {x["_id"]: x["text"] for x in map(json.loads, open(os.path.join(d, "queries.jsonl")))}
    qrels = collections.defaultdict(dict)
    for line in list(open(os.path.join(d, "qrels", "test.tsv")))[1:]:
        a, b, g = line.rstrip("\n").split("\t"); qrels[a][b] = int(g)
    qids = [x for x in qrels if x in q]
    emb = {r["qid"]: r["top10"] for r in map(json.loads, open(os.path.join(ev, f"{s}-embed2-top10.jsonl")))}
    corpus = [json.loads(l) for l in open(os.path.join(d, "corpus.jsonl"))]
    ids = [c["_id"] for c in corpus]
    bm = bm25_rank([(c.get("title") or "") + " " + c["text"] for c in corpus], [q[x] for x in qids], top=10)
    out[s] = {"embed2_768_ndcg10_linear": round(sum(ndcg(emb[x], qrels[x]) for x in qids) / len(qids), 4),
              "bm25_ndcg10_linear": round(sum(ndcg([ids[i] for i in r], qrels[x]) for r, x in zip(bm, qids)) / len(qids), 4)}
print(json.dumps(out, indent=1))
