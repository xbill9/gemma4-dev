"""EmbeddingGemma 2 retrieval on BEIR SciFact and NFCorpus (test split), against BM25.

usage: python3.13 -P beir_eval.py <beir_root> <out_dir>
  <beir_root> holds the unzipped BEIR datasets (scifact/, nfcorpus/), downloaded from
  public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/; sha256 in evidence/.

Fixed before the first model call (2026-10-08):
- Query  -> "task: search result | query: <query>"
- Doc    -> "title: <title or none> | text: <text>"   (the model's own Retrieval prompts)
- Full 768 d, cosine. Matryoshka 512/256/128 by truncating and renormalising the same vectors
  (shown equal to the server's `dimensions` to 4e-8 in the 2026-10-07 run).
- BM25: lowercase \\w+ tokens, no stemming or stopwords, k1=1.2, b=0.75, over title + text.
- Metrics: nDCG@10 with graded qrels (BEIR's definition), Recall@100. Queries without test qrels skipped.
"""
import json, math, os, re, sys, time, urllib.request, collections
from concurrent.futures import ThreadPoolExecutor
import numpy as np

U = os.environ.get("URL", "http://127.0.0.1:8000/v1/embeddings")
MODEL = "google/embeddinggemma-2"
QRY, DOC = "task: search result | query: ", "title: {} | text: "
SETS = ["scifact", "nfcorpus"]
DIMS = [768, 512, 256, 128]


def embed_batch(texts):
    body = json.dumps({"model": MODEL, "input": texts}).encode()
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(U, data=body, headers={"Content-Type": "application/json"}), timeout=900))
    except urllib.error.HTTPError:
        if len(texts) == 1:
            raise
        parts = [embed_batch([t]) for t in texts]
        return [v for vs, _ in parts for v in vs], sum(t for _, t in parts)
    return [d["embedding"] for d in sorted(r["data"], key=lambda d: d["index"])], r["usage"]["prompt_tokens"]


def embed(texts, batch=16, conc=4):
    chunks = [texts[i:i + batch] for i in range(0, len(texts), batch)]
    with ThreadPoolExecutor(conc) as ex:
        res = list(ex.map(embed_batch, chunks))
    m = np.asarray([v for vs, _ in res for v in vs], dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True), sum(t for _, t in res)


def ndcg10(ranked, rel):
    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall100(ranked, rel):
    pos = {d for d, g in rel.items() if g > 0}
    return len(pos & set(ranked[:100])) / len(pos) if pos else 0.0


TOK = re.compile(r"\w+")


def bm25_rank(doc_texts, queries, k1=1.2, b=0.75, top=100):
    toks = [TOK.findall(t.lower()) for t in doc_texts]
    N, avg = len(toks), sum(map(len, toks)) / len(toks)
    inv = collections.defaultdict(list)
    for i, ts in enumerate(toks):
        for w, c in collections.Counter(ts).items():
            inv[w].append((i, c))
    lens = [len(t) for t in toks]
    out = []
    for q in queries:
        s = collections.defaultdict(float)
        for w in set(TOK.findall(q.lower())):
            post = inv.get(w)
            if not post:
                continue
            idf = math.log(1 + (N - len(post) + 0.5) / (len(post) + 0.5))
            for i, c in post:
                s[i] += idf * c * (k1 + 1) / (c + k1 * (1 - b + b * lens[i] / avg))
        out.append([i for i, _ in sorted(s.items(), key=lambda x: -x[1])[:top]])
    return out


def main(root, out):
    os.makedirs(out, exist_ok=True)
    summary = {}
    for name in SETS:
        d = os.path.join(root, name)
        corpus = [json.loads(l) for l in open(os.path.join(d, "corpus.jsonl"))]
        queries = {q["_id"]: q["text"] for q in map(json.loads, open(os.path.join(d, "queries.jsonl")))}
        qrels = collections.defaultdict(dict)
        for line in list(open(os.path.join(d, "qrels", "test.tsv")))[1:]:
            qid, did, g = line.rstrip("\n").split("\t")
            qrels[qid][did] = int(g)
        qids = [q for q in qrels if q in queries]
        ids = [c["_id"] for c in corpus]
        doc_texts = [DOC.format(c.get("title") or "none") + c["text"] for c in corpus]
        t0 = time.time()
        D, dtok = embed(doc_texts)
        t_docs = time.time() - t0
        t0 = time.time()
        Q, qtok = embed([QRY + queries[q] for q in qids])
        t_q = time.time() - t0
        res = {"docs": len(ids), "queries": len(qids), "doc_tokens": dtok, "doc_embed_s": round(t_docs, 1),
               "doc_tok_per_s": round(dtok / t_docs), "query_embed_s": round(t_q, 1)}
        for dim in DIMS:
            Dd = D[:, :dim] / np.linalg.norm(D[:, :dim], axis=1, keepdims=True)
            Qd = Q[:, :dim] / np.linalg.norm(Q[:, :dim], axis=1, keepdims=True)
            t0 = time.time()
            S = Qd @ Dd.T
            top = np.argsort(-S, axis=1)[:, :100]
            res[f"search_ms_per_query_{dim}"] = round(1000 * (time.time() - t0) / len(qids), 3)
            ranked = [[ids[i] for i in row] for row in top]
            res[f"ndcg10_{dim}"] = round(float(np.mean([ndcg10(r, qrels[q]) for r, q in zip(ranked, qids)])), 4)
            res[f"recall100_{dim}"] = round(float(np.mean([recall100(r, qrels[q]) for r, q in zip(ranked, qids)])), 4)
            if dim == 768:
                with open(os.path.join(out, f"{name}-embed2-top10.jsonl"), "w") as f:
                    for q, r in zip(qids, ranked):
                        f.write(json.dumps({"qid": q, "top10": r[:10], "ndcg10": round(ndcg10(r, qrels[q]), 4)}) + "\n")
        bm = bm25_rank([(c.get("title") or "") + " " + c["text"] for c in corpus], [queries[q] for q in qids])
        bm = [[ids[i] for i in row] for row in bm]
        res["bm25_ndcg10"] = round(float(np.mean([ndcg10(r, qrels[q]) for r, q in zip(bm, qids)])), 4)
        res["bm25_recall100"] = round(float(np.mean([recall100(r, qrels[q]) for r, q in zip(bm, qids)])), 4)
        summary[name] = res
        print(name, json.dumps(res), file=sys.stderr, flush=True)
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print("| | " + " | ".join(SETS) + " |\n|---|" + "---|" * len(SETS))
    print("| BM25 nDCG@10 | " + " | ".join(f"{summary[s]['bm25_ndcg10']:.3f}" for s in SETS) + " |")
    for dim in DIMS:
        print(f"| embed2 {dim}-d nDCG@10 | " + " | ".join(f"{summary[s][f'ndcg10_{dim}']:.3f}" for s in SETS) + " |")
    print("| BM25 Recall@100 | " + " | ".join(f"{summary[s]['bm25_recall100']:.3f}" for s in SETS) + " |")
    print("| embed2 768-d Recall@100 | " + " | ".join(f"{summary[s]['recall100_768']:.3f}" for s in SETS) + " |")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
