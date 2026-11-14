"""Thinking and visible tokens per problem at each reasoning_effort, first 100 GSM8K problems (levels-sample/run.sh)."""
import glob, json, os, statistics
H = os.path.dirname(os.path.abspath(__file__))
out = []
for m in ("gemini-4-hydrogen", "gemini-3.8-flash", "gemini-4-carbon", "gemini-4-argon"):
    for e in ("none", "low", "medium", "unset"):
        rows = [json.loads(l) for l in open(f"{H}/{m}-{e}.jsonl")]
        rows = [r for r in rows if "id" in r]
        th = [r.get("reasoning_tokens") or 0 for r in rows]
        out.append(f"{m:18s} {e:7s} n {len(rows)}  right {sum(bool(r.get('right')) for r in rows)}  errors {sum('error' in r for r in rows)}"
                   f"  problems with thinking {sum(t > 0 for t in th)}  thinking tokens total {sum(th):,} mean {statistics.mean(th):.1f} median {statistics.median(th):.0f}"
                   f"  answer tokens mean {statistics.mean(r.get('completion_tokens') or 0 for r in rows):.1f}  median s {statistics.median(r['seconds'] for r in rows):.2f}")
    out.append("")
t = "\n".join(out); open(f"{H}/summary.txt", "w").write(t); print(t)
