"""Ratios for benchmarks/runs/<run-id>/results.json, computed here: python3 summarize.py <run-id>"""
import json, sys
r = json.load(open(f"benchmarks/runs/{sys.argv[1]}/results.json"))
ea, ga, ge, eg = r["embed_alone"], r["gen_alone"], r["gen_under_embed"], r["embed_under_gen"]
print("## Embedding alone (generator loaded, idle)\n\n| requests | input tokens | req/s | input tok/s | median latency ms |\n|---:|---:|---:|---:|---:|")
for k, v in ea.items():
    c, n = k[1:].split("-in")
    print(f"| {c} | {n} | {v['request_throughput']:,.1f} | {v['total_token_throughput']:,.0f} | {v['median_e2el_ms']:,.1f} |")
print("\n## Generator (12B fp8) alone and under a steady 128-way, 1,024-token embedding load\n\n| requests | alone out tok/s | under load | ratio | median TPOT alone / under load ms |\n|---:|---:|---:|---:|---|")
for c in ga:
    a, b = ga[c], ge[c]
    print(f"| {c[1:]} | {a['output_throughput']:,.0f} | {b['output_throughput']:,.0f} | {b['output_throughput']/a['output_throughput']:.2f} | {a['median_tpot_ms']:.1f} / {b['median_tpot_ms']:.1f} |")
print("\n## Embedding alone and under a steady 64-way generation load (1,024-token inputs)\n\n| requests | alone req/s | under load | ratio | median latency alone / under load ms |\n|---:|---:|---:|---:|---|")
for k, b in eg.items():
    a = ea[k.replace("-under-gen", "")] if k in ea else ea[k]
    print(f"| {k[1:].split('-in')[0]} | {a['request_throughput']:,.1f} | {b['request_throughput']:,.1f} | {b['request_throughput']/a['request_throughput']:.2f} | {a['median_e2el_ms']:,.0f} / {b['median_e2el_ms']:,.0f} |")
g, e = ge["c64"]["output_throughput"] / ga["c64"]["output_throughput"], eg["c128-in1024"]["request_throughput"] / ea["c128-in1024"]["request_throughput"]
print(f"\nShare of each server's alone throughput kept at the heaviest load: generator c64 {g:.2f}, embedder c128 {e:.2f}, sum {g+e:.2f}")
print("\nOverlap of background load and measurement (from result timestamps):", json.dumps(r.get("overlap", {}), indent=1))
