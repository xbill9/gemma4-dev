"""Inside a vllm-tpu container: summarize a jax xplane.pb with xprof's hlo_stats.
anahlo.py <xplane.pb> <label> <devices> -> JSON: self time per HLO category and per op family, ms per device."""
import collections, json, re, sys
import xprof.convert.raw_to_tool_data as r
path, label, ndev = sys.argv[1], sys.argv[2], int(sys.argv[3])
data, _ = r.xspace_to_tool_data([path], "hlo_stats", {})
d = json.loads(data)
cols = [c.get("label") for c in d["cols"]]
ix = {c: i for i, c in enumerate(cols)}
cat, fam, occ = collections.Counter(), collections.Counter(), collections.Counter()
total = 0.0
for row in d["rows"]:
    v = [c.get("v") if c else None for c in row["c"]]
    t = v[ix["Total self time (us)"]] or 0.0
    total += t
    cat[v[ix["HLO op category"]]] += t
    name = re.sub(r"\.\d+$", "", v[ix["HLO op name"]])
    name = re.sub(r"-p_\d+.*", "", name)  # RPAd-p_32-... -> RPAd
    fam[name] += t
    occ[name] += v[ix["#Occurrences"]] or 0
ms = lambda us: round(us / 1e3 / ndev, 2)
out = {
    "label": label, "devices": ndev, "source": "xprof hlo_stats, self time summed over devices then divided by devices",
    "self_time_ms_per_device": ms(total),
    "by_category_ms_per_device": {k: ms(t) for k, t in cat.most_common()},
    "top_ops_ms_per_device": {k: {"ms": ms(t), "occurrences_per_device": occ[k] / ndev} for k, t in fam.most_common(15)},
}
print(json.dumps(out, indent=1))
