"""Inside the container: distinct attention-kernel names (RPA*) in an xplane, with self time and count."""
import collections, json, re, sys
import xprof.convert.raw_to_tool_data as r
d = json.loads(r.xspace_to_tool_data([sys.argv[1]], "hlo_stats", {})[0])
ix = {c.get("label"): i for i, c in enumerate(d["cols"])}
agg = collections.defaultdict(lambda: [0.0, 0])
for row in d["rows"]:
    v = [c.get("v") if c else None for c in row["c"]]
    n = re.sub(r"\.\d+$", "", v[ix["HLO op name"]])
    if n.startswith("RPA"):
        agg[n][0] += v[ix["Total self time (us)"]]; agg[n][1] += v[ix["#Occurrences"]]
print(json.dumps({sys.argv[2]: {k: {"self_ms_all_devices": round(t / 1e3, 1), "occurrences_all_devices": c} for k, (t, c) in sorted(agg.items(), key=lambda kv: -kv[1][0])}}, indent=1))
