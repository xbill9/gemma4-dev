import sys, json, urllib.request, io, pyarrow.parquet as pq, datetime
repo, config, split, out = sys.argv[1:5]
urls = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo}/parquet/{config}/{split}"))
n = 0
def default(o):
    if isinstance(o, (datetime.date, datetime.datetime)): return o.isoformat()
    if isinstance(o, bytes): return o.decode("utf-8", "replace")
    raise TypeError(type(o))
with open(out, "w") as f:
    for u in urls:
        t = pq.read_table(io.BytesIO(urllib.request.urlopen(u).read()))
        for row in t.to_pylist():
            f.write(json.dumps(row, ensure_ascii=False, default=default) + "\n"); n += 1
print(repo, config, split, n, "rows ->", out)
