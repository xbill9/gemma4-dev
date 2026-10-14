import json, urllib.request, numpy as np, base64, io
from PIL import Image, ImageDraw
U = "http://127.0.0.1:8000/v1/embeddings"; M = "google/embeddinggemma-2"
def post(body):
    req = urllib.request.Request(U, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req))
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode()[:200]}
q = "task: search result | query: a red circle"
full = np.array(post({"model": M, "input": q})["data"][0]["embedding"])
for d in (256, 128):
    t = np.array(post({"model": M, "input": q, "dimensions": d})["data"][0]["embedding"])
    p = full[:d] / np.linalg.norm(full[:d])
    print(f"dimensions={d}: len {len(t)} norm {np.linalg.norm(t):.6f} 1-cos(prefix-renorm) {1-float(t@p):.1e}")
print("dimensions=300:", post({"model": M, "input": q, "dimensions": 300}).get("error", "ACCEPTED?")[:140])
def img(color, shape):
    im = Image.new("RGB", (224, 224), "white"); d = ImageDraw.Draw(im)
    (d.ellipse if shape == "circle" else d.rectangle)((40, 40, 184, 184), fill=color)
    b = io.BytesIO(); im.save(b, "PNG"); return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
embs = {}
for name, (c, s) in {"red circle": ("red", "circle"), "blue square": ("blue", "square")}.items():
    r = post({"model": M, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": img(c, s)}}]}]})
    if "error" in r: print("image:", r["error"]); raise SystemExit
    embs[name] = np.array(r["data"][0]["embedding"]); print(f"image {name}: dim {len(embs[name])} prompt_tokens {r['usage']['prompt_tokens']}")
for t in ("a red circle", "a blue square"):
    e = np.array(post({"model": M, "input": "task: search result | query: " + t})["data"][0]["embedding"])
    print(f"query '{t}':", {k: round(float(e @ v), 4) for k, v in embs.items()})
