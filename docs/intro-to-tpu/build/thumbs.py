import json, subprocess, sys, urllib.request, io, concurrent.futures as cf
from PIL import Image
pid, out = sys.argv[1], sys.argv[2]
tok = subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()
H = {"Authorization": f"Bearer {tok}", "x-goog-user-project": "aisprint-491218"}
def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=120).read()
base = f"https://slides.googleapis.com/v1/presentations/{pid}"
ids = [s["objectId"] for s in json.loads(get(base + "?fields=slides.objectId"))["slides"]]
only = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
ids = [i for i in ids if not only or i in only]
def thumb(i):
    u = json.loads(get(f"{base}/pages/{i}/thumbnail?thumbnailProperties.thumbnailSize=MEDIUM"))["contentUrl"]
    return Image.open(io.BytesIO(urllib.request.urlopen(u).read())).convert("RGB")
with cf.ThreadPoolExecutor(6) as ex: ims = list(ex.map(thumb, ids))
per = 6
for g in range(0, len(ims), per):
    grid = Image.new("RGB", (1600, 1350), "white")
    for k, im in enumerate(ims[g:g + per]):
        grid.paste(im.resize((800, 450)), ((k % 2) * 800, (k // 2) * 450))
    grid.save(f"{out}_{g // per + 1}.png")
print(len(ids), ids)
