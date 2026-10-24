import json, subprocess, sys, urllib.request, urllib.error
pid, path = sys.argv[1], sys.argv[2]
reqs = json.load(open(path))
tok = subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()
req = urllib.request.Request(f"https://slides.googleapis.com/v1/presentations/{pid}:batchUpdate",
    data=json.dumps({"requests": reqs}).encode(), method="POST",
    headers={"Authorization": f"Bearer {tok}", "x-goog-user-project": "aisprint-491218", "Content-Type": "application/json"})
try:
    r = json.load(urllib.request.urlopen(req, timeout=300)); print("ok", len(r.get("replies", [])))
except urllib.error.HTTPError as e:
    msg = json.load(e)["error"]["message"]; print("ERR", msg[:600])
    import re; m = re.search(r"requests\[(\d+)\]", msg)
    if m: print(json.dumps(reqs[int(m.group(1))])[:800])
