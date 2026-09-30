#!/usr/bin/env python3
"""strip_tied_head.py -- copy a single-file Hugging Face checkpoint to gs:// without `lm_head.weight`.

  strip_tied_head.py <hf-repo> <gs://dest-dir>

For checkpoints whose config ties the embeddings but which also store `lm_head.weight`: vLLM's JAX
path loads the stored head, so the copy costs its full size in HBM. The script refuses unless the
config says tied and the stored head is byte-identical (sha256) to `embed_tokens`; every other tensor
is streamed through unchanged, so the served values are exactly the source's. Nothing lands on local
disk: model.safetensors is piped into `gcloud storage cp -`. The small files are copied as they are.
Writes <dest>/STRIPPED.json recording the source revision, both hashes and what was removed.
"""

import hashlib
import json
import struct
import subprocess
import sys
import urllib.request

HEAD = "lm_head.weight"
EMBED = "model.language_model.embed_tokens.weight"
SMALL = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "processor_config.json",
    "recipe.yaml",
    "README.md",
)


def get(url, start=None, end=None):
    h = {"Range": f"bytes={start}-{end}"} if start is not None else {}
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=600)


def stream(url, start, end, sink, hasher=None):
    r = get(url, start, end)
    while chunk := r.read(1 << 24):
        if hasher:
            hasher.update(chunk)
        if sink:
            sink.write(chunk)


def main(repo, dest):
    info = json.load(get(f"https://huggingface.co/api/models/{repo}"))
    base = f"https://huggingface.co/{repo}/resolve/{info['sha']}/"
    cfg = json.load(get(base + "config.json"))
    if not (cfg.get("tie_word_embeddings") and cfg.get("text_config", {}).get("tie_word_embeddings", True)):
        sys.exit("config does not tie the embeddings; refusing")
    url = base + "model.safetensors"
    n = struct.unpack("<Q", get(url, 0, 7).read())[0]
    header = json.loads(get(url, 8, 8 + n - 1).read())
    meta = header.pop("__metadata__", None)
    hashes = {}
    for k in (EMBED, HEAD):
        a, b = header[k]["data_offsets"]
        h = hashlib.sha256()
        stream(url, 8 + n + a, 8 + n + b - 1, None, h)
        hashes[k] = h.hexdigest()
    if hashes[EMBED] != hashes[HEAD]:
        sys.exit(f"stored head differs from embed_tokens: {hashes}")
    # New header: every tensor but the head, packed in source order.
    keep = sorted((k for k in header if k != HEAD), key=lambda k: header[k]["data_offsets"][0])
    new, off = {}, 0
    for k in keep:
        a, b = header[k]["data_offsets"]
        new[k] = dict(header[k], data_offsets=[off, off + b - a])
        off += b - a
    if meta is not None:
        new["__metadata__"] = meta
    hb = json.dumps(new, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    cp = subprocess.Popen(["gcloud", "storage", "cp", "-q", "-", f"{dest}/model.safetensors"], stdin=subprocess.PIPE)
    out = hashlib.sha256()

    class Tee:
        def write(self, c):
            out.update(c)
            cp.stdin.write(c)

    tee = Tee()
    tee.write(struct.pack("<Q", len(hb)) + hb)
    # Contiguous runs of kept tensors go as one ranged request each.
    runs, cur = [], None
    for k in keep:
        a, b = header[k]["data_offsets"]
        if cur and cur[1] == a:
            cur[1] = b
        else:
            cur = [a, b]
            runs.append(cur)
    for a, b in runs:
        stream(url, 8 + n + a, 8 + n + b - 1, tee)
    cp.stdin.close()
    if cp.wait():
        sys.exit("upload failed")
    for f in SMALL:
        if any(s["rfilename"] == f for s in info["siblings"]):
            subprocess.run(
                ["gcloud", "storage", "cp", "-q", "-", f"{dest}/{f}"], input=get(base + f).read(), check=True
            )
    rec = {
        "source": repo,
        "revision": info["sha"],
        "removed": [HEAD],
        "reason": "tied; byte-identical to embed_tokens",
        "sha256": hashes,
        "tensors_kept": len(keep),
        "bytes_written": 8 + len(hb) + off,
        "model.safetensors_sha256": out.hexdigest(),
    }
    subprocess.run(
        ["gcloud", "storage", "cp", "-q", "-", f"{dest}/STRIPPED.json"],
        input=json.dumps(rec, indent=1).encode(),
        check=True,
    )
    print(json.dumps(rec))


if __name__ == "__main__":
    main(*sys.argv[1:3])
