#!/usr/bin/env python3
"""Download Hugging Face repos at pinned revisions during the image build.

  fetch.py dir   REPO REVISION DEST [PATTERN ...]   files into DEST (the checkpoint as a directory)
  fetch.py cache REPO REVISION HF_HOME [PATTERN ...] the hub cache layout under HF_HOME/hub, with
                                                    refs/main pointing at REVISION, so code that
                                                    loads the repo by name finds it with HF_HUB_OFFLINE=1
  fetch.py file  REPO REVISION FILENAME DEST SHA256  one file, checked against its SHA-256

The token, when one is needed, comes from the build secret `hf` as HF_TOKEN.
"""

import hashlib
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

DEFAULT = ["*.json", "*.jinja", "*.safetensors", "*.model", "*.txt"]


def size(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())


def main(a):
    mode, repo, rev = a[0], a[1], a[2]
    tok = os.environ.get("HF_TOKEN") or None
    if mode == "dir":
        dest, pats = a[3], a[4:] or DEFAULT
        snapshot_download(repo, revision=rev, local_dir=dest, allow_patterns=pats, token=tok)
        print(f"{repo}@{rev} -> {dest}: {size(dest)} bytes")
    elif mode == "cache":
        home, pats = a[3], a[4:] or DEFAULT
        hub = Path(home) / "hub"
        snap = snapshot_download(repo, revision=rev, cache_dir=hub, allow_patterns=pats, token=tok)
        if Path(snap).name != rev:
            sys.exit(f"{repo}: snapshot {Path(snap).name} is not the pinned revision {rev}")
        ref = hub / f"models--{repo.replace('/', '--')}" / "refs" / "main"
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(rev)
        print(f"{repo}@{rev} -> {snap}: {size(Path(snap).resolve().parent.parent / 'blobs')} bytes")
    elif mode == "file":
        fn, dest, want = a[3], a[4], a[5]
        p = hf_hub_download(repo, fn, revision=rev, local_dir=dest, token=tok)
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        if h.hexdigest() != want:
            sys.exit(f"{p}: SHA-256 {h.hexdigest()} is not {want}")
        print(f"{repo}@{rev}/{fn} -> {p}: {os.path.getsize(p)} bytes, SHA-256 ok")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
