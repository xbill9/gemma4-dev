"""Upload each new build: weights first, then card and evidence, one repo at a time, smallest first."""
import glob, os, time
from huggingface_hub import HfApi

api = HfApi()
EXTRA = "/mnt/scratch/upload-extra"
SRC = {os.path.basename(d): d for d in glob.glob("/mnt/scratch/hf-cache/local/*") + glob.glob("/mnt/scratch/gguf/*")}
names = sorted(os.listdir(EXTRA), key=lambda n: sum(os.path.getsize(f) for f in glob.glob(SRC[n] + "/*")))
for n in names:
    repo = f"xbill9/{n}"
    t = time.time()
    api.create_repo(repo, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=repo, folder_path=SRC[n], ignore_patterns=["README.md", ".*"],
                      commit_message="Upload weights")
    api.upload_folder(repo_id=repo, folder_path=f"{EXTRA}/{n}", commit_message="Add model card, build scripts and evidence")
    remote = {f.path: f.size for f in api.list_repo_tree(repo, recursive=True) if hasattr(f, "size")}
    local = {os.path.relpath(f, SRC[n]): os.path.getsize(f) for f in glob.glob(SRC[n] + "/*")
             if os.path.basename(f) != "README.md" and not os.path.basename(f).startswith(".")}
    bad = [k for k, v in local.items() if remote.get(k) != v]
    print(f"{time.strftime('%H:%M:%S')} {repo}: {len(remote)} files, {sum(local.values()) / 1e9:.2f} GB, "
          f"{time.time() - t:.0f}s, {'SIZES MATCH' if not bad else 'MISMATCH ' + str(bad)}", flush=True)
print("all done", flush=True)
