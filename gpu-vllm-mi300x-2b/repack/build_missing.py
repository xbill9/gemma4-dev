"""Build the xbill9 Gemma 4 repacks that are not on Hugging Face yet, on the MI300X droplet.

Run ON THE DROPLET, from a synced copy of gemma4-dev at /root/gemma4-dev:

    python3 /root/gemma4-dev/gpu-vllm-mi300x-2b/repack/build_missing.py --image <pinned digest> [build ...]

Every step runs inside the pinned vLLM image, which carries numpy and ml_dtypes, with
no GPU attached: the builds are CPU and disk only. Sources are downloaded from Hugging
Face into /root/build/src, outputs land in /root/build/out/<repo name>, and each step
writes its log to /root/build/logs/<repo name>.log. A step whose output carries a
`.built` marker is skipped, so re-running resumes. Nothing is uploaded.

26B-A4B's experts are fused 3-D banks in the source; `w8a8_from_qat.py` and
`fp8_text.py` split them into the per-expert modules the W4A16 repack uses. Its exact
GGUF maps Google's fused expert banks onto the source's (`gguf_exact.py`).

Stdlib only (the droplet's own python3).
"""

import argparse
import concurrent.futures
import os
import subprocess
import sys
import time

W = "/root/build"
T = "/root/gemma4-dev"
FP8 = f"{T}/gpu-vllm-mi300x-2b/repack/fp8_text.py"
GG = f"{T}/gpu-vllm-mi300x-2b/repack/gguf_exact.py"
J = f"{T}/jev-tpu-31b"
S = f"{W}/src"
OUT = f"{W}/out"

# local name -> (repo, allow_patterns or None for everything)
SOURCES = {
    "E2B-unq": ("google/gemma-4-E2B-it-qat-q4_0-unquantized", None),
    "E2B-ct-text": ("xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text", ["config.json", "*.index.json"]),
    "E2B-ct-text-emb4": ("xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4", None),
    "E4B-unq": ("google/gemma-4-E4B-it-qat-q4_0-unquantized", None),
    "E4B-ct-text": ("xbill9/gemma-4-E4B-it-qat-q4_0-w4a16-ct-text", None),
    "E4B-ct-text-emb4": ("xbill9/gemma-4-E4B-it-qat-q4_0-w4a16-ct-text-emb4", None),
    "12B-unq": ("google/gemma-4-12B-it-qat-q4_0-unquantized", None),
    "12B-ct-text-emb4": ("xbill9/gemma-4-12B-it-qat-q4_0-w4a16-ct-text-emb4", None),
    "12B-gguf": ("google/gemma-4-12B-it-qat-q4_0-gguf", ["gemma-4-12b-it-qat-q4_0.gguf"]),
    "26B-unq": ("google/gemma-4-26B-A4B-it-qat-q4_0-unquantized", None),
    "26B-ct-text": ("xbill9/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct-text", ["config.json", "*.index.json"]),
    "26B-ct-text-emb4": ("xbill9/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct-text-emb4", None),
    "26B-gguf": ("google/gemma-4-26B-A4B-it-qat-q4_0-gguf", ["gemma-4-26B_q4_0-it.gguf"]),
    "31B-unq": ("google/gemma-4-31B-it-qat-q4_0-unquantized", None),
    "31B-ct-text": ("xbill9/gemma-4-31B-it-qat-q4_0-w4a16-ct-text", ["config.json", "*.index.json"]),
    "31B-ct-text-emb4": ("xbill9/gemma-4-31B-it-qat-q4_0-w4a16-ct-text-emb4", None),
    "31B-gguf": ("google/gemma-4-31B-it-qat-q4_0-gguf", ["gemma-4-31B_q4_0-it.gguf"]),
}


def fp8(sub: str, *args: str, fnuz: bool = False) -> list[str]:
    return ["python3", FP8, sub, *args] + (["--fnuz"] if fnuz else [])


# repo name -> (sources needed, steps needing to be built first, commands)
BUILDS = {
    "gemma-4-E2B-it-qat-q4_0-fp8fnuz-text": (
        ["E2B-unq", "E2B-ct-text"],
        [],
        [
            fp8("build", f"{S}/E2B-unq", f"{S}/E2B-ct-text/config.json",
                f"{S}/E2B-ct-text/model.safetensors.index.json", "{out}", fnuz=True),
            fp8("verify", f"{S}/E2B-unq", "{out}", fnuz=True),
        ],
    ),
    "gemma-4-E2B-it-qat-q4_0-fp8fnuz-text-emb4": (
        ["E2B-unq", "E2B-ct-text-emb4"],
        [],
        [
            fp8("build-on", f"{S}/E2B-unq", f"{S}/E2B-ct-text-emb4", "{out}", fnuz=True),
            fp8("verify", f"{S}/E2B-unq", "{out}", fnuz=True),
        ],
    ),
    "gemma-4-E4B-it-qat-q4_0-w4a16-ct-text-ple4": (
        ["E4B-ct-text"],
        [],
        [["python3", f"{J}/embed_int4.py", f"{S}/E4B-ct-text", "{out}"]],
    ),
    "gemma-4-E4B-it-qat-q4_0-fp8-text-emb4": (
        ["E4B-unq", "E4B-ct-text-emb4"],
        [],
        [
            fp8("build-on", f"{S}/E4B-unq", f"{S}/E4B-ct-text-emb4", "{out}"),
            fp8("verify", f"{S}/E4B-unq", "{out}"),
        ],
    ),
    "gemma-4-12B-it-qat-q4_0-fp8-text-emb4": (
        ["12B-unq", "12B-ct-text-emb4"],
        [],
        [
            fp8("build-on", f"{S}/12B-unq", f"{S}/12B-ct-text-emb4", "{out}"),
            fp8("verify", f"{S}/12B-unq", "{out}"),
        ],
    ),
    "gemma-4-12B-it-qat-q4_0-exact-gguf": (
        ["12B-unq", "12B-gguf"],
        [],
        [["python3", GG, f"{S}/12B-unq", f"{S}/12B-gguf/gemma-4-12b-it-qat-q4_0.gguf",
          "{out}/gemma-4-12B-it-q4_0-exact.gguf"]],
    ),
    "gemma-4-31B-it-qat-w8a8-int8": (
        ["31B-unq"],
        [],
        [["python3", f"{J}/w8a8_from_qat.py", f"{S}/31B-unq", "{out}"]],
    ),
    "gemma-4-31B-it-qat-w8a8-int8-emb4": (
        ["31B-ct-text-emb4"],
        ["gemma-4-31B-it-qat-w8a8-int8"],
        [["python3", f"{J}/w8a8_emb4.py", f"{OUT}/gemma-4-31B-it-qat-w8a8-int8", f"{S}/31B-ct-text-emb4", "{out}"]],
    ),
    "gemma-4-31B-it-qat-q4_0-fp8-text": (
        ["31B-unq", "31B-ct-text"],
        [],
        [
            fp8("build", f"{S}/31B-unq", f"{S}/31B-ct-text/config.json",
                f"{S}/31B-ct-text/model.safetensors.index.json", "{out}"),
            fp8("verify", f"{S}/31B-unq", "{out}"),
        ],
    ),
    "gemma-4-31B-it-qat-q4_0-fp8-text-emb4": (
        ["31B-unq", "31B-ct-text-emb4"],
        [],
        [
            fp8("build-on", f"{S}/31B-unq", f"{S}/31B-ct-text-emb4", "{out}"),
            fp8("verify", f"{S}/31B-unq", "{out}"),
        ],
    ),
    "gemma-4-26B-A4B-it-qat-w8a8-int8": (
        ["26B-unq"],
        [],
        [["python3", f"{J}/w8a8_from_qat.py", f"{S}/26B-unq", "{out}"]],
    ),
    "gemma-4-26B-A4B-it-qat-w8a8-int8-emb4": (
        ["26B-ct-text-emb4"],
        ["gemma-4-26B-A4B-it-qat-w8a8-int8"],
        [["python3", f"{J}/w8a8_emb4.py", f"{OUT}/gemma-4-26B-A4B-it-qat-w8a8-int8", f"{S}/26B-ct-text-emb4", "{out}"]],
    ),
    "gemma-4-26B-A4B-it-qat-q4_0-fp8-text": (
        ["26B-unq", "26B-ct-text"],
        [],
        [
            fp8("build", f"{S}/26B-unq", f"{S}/26B-ct-text/config.json",
                f"{S}/26B-ct-text/model.safetensors.index.json", "{out}"),
            fp8("verify", f"{S}/26B-unq", "{out}"),
        ],
    ),
    "gemma-4-26B-A4B-it-qat-q4_0-fp8-text-emb4": (
        ["26B-unq", "26B-ct-text-emb4"],
        [],
        [
            fp8("build-on", f"{S}/26B-unq", f"{S}/26B-ct-text-emb4", "{out}"),
            fp8("verify", f"{S}/26B-unq", "{out}"),
        ],
    ),
    "gemma-4-26B-A4B-it-qat-q4_0-exact-gguf": (
        ["26B-unq", "26B-gguf"],
        [],
        [["python3", GG, f"{S}/26B-unq", f"{S}/26B-gguf/gemma-4-26B_q4_0-it.gguf",
          "{out}/gemma-4-26B-A4B-it-q4_0-exact.gguf"]],
    ),
    "gemma-4-31B-it-qat-q4_0-w4a16-ct": (
        ["31B-unq"],
        [],
        [
            ["python3", f"{J}/repack_q4_0.py", "repack", f"{S}/31B-unq", "{out}", "--workers", "2"],
            ["python3", f"{J}/repack_q4_0.py", "verify", f"{S}/31B-unq", "{out}"],
        ],
    ),
    "gemma-4-31B-it-qat-q4_0-exact-gguf": (
        ["31B-unq", "31B-gguf"],
        [],
        [["python3", GG, f"{S}/31B-unq", f"{S}/31B-gguf/gemma-4-31B_q4_0-it.gguf",
          "{out}/gemma-4-31B-it-q4_0-exact.gguf"]],
    ),
}


CPUS: list[str] = []  # --cpus N caps every container, e.g. while a benchmark shares the host


def docker(image: str, argv: list[str]) -> list[str]:
    return [
        "docker", "run", "--rm", "--network", "host", *CPUS,
        "-v", f"{T}:{T}", "-v", f"{W}:{W}", "-e", f"HF_HOME={W}/hf", "-w", W,
        "--entrypoint", argv[0], image, *argv[1:],
    ]  # fmt: skip


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def download(image: str, name: str) -> None:
    repo, patterns = SOURCES[name]
    dest = f"{S}/{name}"
    if os.path.exists(f"{dest}/.downloaded"):
        return
    code = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download({repo!r}, local_dir={dest!r}, allow_patterns={patterns!r})"
    )
    with open(f"{W}/logs/download-{name}.log", "w") as f:
        subprocess.run(docker(image, ["python3", "-c", code]), stdout=f, stderr=subprocess.STDOUT, check=True)
    open(f"{dest}/.downloaded", "w").close()
    log(f"downloaded {repo}")


def build(image: str, name: str) -> str:
    out = f"{OUT}/{name}"
    if os.path.exists(f"{out}/.built"):
        return f"{name}: already built"
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    with open(f"{W}/logs/{name}.log", "w") as f:
        for argv in BUILDS[name][2]:
            argv = [a.replace("{out}", out) for a in argv]
            f.write(f"$ {' '.join(argv)}\n")
            f.flush()
            rc = subprocess.run(docker(image, argv), stdout=f, stderr=subprocess.STDOUT).returncode
            if rc:
                return f"{name}: FAILED (exit {rc}) after {time.time() - t0:.0f}s, see logs/{name}.log"
    open(f"{out}/.built", "w").close()
    return f"{name}: built in {time.time() - t0:.0f}s"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--image", required=True)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--cpus", help="cap each container at this many CPUs")
    p.add_argument("--download-only", action="store_true")
    p.add_argument("builds", nargs="*", help="default: all")
    a = p.parse_args()
    if a.cpus:
        CPUS[:] = ["--cpus", a.cpus]
    todo = a.builds or list(BUILDS)
    for d in (S, OUT, f"{W}/logs", f"{W}/hf"):
        os.makedirs(d, exist_ok=True)

    needed = sorted({s for b in todo for s in BUILDS[b][0]})
    with concurrent.futures.ThreadPoolExecutor(1 if a.cpus else 4) as ex:
        list(ex.map(lambda n: download(a.image, n), needed))
    if a.download_only:
        return 0

    done: set[str] = set()
    pending = list(todo)
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as ex:
        running: dict = {}
        while pending or running:
            for b in [b for b in pending if all(d in done for d in BUILDS[b][1] if d in todo)]:
                if len(running) < a.jobs:
                    pending.remove(b)
                    running[ex.submit(build, a.image, b)] = b
                    log(f"start {b}")
            fin, _ = concurrent.futures.wait(running, return_when=concurrent.futures.FIRST_COMPLETED)
            for fu in fin:
                b = running.pop(fu)
                log(fu.result())
                if "FAILED" not in fu.result():
                    done.add(b)
                else:
                    pending = [p for p in pending if b not in BUILDS[p][1]]
    return 0


if __name__ == "__main__":
    sys.exit(main())
