"""ARITHMETIC, not a measurement: bytes one text decode step must read, per checkpoint.

Reads only the safetensors headers (tensor names, dtypes, shapes), so it needs no
GPU and no torch. Classifies every tensor, then counts the ones a batch-1 text
decode step reads in full:

  read      decoder layers (attention, MLP, norms, per-layer projections) and the
            tied embedding, which is the LM head — 262,144 x 1536 read every token
  lookup    embed_tokens_per_layer (PLE) — one row per layer per token, not the table
  idle      vision and audio towers — not touched by a text-only decode

Bytes are counted at the dtype the engine RUNS, not the dtype on disk: the
server is started with --dtype float16, so every bf16 tensor becomes 2 bytes of
fp16 either way, and packed int4 weights and their scales are counted at their
stored width. The bound is then bandwidth / bytes; 277.0 GB/s is the T4's
measured streaming read from @HARDWARE.md, 320.1 GB/s its theoretical peak.

    python3 bytes_per_token.py <snapshot-dir> [<snapshot-dir> ...]
"""

import json
import struct
import sys
from collections import Counter
from pathlib import Path

WIDTH = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8, "U8": 1, "I8": 1}
STREAM_GBPS = 277.0
PEAK_GBPS = 320.1


def classify(name: str) -> str:
    if "embed_tokens_per_layer" in name:
        return "lookup"
    if "vision" in name or "audio" in name:
        return "idle"
    if "language_model" in name:
        return "read"
    return "idle"


def main() -> None:
    for snap in sys.argv[1:]:
        read_b, by_class = 0, Counter()
        for fn in sorted(Path(snap).glob("*.safetensors")):
            with open(fn, "rb") as fh:
                header = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                n = 1
                for d in meta["shape"]:
                    n *= d
                # Anything stored as bf16/f32 is run as fp16 under --dtype float16.
                width = 2 if meta["dtype"] in ("BF16", "F32", "F16") else WIDTH[meta["dtype"]]
                b = n * width
                cls = classify(name)
                by_class[cls] += b
                if cls == "read":
                    read_b += b
        print(f"== {snap}")
        for cls in ("read", "lookup", "idle"):
            print(f"  {cls:7s} {by_class[cls] / 1e9:7.3f} GB")
        print(f"  bytes read per decode token: {read_b / 1e9:.3f} GB")
        print(f"  bound @ {STREAM_GBPS} GB/s measured stream: {STREAM_GBPS / (read_b / 1e9):6.1f} tok/s")
        print(f"  bound @ {PEAK_GBPS} GB/s theoretical:      {PEAK_GBPS / (read_b / 1e9):6.1f} tok/s")


if __name__ == "__main__":
    main()
