"""Pack a Gemma 4 E checkpoint's per-layer-embedding (PLE) table as compressed-tensors int4.

    python3 ple_int4.py SRC_DIR OUT_DIR [--chunk 4096]

SRC_DIR is a `repack_q4_0.py` or `text_only.py` output whose
`embed_tokens_per_layer.weight` is still bf16. The QAT export trains that table
onto the same 4-bit grid as the linears (group 32 along each row; measured
2026-09-29: every sampled group on-grid, against none in the bf16 base model),
so this recovers the grid rather than quantizing: `repack_q4_0.quantize` finds
each group's step, and any off-grid group aborts the run.

It is written in the layout vLLM's `CompressedTensorsEmbeddingWNA16Int` loads
(vLLM >= 0.29): `weight_packed` int32 [vocab, dim / 8], `weight_scale` bf16
[vocab, dim / 32], `weight_shape` int64 [2]. config.json gains a second
config group targeting `re:.*embed_tokens_per_layer$`. Rows are processed in
chunks so host memory stays near one chunk in float32. Shards that do not hold
the table are hard-linked (copied across filesystems).
"""

import argparse
import copy
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repack_q4_0 import (GROUP, bf16_to_f32, open_checkpoint, pack,  # noqa: E402
                         quantize, write_safetensors)

TARGET = "re:.*embed_tokens_per_layer$"


def link_or_copy(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(os.path.realpath(src), dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("src")
    p.add_argument("out")
    p.add_argument("--chunk", type=int, default=4096)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    index = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"]
    names = [n for n in index if n.endswith("embed_tokens_per_layer.weight")]
    if len(names) != 1:
        sys.exit(f"expected one embed_tokens_per_layer.weight, found {names}")
    name = names[0]
    module = name[: -len(".weight")]
    ck = open_checkpoint(a.src)
    if ck[name].header[name]["dtype"] != "BF16":
        sys.exit(f"{name} is {ck[name].header[name]['dtype']}, expected BF16")
    w = ck[name].raw(name)
    vocab, dim = w.shape
    packed = np.empty((vocab, dim // 8), dtype=np.int32)
    scale = np.empty((vocab, dim // GROUP), dtype=np.uint16)
    bad = exact = 0
    worst = 0.0
    for r0 in range(0, vocab, a.chunk):
        src = np.asarray(w[r0:r0 + a.chunk])
        q, s, n_bad = quantize(src)
        bad += n_bad
        packed[r0:r0 + a.chunk] = pack(q)
        scale[r0:r0 + a.chunk] = s
        g = bf16_to_f32(src).reshape(len(src), -1, GROUP)
        rec = q.reshape(g.shape).astype(np.float32) * bf16_to_f32(s)[..., None]
        exact += int((rec == g).sum())
        amax = np.abs(g).max(axis=-1, keepdims=True)
        rel = np.abs(rec - g) / np.where(amax > 0, amax, 1.0)
        worst = max(worst, float(rel.max()))
        if r0 // a.chunk % 8 == 0:
            print(f"rows {r0:>6}/{vocab}  off-grid groups so far {bad}", flush=True)
    if bad:
        sys.exit(f"{bad} groups are off the 4-bit grid; not writing a lossy table")

    shard_of = index[name]
    weight_map = {}
    for fname in sorted(set(index.values())):
        if fname == shard_of:
            continue
        link_or_copy(os.path.join(a.src, fname), os.path.join(a.out, fname))
        weight_map.update({n: fname for n, f in index.items() if f == fname})
    # The table's shard: its other tensors, byte for byte, plus the packed table.
    others = [(n, ck[n].header[n]["dtype"], np.asarray(ck[n].raw(n)))
              for n, f in sorted(index.items()) if f == shard_of and n != name]
    new = others + [(module + ".weight_packed", "I32", packed),
                    (module + ".weight_scale", "BF16", scale),
                    (module + ".weight_shape", "I64", np.array([vocab, dim], dtype=np.int64))]
    write_safetensors(os.path.join(a.out, shard_of), new)
    weight_map.update({t[0]: shard_of for t in new})

    total = sum(os.path.getsize(os.path.join(a.out, f)) for f in set(weight_map.values()))
    json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))},
              open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=2)

    cfg = json.load(open(os.path.join(a.src, "config.json")))
    q = cfg["quantization_config"]
    groups = q["config_groups"]
    ple = copy.deepcopy(groups["group_0"])
    ple["targets"] = [TARGET]
    groups["group_ple"] = ple
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)

    for f in os.listdir(a.src):
        path = os.path.join(a.src, f)
        if (f not in ("config.json", "model.safetensors.index.json")
                and not f.endswith(".safetensors") and os.path.isfile(path)):
            shutil.copy2(path, a.out)

    n = vocab * dim
    print(f"{name}: {n * 2 / 2**30:.3f} GiB bf16 -> "
          f"{(packed.nbytes + scale.nbytes) / 2**30:.3f} GiB int4 + scales; "
          f"0 off-grid groups; {100 * exact / n:.2f}% of values bit-identical, "
          f"worst error {worst:.2e} of the group max")
    print(f"total {total / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
