"""Pack a Gemma 4 checkpoint's embedding tables as compressed-tensors int4.

    python3 embed_int4.py SRC_DIR OUT_DIR [--embed-tokens] [--scale-dtype f16|bf16] [--chunk 4096]

SRC_DIR is a `repack_q4_0.py` or `text_only.py` output whose embedding tables
are still bf16. The QAT export trains them onto the same 4-bit grid as the
linears, group 32 along each row (measured 2026-09-29 on E2B: every sampled
group on-grid, against none in the bf16 base model), so this recovers the grid
rather than quantizing, and any off-grid group aborts the run.

What it packs:

- `embed_tokens_per_layer` (the PLE table of the E sizes), wherever it exists;
  12B, 26B and 31B have none, so there --embed-tokens is required;
- with --embed-tokens, `embed_tokens` too. vLLM ties `lm_head` by copying
  `.weight`, which a packed embedding does not have, so this unties them:
  `tie_word_embeddings` becomes false and the same levels and scales are
  written a second time as `lm_head`, which vLLM then runs as an int4 linear.
  The model was trained tied, so both copies hold the trained values.

Layout is what vLLM (>= 0.29) loads: `weight_packed` int32 [rows, cols / 8]
(nibble i of word j = level + 8 for column 8j + i), `weight_scale`
[rows, cols / 32], `weight_shape` int64 [2]. Embeddings go through
`CompressedTensorsEmbeddingWNA16Int`, matched by a config group per table.

--scale-dtype f16 (the default) stores each group's step as float16, which
represents the recovered steps more closely than bf16 and is exact on an fp16
runtime such as a T4. On a bf16 runtime vLLM casts it to bf16 at load, which
is no worse than storing bf16. Rows are processed in chunks so host memory
stays near one chunk in float32; shards with nothing to pack are hard-linked
(copied across filesystems).
"""

import argparse
import copy
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repack_q4_0 import (GROUP, REL_TOL, bf16_to_f32, f32_to_bf16,  # noqa: E402
                         open_checkpoint, pack, write_safetensors)

PLE = "embed_tokens_per_layer"
EMBED = "embed_tokens"


def recover(w_u16, scale_dtype):
    """bf16 bits [rows, cols] -> (levels int8, scale bits, off-grid groups, exact values, worst rel).

    Same step recovery as repack_q4_0.quantize (the first m in 1..8 with
    step = amax / m that reproduces the group, refined by least squares), with
    the step stored at `scale_dtype` and the levels derived against that
    stored value.
    """
    rows, cols = w_u16.shape
    g = bf16_to_f32(w_u16).reshape(rows, cols // GROUP, GROUP)
    amax = np.abs(g).max(axis=-1, keepdims=True)
    nonzero = amax > 0
    tol = amax * REL_TOL
    d = np.where(nonzero, amax / 8.0, 0.0).astype(np.float32)
    ok = ~nonzero
    for m in range(1, 9):
        cand = np.where(nonzero, amax / m, 1.0).astype(np.float32)
        q = np.clip(np.rint(g / cand), -8, 7)
        err = np.abs(g - q * cand).max(axis=-1, keepdims=True)
        good = (err <= tol) & nonzero & ~ok
        d = np.where(good, cand, d)
        ok |= good
    q = np.clip(np.rint(g / np.where(d > 0, d, 1.0)), -8, 7)
    den = (q * q).sum(axis=-1, keepdims=True)
    num = (q * g).sum(axis=-1, keepdims=True)
    d_ls = np.where(den > 0, num / np.where(den > 0, den, 1.0), d).astype(np.float32)

    def levels(s):
        """Levels against stored step s, and the group's fit: values that come
        back exactly once the runtime rounds the fp32 product to fp16 (the
        source bf16 values are exact in fp16), then squared error."""
        q = np.clip(np.rint(g / np.where(s > 0, s, 1.0)), -8, 7)
        rec = (q * s).astype(np.float16).astype(np.float32)
        miss = (rec != g).sum(axis=-1, keepdims=True)
        return q, miss, ((q * s - g) ** 2).sum(axis=-1, keepdims=True)

    if scale_dtype == "f16":
        # Per group, the best of three fp16 steps: the grid step, the refined
        # step, and the refined step rounded through bf16. Every bf16 value in
        # this range is exact in fp16, so the result is never worse than bf16.
        # Ranked by exact values first: the refined step alone minimizes
        # squared error but keeps a drift that no value lands on.
        cands = [d.astype(np.float16), d_ls.astype(np.float16),
                 bf16_to_f32(f32_to_bf16(d_ls)).astype(np.float16)]
        best = cands[0]
        _, best_miss, best_sse = levels(best.astype(np.float32))
        for c in cands[1:]:
            _, miss, sse = levels(c.astype(np.float32))
            better = (miss < best_miss) | ((miss == best_miss) & (sse < best_sse))
            best = np.where(better, c, best)
            best_miss = np.where(better, miss, best_miss)
            best_sse = np.where(better, sse, best_sse)
        bits = best
        s = bits.astype(np.float32)
    else:
        bits = f32_to_bf16(d_ls)
        s = bf16_to_f32(bits)
    q, miss, _ = levels(s)
    rec = q * s
    exact = int(g.size - miss.sum())
    worst = float((np.abs(rec - g) / np.where(amax > 0, amax, 1.0)).max())
    return (q.astype(np.int8).reshape(rows, cols), bits.reshape(rows, cols // GROUP),
            int((~ok).sum()), exact, worst)


def pack_table(ck, name, scale_dtype, chunk):
    w = ck[name].raw(name)
    rows, cols = w.shape
    packed = np.empty((rows, cols // 8), dtype=np.int32)
    scale = np.empty((rows, cols // GROUP), dtype=np.float16 if scale_dtype == "f16" else np.uint16)
    bad = exact = 0
    worst = 0.0
    for r0 in range(0, rows, chunk):
        q, s, b, e, wr = recover(np.asarray(w[r0:r0 + chunk]), scale_dtype)
        packed[r0:r0 + chunk] = pack(q)
        scale[r0:r0 + chunk] = s
        bad, exact, worst = bad + b, exact + e, max(worst, wr)
    if bad:
        sys.exit(f"{name}: {bad} groups are off the 4-bit grid; not writing a lossy table")
    s32 = scale.astype(np.float32) if scale_dtype == "f16" else bf16_to_f32(scale)
    print(f"{name}: {rows * cols * 2 / 2**30:.3f} GiB bf16 -> "
          f"{(packed.nbytes + scale.nbytes) / 2**30:.3f} GiB int4 + {scale_dtype} scales; "
          f"0 off-grid groups; {100 * exact / (rows * cols):.2f}% of values bit-identical, "
          f"worst error {worst:.2e} of the group max; scales {s32[s32 > 0].min():.3g}..{s32.max():.3g}",
          flush=True)
    tag = "F16" if scale_dtype == "f16" else "BF16"
    return packed, (tag, scale), np.array([rows, cols], dtype=np.int64)


def entries(module, packed, scale, shape):
    return [(module + ".weight_packed", "I32", packed),
            (module + ".weight_scale", scale[0], scale[1]),
            (module + ".weight_shape", "I64", shape)]


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
    p.add_argument("--embed-tokens", action="store_true",
                   help="also pack embed_tokens, and untie lm_head as an int4 copy")
    p.add_argument("--scale-dtype", choices=["f16", "bf16"], default="f16")
    p.add_argument("--chunk", type=int, default=4096)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    index = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"]
    ck = open_checkpoint(a.src)
    # 12B, 26B and 31B have no per-layer embeddings, so the PLE table is packed
    # only where it exists; there, --embed-tokens is the whole point.
    has_ple = any(n.endswith(f".{PLE}.weight") for n in index)
    tables = ([PLE] if has_ple else []) + ([EMBED] if a.embed_tokens else [])
    if not tables:
        sys.exit("no per-layer embedding table here: pass --embed-tokens")
    names = {}
    for t in tables:
        found = [n for n in index if n.endswith(f".{t}.weight")]
        if len(found) != 1:
            sys.exit(f"expected one {t}.weight, found {found}")
        if ck[found[0]].header[found[0]]["dtype"] != "BF16":
            sys.exit(f"{found[0]} is not BF16")
        names[t] = found[0]

    new = {}  # shard -> list of entries
    for t, name in names.items():
        packed, scale, shape = pack_table(ck, name, a.scale_dtype, a.chunk)
        new.setdefault(index[name], []).extend(entries(name[: -len(".weight")], packed, scale, shape))
        if t == EMBED:
            new[index[name]].extend(entries("lm_head", packed, scale, shape))

    dropped = set(names.values())
    weight_map = {}
    for fname in sorted(set(index.values())):
        mine = [n for n, f in index.items() if f == fname]
        if fname not in new:
            link_or_copy(os.path.join(a.src, fname), os.path.join(a.out, fname))
            weight_map.update({n: fname for n in mine})
            continue
        keep = [(n, ck[n].header[n]["dtype"], np.asarray(ck[n].raw(n)))
                for n in sorted(mine) if n not in dropped]
        shard = keep + new[fname]
        write_safetensors(os.path.join(a.out, fname), shard)
        weight_map.update({e[0]: fname for e in shard})

    total = sum(os.path.getsize(os.path.join(a.out, f)) for f in set(weight_map.values()))
    json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))},
              open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=2)

    cfg = json.load(open(os.path.join(a.src, "config.json")))
    q = cfg["quantization_config"]
    base = q["config_groups"]["group_0"]
    for t in tables:
        g = copy.deepcopy(base)
        g["targets"] = [f"re:.*\\.{t}$"]
        q["config_groups"][f"group_{t}"] = g
    if a.embed_tokens:
        g = copy.deepcopy(base)
        g["targets"] = ["re:^lm_head$"]
        q["config_groups"]["group_lm_head"] = g
        q["ignore"] = [m for m in q.get("ignore", []) if m != "lm_head"]
        cfg["tie_word_embeddings"] = False
        if isinstance(cfg.get("text_config"), dict):
            cfg["text_config"]["tie_word_embeddings"] = False
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)

    for f in os.listdir(a.src):
        path = os.path.join(a.src, f)
        if (f not in ("config.json", "model.safetensors.index.json", "README.md")
                and not f.endswith(".safetensors") and os.path.isfile(path)):
            shutil.copy2(path, a.out)
    print(f"total {total / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
