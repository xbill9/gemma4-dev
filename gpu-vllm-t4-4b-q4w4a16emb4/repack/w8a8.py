"""Turn a W4A16 checkpoint's linears into W8A8 int8, leaving everything else as it is.

    python3 w8a8.py SRC_DIR OUT_DIR

SRC_DIR is a compressed-tensors `pack-quantized` checkpoint from this rig's other
tools (e.g. the `-text-emb4` build). Every module that config group `group_0`
(target `Linear`) quantizes is unpacked to its exact values (level x group
step), then re-encoded as int8 with one scale per output channel, in the layout
vLLM's `CompressedTensorsW8A8Int8` loads: `weight` int8 [out, in] and
`weight_scale` float32 [out, 1]. Activations are quantized per token at run
time (`dynamic: true`), so no calibration data is needed.

Modules matched by any other config group keep their packing untouched: on
`-text-emb4` that is the three int4 embedding tables and the int4 `lm_head`, so
a sweep against that build isolates the decoder linears' compute format.

Why this exists: the T4's INT8 tensor cores have twice FP16's throughput, and
long-prompt serving on it is prefill-bound. W8A8 is the only format that uses
them. It is NOT lossless here: QAT trained the weights for a 4-bit grid in
groups of 32, and one int8 scale per row cannot hold every group's step
exactly; the activations are rounded to int8 too, which QAT never saw. This
script reports the weight error; accuracy has to be checked separately.
"""

import argparse
import copy
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repack_q4_0 import GROUP, bf16_to_f32, open_checkpoint, unpack, write_safetensors  # noqa: E402

INT8_GROUP = {
    "targets": ["Linear"],
    "format": "int-quantized",
    "weights": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "channel",
                "group_size": None, "dynamic": False, "actorder": None,
                "block_structure": None, "observer": None, "observer_kwargs": {}},
    "input_activations": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "token",
                          "group_size": None, "dynamic": True, "actorder": None,
                          "block_structure": None, "observer": None, "observer_kwargs": {}},
    "output_activations": None,
}


def scale_f32(ck, name):
    tag = ck[name].header[name]["dtype"]
    raw = np.asarray(ck[name].raw(name))
    return bf16_to_f32(raw) if tag == "BF16" else raw.astype(np.float32)


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
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cfg = json.load(open(os.path.join(a.src, "config.json")))
    q = cfg["quantization_config"]
    others = [t[3:] for k, g in q["config_groups"].items() if k != "group_0"
              for t in g["targets"] if t.startswith("re:")]
    import re
    other_re = [re.compile(t) for t in others]

    index = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"]
    ck = open_checkpoint(a.src)
    modules = sorted(n[: -len(".weight_packed")] for n in index if n.endswith(".weight_packed"))
    linears = [m for m in modules if not any(r.match(m) or r.match(m.replace("model.language_model.", "model."))
                                             for r in other_re)]
    kept = [m for m in modules if m not in linears]
    print(f"{len(linears)} linears -> W8A8; kept packed: {kept}", flush=True)

    by_shard = {}
    for m in linears:
        by_shard.setdefault(index[m + ".weight_packed"], []).append(m)
    weight_map, worst, rel_sum, n_mod = {}, 0.0, 0.0, 0
    for fname in sorted(set(index.values())):
        names = [n for n, f in index.items() if f == fname]
        mods = by_shard.get(fname, [])
        if not mods:
            link_or_copy(os.path.join(a.src, fname), os.path.join(a.out, fname))
            weight_map.update({n: fname for n in names})
            continue
        drop = {m + s for m in mods for s in (".weight_packed", ".weight_scale", ".weight_shape")}
        shard = [(n, ck[n].header[n]["dtype"], np.asarray(ck[n].raw(n))) for n in sorted(names) if n not in drop]
        for m in mods:
            lv = unpack(np.asarray(ck[m + ".weight_packed"].raw(m + ".weight_packed"))).astype(np.float32)
            s = scale_f32(ck, m + ".weight_scale")
            out, inn = lv.shape
            w = (lv.reshape(out, inn // GROUP, GROUP) * s[..., None]).reshape(out, inn)
            amax = np.abs(w).max(axis=1, keepdims=True)
            sc = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
            q8 = np.clip(np.rint(w / sc), -127, 127).astype(np.int8)
            err = np.linalg.norm(q8 * sc - w) / max(np.linalg.norm(w), 1e-30)
            worst, rel_sum, n_mod = max(worst, err), rel_sum + err, n_mod + 1
            shard += [(m + ".weight", "I8", q8), (m + ".weight_scale", "F32", sc)]
        write_safetensors(os.path.join(a.out, fname), shard)
        weight_map.update({t[0]: fname for t in shard})
        print(f"{fname}: {len(mods)} linears", flush=True)

    total = sum(os.path.getsize(os.path.join(a.out, f)) for f in set(weight_map.values()))
    json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))},
              open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=2)
    q["config_groups"]["group_0"] = copy.deepcopy(INT8_GROUP)
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    for f in os.listdir(a.src):
        path = os.path.join(a.src, f)
        if (f not in ("config.json", "model.safetensors.index.json", "README.md")
                and not f.endswith(".safetensors") and os.path.isfile(path)):
            shutil.copy2(path, a.out)
    print(f"{n_mod} linears: weight error (relative Frobenius) mean {rel_sum / n_mod:.4f}, "
          f"worst {worst:.4f}; total {total / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
