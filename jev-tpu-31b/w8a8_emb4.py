"""Put the int4 vocabulary tables of an `-emb4` checkpoint into a W8A8 checkpoint.

    python3 w8a8_emb4.py W8A8_DIR EMB4_DIR OUT_DIR

W8A8_DIR is a `w8a8_from_qat.py` build (int8 linears, bf16 embeddings, tied lm_head); EMB4_DIR
is the `-emb4` build of the same model (int4 `embed_tokens`, `embed_tokens_per_layer` and an
untied `lm_head`, pack-quantized, group 32). The output takes the linears and every other
tensor from W8A8_DIR and the three vocabulary tables from EMB4_DIR, byte for byte, with the
config groups of both. Both builds come from the same QAT export, so the tensors they share
are checked to be identical.
"""

import argparse
import json
import os
import re
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repack_q4_0 import open_checkpoint, write_safetensors  # noqa: E402

VOCAB = re.compile(r"^(model\.language_model\.(embed_tokens|embed_tokens_per_layer)|lm_head)\.")
SHARD_BYTES = 2 << 30


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("w8a8")
    p.add_argument("emb4")
    p.add_argument("out")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    w8, e4 = open_checkpoint(a.w8a8), open_checkpoint(a.emb4)

    # Quantized tensors (`weight_scale` and friends) share names but not schemes; the rest
    # (norms, scalars) are copies of the QAT export in both.
    shared = [n for n in w8 if n in e4 and not VOCAB.match(n)
              and not n.endswith((".weight_scale", ".weight_packed", ".weight_shape"))]
    differ = [n for n in shared if bytes(w8[n].raw(n)) != bytes(e4[n].raw(n))
              and w8[n].header[n]["dtype"] == e4[n].header[n]["dtype"]]
    if differ:
        sys.exit(f"{len(differ)} tensors the two builds share differ, e.g. {differ[:3]}")

    picks = [(n, w8) for n in sorted(w8) if not VOCAB.match(n)]
    picks += [(n, e4) for n in sorted(e4) if VOCAB.match(n)]
    shard, size, n_shard, weight_map = [], 0, 0, {}

    def flush():
        nonlocal shard, size, n_shard
        if shard:
            fname = f"model-{n_shard:05d}.safetensors"
            write_safetensors(os.path.join(a.out, fname), shard)
            weight_map.update({t[0]: fname for t in shard})
            shard, size, n_shard = [], 0, n_shard + 1

    for name, ck in picks:
        raw = np.asarray(ck[name].raw(name))
        if size + raw.nbytes > SHARD_BYTES:
            flush()
        shard.append((name, ck[name].header[name]["dtype"], raw))
        size += raw.nbytes
    flush()

    total = sum(os.path.getsize(os.path.join(a.out, f)) for f in set(weight_map.values()))
    json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))},
              open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=2)

    cfg = json.load(open(os.path.join(a.w8a8, "config.json")))
    e4cfg = json.load(open(os.path.join(a.emb4, "config.json")))
    q = cfg["quantization_config"]
    for k, g in e4cfg["quantization_config"]["config_groups"].items():
        if k != "group_0":  # the emb4 build's group_0 is its W4A16 linears
            q["config_groups"][k] = g
    q["ignore"] = [t for t in q.get("ignore", []) if t != "lm_head"]
    cfg["tie_word_embeddings"] = False
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    for f in os.listdir(a.w8a8):
        path = os.path.join(a.w8a8, f)
        if (f not in ("config.json", "model.safetensors.index.json", "README.md", "w8a8_report.json")
                and not f.endswith(".safetensors") and os.path.isfile(path)):
            shutil.copy2(path, a.out)

    print(f"{len(shared)} shared tensors identical; {sum(1 for _, c in picks if c is e4)} vocabulary "
          f"tensors from {a.emb4}; total {total / 2**30:.2f} GiB in {n_shard} shards")
    print("config groups:", {k: (g["targets"], g["weights"]["num_bits"]) for k, g in q["config_groups"].items()},
          "ignore", q["ignore"])


if __name__ == "__main__":
    main()
