"""Strip the vision and audio towers from a repacked Gemma 4 W4A16 checkpoint.

    python3 text_only.py SRC_DIR OUT_DIR

SRC_DIR is a `repack_q4_0.py` output (or any Gemma 4 `Gemma4ForConditionalGeneration`
checkpoint). OUT_DIR gets a text-only `Gemma4ForCausalLM` checkpoint:

- every tensor under `model.{vision_tower,audio_tower,embed_vision,embed_audio}.`
  is dropped; every other tensor is copied byte for byte, names unchanged
  (vLLM's Gemma4ForCausalLM maps `model.language_model.*` itself);
- config.json becomes the source `text_config` (model_type `gemma4_text`) plus the
  top-level dtype and quantization_config, with tower modules removed from `ignore`;
- processor_config.json is not copied: with no towers there is nothing to process;
- shards that lose no tensors are hard-linked, not rewritten, so the output costs
  only the rewritten shards on disk when it sits on the same filesystem as SRC_DIR.

Nothing is re-quantized, so the language model is bit-identical to SRC_DIR.
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

TOWER = re.compile(r"^model\.(vision_tower|audio_tower|embed_vision|embed_audio)\.")
SKIP_FILES = {"config.json", "model.safetensors.index.json", "processor_config.json",
              "repack_report.json"}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("src")
    p.add_argument("out")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    index = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"]
    ck = open_checkpoint(a.src)
    by_file, dropped = {}, 0
    for name, fname in index.items():
        if TOWER.match(name):
            dropped += 1
        else:
            by_file.setdefault(fname, []).append(name)

    weight_map = {}
    for fname, names in sorted(by_file.items()):
        src, dst = os.path.join(a.src, fname), os.path.join(a.out, fname)
        if len(names) == sum(f == fname for f in index.values()):
            # Nothing dropped from this shard: link it (copy across filesystems).
            if os.path.exists(dst):
                os.remove(dst)
            try:
                os.link(os.path.realpath(src), dst)
            except OSError:
                shutil.copy2(src, dst)
            how = "linked"
        else:
            shard = [(n, ck[n].header[n]["dtype"], np.asarray(ck[n].raw(n))) for n in sorted(names)]
            write_safetensors(dst, shard)
            how = "rewritten"
        weight_map.update({n: fname for n in names})
        print(f"{fname}: {len(names)} tensors, {how}", flush=True)

    total = sum(os.path.getsize(os.path.join(a.out, f)) for f in by_file)
    json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))},
              open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=2)

    src_cfg = json.load(open(os.path.join(a.src, "config.json")))
    cfg = dict(src_cfg["text_config"])
    cfg["architectures"] = ["Gemma4ForCausalLM"]
    cfg["dtype"] = src_cfg.get("dtype", cfg.get("dtype"))
    cfg["tie_word_embeddings"] = src_cfg.get("tie_word_embeddings", cfg.get("tie_word_embeddings"))
    if "transformers_version" in src_cfg:
        cfg["transformers_version"] = src_cfg["transformers_version"]
    if "quantization_config" in src_cfg:
        q = dict(src_cfg["quantization_config"])
        q["ignore"] = [m for m in q.get("ignore", []) if not TOWER.match(m + ".")]
        cfg["quantization_config"] = q
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)

    for f in os.listdir(a.src):
        path = os.path.join(a.src, f)
        if f not in SKIP_FILES and not f.endswith(".safetensors") and os.path.isfile(path):
            shutil.copy2(path, a.out)

    print(f"dropped {dropped} tower tensors; kept {len(weight_map)} in {len(by_file)} shards, "
          f"{total / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
