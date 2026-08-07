"""Identify the 31B's S-independent prefill temp floor from XLA's buffer assignment.

Prior attempts inferred it from resident memory and from HLO instruction shapes, and
both were wrong — instruction output size is not allocation, because fusion. This
asks XLA's buffer-assignment pass directly, which reports the buffers actually
allocated and their sizes.

The floor is ~9-10 GB and S-independent (9.17 GB at S=1024, 10.34 GB at S=4096), so
it is ~1/3 of a v6e-1 spent before sequence length matters. Naming it is worth more
than any per-token optimization.

Dumps at two S so the S-independent allocations can be separated from the ones that
grow.

    python3.13 ports/gemma4/jax_31b_bufassign.py --contexts 1024,4096
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DUMP_ROOT = os.environ.get("BUFASSIGN_DUMP", "/tmp/bufassign")

# Must be set before JAX initializes the backend.
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "")
    + f" --xla_dump_to={DUMP_ROOT}"
      " --xla_dump_hlo_pass_re=buffer-assignment"
).strip()

import jax                                                        # noqa: E402
import jax.numpy as jnp                                           # noqa: E402

from ports.gemma4.jax_e_model import (                            # noqa: E402
    pad_to_tpu_v6e_bucket,
    prefill_with_kv_cache,
)
from ports.gemma4.jax_31b_port import Streaming31BEngine          # noqa: E402

DEFAULT_MODEL = "google/gemma-4-31B-it-qat-w4a16-ct"

# "allocation 12: size 231211008, output shape is bf16[5376,21504], maybe-live-out"
_ALLOC_RE = re.compile(r"allocation\s+(\d+):.*?size\s+(\d+)", re.S)
_SHAPE_IN_ALLOC = re.compile(r"shape is ([a-z0-9]+\[[\d,]*\])")


def parse_buffer_assignment(path: str):
    """Return (total_bytes, [ {alloc, size, shape, snippet} ]) sorted by size."""
    with open(path, errors="replace") as fh:
        text = fh.read()
    # Split on allocation boundaries so each record keeps its own detail lines.
    parts = re.split(r"\n(?=allocation \d+:)", text)
    rows = []
    for p in parts:
        m = re.match(r"allocation (\d+):.*?size (\d+)", p, re.S)
        if not m:
            continue
        shape = _SHAPE_IN_ALLOC.search(p)
        # first non-empty detail line after the header, for provenance
        lines = [l.strip() for l in p.splitlines()[1:] if l.strip()]
        rows.append({"alloc": int(m.group(1)), "size": int(m.group(2)),
                     "shape": shape.group(1) if shape else "?",
                     "detail": lines[0][:150] if lines else ""})
    rows.sort(key=lambda r: -r["size"])
    total = sum(r["size"] for r in rows)
    return total, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--contexts", default="1024,4096")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--json-out", default="gemma4_31b_bufassign.json")
    args = ap.parse_args()

    if os.path.isdir(DUMP_ROOT):
        shutil.rmtree(DUMP_ROOT)
    os.makedirs(DUMP_ROOT, exist_ok=True)

    eng = Streaming31BEngine(model_id=args.model, quant_mode="w4a16",
                             max_model_len=4096, window_kv=True)
    eng.load()
    print(f"\nweights {eng.weight_bytes / 1e9:.3f} GB")
    print(f"XLA_FLAGS = {os.environ['XLA_FLAGS']}\n")

    out = {"weights_gb": round(eng.weight_bytes / 1e9, 4), "contexts": {}}

    for S in [int(x) for x in args.contexts.split(",")]:
        for f in glob.glob(os.path.join(DUMP_ROOT, "*")):
            os.remove(f)
        ids = jnp.ones((1, S), dtype=jnp.int32)
        padded, valid = pad_to_tpu_v6e_bucket(ids)
        comp = jax.jit(
            prefill_with_kv_cache,
            static_argnames=("model", "max_new_tokens", "quant_mode",
                             "cache_dtype", "window_kv"),
        ).lower(model=eng.model, prompt_ids=padded, prompt_valid=valid,
                params=eng.params, max_new_tokens=8, quant_mode="w4a16",
                window_kv=True).compile()
        temp = getattr(comp.memory_analysis(), "temp_size_in_bytes", 0)

        cands = sorted(glob.glob(os.path.join(DUMP_ROOT, "*buffer-assignment*")))
        print(f"=== S={S}  temp {temp / 1e9:.3f} GB  "
              f"({len(cands)} dump file(s)) ===")
        if not cands:
            print("  no buffer-assignment dump produced; files present:")
            for f in sorted(glob.glob(os.path.join(DUMP_ROOT, "*")))[:10]:
                print("   ", os.path.basename(f))
            out["contexts"][S] = {"temp_gb": round(temp / 1e9, 4),
                                  "status": "NO_DUMP"}
            continue

        path = max(cands, key=os.path.getsize)
        total, rows = parse_buffer_assignment(path)
        print(f"  {os.path.basename(path)}: {len(rows)} allocations, "
              f"{total / 1e9:.3f} GB total")
        for r in rows[:args.top]:
            print(f"   {r['size'] / 1e9:8.3f} GB  alloc {r['alloc']:<5d} "
                  f"{r['shape']:<28s} {r['detail'][:60]}")
        out["contexts"][S] = {
            "temp_gb": round(temp / 1e9, 4),
            "total_allocated_gb": round(total / 1e9, 4),
            "n_allocations": len(rows),
            "top": [{**r, "gb": round(r["size"] / 1e9, 4)} for r in rows[:args.top]],
        }
        # keep the dump for offline inspection
        shutil.copy(path, os.path.join(os.path.expanduser("~"),
                                       f"bufassign_S{S}.txt"))
        print()

    with open(args.json_out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
