#!/usr/bin/env python3
"""Summarise an xprof/TensorBoard trace captured from the vLLM TPU engine.

    python3 analyze_trace.py benchmarks/runs/2026-08-25-xprof-tp4

This is the TPU counterpart of `gpu-jax-g5g-2b/profile_decode.py`'s kernel table, which
answered "why is decode slow" in one run when six plausible theories had failed. The shape of
the question here is the same: which HLO ops own the decode step, and are the attention ops
the ones the serving stack claims to be running.

WHAT IT IS FOR ON THIS RIG. Zimbres 2026 §6.4 decomposes decode attention by LAYER TYPE and
finds the two populations scale differently once the tensor parallel degree crosses the KV
head count. Reproducing that needs per-op device time attributed to windowed vs global
layers, which is what `--by-attention` below approximates from op names.

TWO TRAPS, both learned from that paper's methodology section and the sibling rig's:

- **Host events are not device events.** The export interleaves Python/queue/framework spans
  with device work on separate tracks. Summing them together makes device work look like a
  few percent of the total. xprof's op profile is already device-only; if you ever parse the
  raw trace yourself, resolve the lane first.
- **The export truncates silently at a fixed event count.** Zimbres captured ~7 of 127 decode
  steps at TP=8 and ~14 at TP=4. Any statistic formed by dividing a raw SUM by the requested
  step count is wrong by the truncation ratio. Per-op MEANS are safe because each per-layer op
  fires exactly once per step per chip, so this script reports means and occurrences and
  refuses to divide by a step count it cannot observe.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys


def find_session(logdir: str) -> str:
    """xprof commands take the run directory holding the .xplane.pb."""
    hits = glob.glob(os.path.join(logdir, "**", "*.xplane.pb"), recursive=True)
    if not hits:
        sys.exit(
            f"No *.xplane.pb under {logdir}.\n"
            "Capture one with ./capture_profile.sh — note that vLLM's own /start_profile is\n"
            "absent from vllm-tpu:nightly, so the sidecar is the only route."
        )
    return os.path.dirname(sorted(hits)[0])


def xprof(cmd: str, session: str, extra: list[str] | None = None) -> str:
    """Run one xprof CLI analysis. Returns '' if that analysis is unavailable."""
    argv = ["xprof", cmd, session, *(extra or [])]
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return f"({cmd} unavailable: {type(exc).__name__})"
    return done.stdout.strip() or f"({cmd} returned nothing: {done.stderr.strip()[:200]})"


def classify(name: str) -> str:
    """Tag the ops that answer the questions this rig actually has.

    The attention split is the point: Gemma 4 31B has 50 sliding-window layers and 10 global
    ones, and on this checkpoint they are the two populations whose scaling diverges once TP
    crosses the 4-head limit. Kernel names carry the window, which is what makes the split
    readable from a trace at all.
    """
    low = name.lower()
    if "window" in low or "sliding" in low:
        return "ATTN sliding"
    if "flash" in low or "attention" in low or "ragged" in low or "paged" in low:
        return "ATTN global/other"
    if "all-reduce" in low or "allreduce" in low or "all-gather" in low or "collective" in low:
        return "COLLECTIVE"
    if "convert" in low:
        return "DTYPE CONVERSION"
    if "fusion" in low or "dot" in low or "gemm" in low or "matmul" in low:
        return "MATMUL/FUSION"
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logdir")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    session = find_session(args.logdir)
    print(f"session: {session}\n")

    print("=" * 78)
    print("DEVICE / HARDWARE")
    print("=" * 78)
    print(xprof("get_device_information", session))

    print("\n" + "=" * 78)
    print(f"TOP {args.top} HLO OPS BY DEVICE TIME")
    print("=" * 78)
    profile = xprof("get_hlo_op_profile", session)
    print(profile)

    # These three detectors encode failure modes this monorepo has actually hit: an unwanted
    # bf16->f32 promotion is exactly the class of bug that cost the GPU rig 55% of its decode
    # step, and layout-mismatch copies are the TPU equivalent.
    print("\n" + "=" * 78)
    print("KNOWN-PATHOLOGY DETECTORS")
    print("=" * 78)
    for cmd, why in [
        ("detect_unnecessary_convert_reduce", "bf16 promoted to f32 in a reduce"),
        ("detect_layout_mismatch_copies", "copies inserted between compute stages"),
        ("detect_unfused_reshapes", "reshape/transpose materialised to HBM"),
    ]:
        print(f"\n--- {cmd}  ({why}) ---")
        print(xprof(cmd, session))

    print("\n" + "=" * 78)
    print("ATTENTION SPLIT (sliding vs global)")
    print("=" * 78)
    print(
        "Gemma 4 31B is 50 sliding + 10 global layers. Zimbres 2026 §6.4 measures these two\n"
        "populations scaling differently across the TP/KV-head crossing. Op names below are\n"
        "tagged heuristically; confirm against the kernel names before quoting a split.\n"
    )
    tagged = [(line, classify(line)) for line in profile.splitlines() if classify(line)]
    if tagged:
        for line, tag in tagged[: args.top]:
            print(f"  [{tag:>18}] {line.strip()[:100]}")
    else:
        print("  (no op names matched the tags — read the table above directly)")

    print("\n" + "=" * 78)
    print(f"view: xprof --logdir {args.logdir}   |   tensorboard --logdir {args.logdir}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
