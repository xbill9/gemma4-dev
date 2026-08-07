"""Compare the bf16 and fp8 KV cache arms and emit the markdown summary.

Takes the four JSON files the two arms produce (quality + cells each) and answers the three
questions the run was designed around:

  1. Capacity  - did kv_cache_size_tokens land on exactly 2x? That is a prediction from
                 get_dtype_packing (32 // itemsize_bits), not a comparison against another
                 config, so it is the one check that can prove the flag actually took effect.
  2. Throughput - did the KEY cells (over bf16 capacity, inside fp8) gain, while the CONTROL
                 cells stayed flat? A gain on the controls means warm-up or noise contaminated
                 the run and the KEY numbers cannot be trusted.
  3. Quality   - how far did the outputs move, weighted toward the long-context probes where
                 uncalibrated scale-1.0 quantization error accumulates.

stdlib only; do not assume pandas is installed.
"""

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

BF16_KV_TOKENS = 321_376
NEEDLE_ANSWER = "48213"


def load(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def first_divergence(a: str, b: str) -> Optional[int]:
    """Index of the first differing character, or None if one is a prefix of the other."""
    for i, (ca, cb) in enumerate(zip(a, b, strict=False)):
        if ca != cb:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def compare_quality(bf16: Dict[str, Any], fp8: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    by_id = {r["id"]: r for r in fp8["results"]}
    rows: List[Dict[str, Any]] = []
    for base in bf16["results"]:
        other = by_id.get(base["id"])
        if other is None:
            rows.append({"id": base["id"], "note": "missing in fp8 arm"})
            continue
        a, b = base.get("text") or "", other.get("text") or ""
        is_needle = base["id"].startswith("needle-")
        rows.append(
            {
                "id": base["id"],
                "identical": a == b,
                "diverge_at": first_divergence(a, b),
                "bf16_chars": len(a),
                "fp8_chars": len(b),
                "bf16_needle_ok": NEEDLE_ANSWER in a if is_needle else None,
                "fp8_needle_ok": NEEDLE_ANSWER in b if is_needle else None,
                "bf16_text": a,
                "fp8_text": b,
            }
        )

    scored = [r for r in rows if "identical" in r]
    identical = sum(1 for r in scored if r["identical"])
    needles = [r for r in scored if r["id"].startswith("needle-")]
    lines = [
        "## Quality",
        "",
        f"Identical outputs: **{identical}/{len(scored)}** at temperature 0.",
        "",
        "| Probe | Identical | First divergence (char) | bf16 needle | fp8 needle |",
        "|---|---|---|---|---|",
    ]
    for r in scored:
        d = "-" if r["diverge_at"] is None else str(r["diverge_at"])
        bn = "-" if r["bf16_needle_ok"] is None else ("yes" if r["bf16_needle_ok"] else "**NO**")
        fn = "-" if r["fp8_needle_ok"] is None else ("yes" if r["fp8_needle_ok"] else "**NO**")
        lines.append(f"| `{r['id']}` | {'yes' if r['identical'] else 'no'} | {d} | {bn} | {fn} |")
    if needles:
        lost = [r for r in needles if r["bf16_needle_ok"] and not r["fp8_needle_ok"]]
        lines += ["", f"Long-context recall lost under fp8: **{len(lost)}/{len(needles)}** probes."]
    return rows, "\n".join(lines)


def pct(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None or not old:
        return None
    return round((new - old) / old * 100.0, 1)


def compare_cells(bf16: Dict[str, Any], fp8: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    by_key = {(c["input_len"], c["concurrency"]): c for c in fp8["cells"]}
    rows: List[Dict[str, Any]] = []
    for base in bf16["cells"]:
        other = by_key.get((base["input_len"], base["concurrency"]))
        if other is None:
            continue
        rows.append(
            {
                "input_len": base["input_len"],
                "concurrency": base["concurrency"],
                "role": base["role"],
                "kv_tokens_needed": base["kv_tokens_needed"],
                "bf16_status": base["status"],
                "fp8_status": other["status"],
                "bf16_out_tok_s": base.get("output_tok_per_s"),
                "fp8_out_tok_s": other.get("output_tok_per_s"),
                "delta_out_tok_s_pct": pct(other.get("output_tok_per_s"), base.get("output_tok_per_s")),
                "bf16_ttft_median_ms": base.get("ttft_ms_median"),
                "fp8_ttft_median_ms": other.get("ttft_ms_median"),
                "delta_ttft_pct": pct(other.get("ttft_ms_median"), base.get("ttft_ms_median")),
            }
        )

    lines = [
        "## Throughput",
        "",
        "| ctx | conc | role | KV needed | bf16 out tok/s | fp8 out tok/s | Δ | bf16 TTFT ms | fp8 TTFT ms | Δ |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:

        def fmt(v: Any, suffix: str = "") -> str:
            return "-" if v is None else f"{v}{suffix}"

        lines.append(
            f"| {r['input_len']} | {r['concurrency']} | {r['role']} | {r['kv_tokens_needed']:,} | "
            f"{fmt(r['bf16_out_tok_s'])} | {fmt(r['fp8_out_tok_s'])} | {fmt(r['delta_out_tok_s_pct'], '%')} | "
            f"{fmt(r['bf16_ttft_median_ms'])} | {fmt(r['fp8_ttft_median_ms'])} | {fmt(r['delta_ttft_pct'], '%')} |"
        )

    controls = [r["delta_out_tok_s_pct"] for r in rows if r["role"] == "control"]
    controls = [c for c in controls if c is not None]
    keys = [r["delta_out_tok_s_pct"] for r in rows if r["role"] == "key"]
    keys = [k for k in keys if k is not None]
    if controls:
        worst = max(abs(c) for c in controls)
        lines += ["", f"Control drift (should be ~0): max |Δ| = **{worst}%**."]
        if worst > 10:
            lines.append("")
            lines.append(
                "> Controls moved more than 10%. The KEY-cell deltas below are not "
                "trustworthy — something other than KV capacity changed between arms."
            )
    band = [r["delta_out_tok_s_pct"] for r in rows if r["role"] == "bandwidth"]
    band = [b for b in band if b is not None]
    if band:
        lines += [
            "",
            f"BANDWIDTH cells (fit in bf16, but KV is ~35-40% of bytes moved): "
            f"Δ = {', '.join(f'{b}%' for b in band)} — predicted ~+15-18% from halved KV traffic alone, "
            f"with no capacity effect.",
        ]
    if keys:
        lines += ["", f"KEY cells (over bf16 capacity, inside fp8): Δ = {', '.join(f'{k}%' for k in keys)}."]
    if controls and band and keys:
        lines += [
            "",
            f"Monotonicity check — mean Δ by role: control {sum(controls) / len(controls):.1f}%, "
            f"bandwidth {sum(band) / len(band):.1f}%, key {sum(keys) / len(keys):.1f}%. "
            f"Increasing across the three is the signature of a real bytes-moved effect; "
            f"a flat or non-monotone profile is not.",
        ]
    return rows, "\n".join(lines)


def capacity_section(bf16_tokens: Optional[int], fp8_tokens: Optional[int]) -> str:
    lines = ["## Capacity", ""]
    if bf16_tokens is None or fp8_tokens is None:
        lines.append("Not recorded — read `vllm:cache_config_info` from /metrics on each arm.")
        return "\n".join(lines)
    ratio = fp8_tokens / bf16_tokens if bf16_tokens else 0.0
    verdict = "as predicted" if abs(ratio - 2.0) < 0.02 else "**does not match the 2x prediction**"
    lines += [
        f"- bf16 `kv_cache_size_tokens`: **{bf16_tokens:,}**",
        f"- fp8 `kv_cache_size_tokens`: **{fp8_tokens:,}**",
        f"- Ratio: **{ratio:.3f}x** — {verdict}.",
        "",
        "A ratio near 1.0 means the flag was accepted but changed nothing.",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16-quality", required=True)
    ap.add_argument("--fp8-quality", required=True)
    ap.add_argument("--bf16-cells", required=True)
    ap.add_argument("--fp8-cells", required=True)
    ap.add_argument("--bf16-kv-tokens", type=int, default=BF16_KV_TOKENS)
    ap.add_argument("--fp8-kv-tokens", type=int, default=None)
    ap.add_argument("--output", default="SUMMARY.md")
    args = ap.parse_args()

    q_rows, q_md = compare_quality(load(args.bf16_quality), load(args.fp8_quality))
    c_rows, c_md = compare_cells(load(args.bf16_cells), load(args.fp8_cells))
    cap_md = capacity_section(args.bf16_kv_tokens, args.fp8_kv_tokens)

    md = "\n\n".join(["# KV cache: bf16 vs fp8 on v5e-1", cap_md, c_md, q_md])
    with open(args.output, "w") as f:
        f.write(md + "\n")
    with open("comparison.json", "w") as f:
        json.dump({"quality": q_rows, "cells": c_rows}, f, indent=2)
    print(f"Wrote {args.output} and comparison.json")


if __name__ == "__main__":
    main()
