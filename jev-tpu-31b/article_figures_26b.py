"""Derived figures for the 26B QAT W4A16 article, computed from the committed run outputs.

    python3 article_figures_26b.py > results/2026-09-26-26b-ARTICLE-FIGURES.md

Every percentage, ratio and cost the article quotes that is not printed verbatim by a run
log is computed here, so the article cites this file rather than arithmetic done by hand.
Rates are europe-west4 v6e per chip-hour, from ../jev-tpu/results/2026-09-24-v6e1-DERIVED.md.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
RATES = {"flex-start": 1.35, "on-demand": 2.97}


def load(name):
    return json.load(open(os.path.join(R, name)))


def main():
    verify = load("2026-09-26-repack-evidence/verify_report.json")
    w4 = load("2026-09-26-moe2-evidence/26b-q4w4.load.json")["output_tok_per_s"]
    fp8 = load("2026-09-26-moe-evidence/26b-fp8.load.json")["output_tok_per_s"]
    kv_w4, kv_fp8 = 421 * 128, 27 * 128  # num_blocks x block tokens, from the two boot logs
    print("# 26B QAT W4A16 article: derived figures\n")
    print("## Repack verify (verify_report.json)\n")
    groups = 0
    for kind, a in sorted(verify["quantized"].items()):
        groups += a["groups"]
        print(f"- {kind}: {a['groups']:,} groups, {a['groups_level_mismatch']} off the grid, "
              f"{100 * a['share_bit_identical']:.1f}% bit-identical, worst {a['max_rel_err']:.1e}")
    print(f"- all quantized tensors: {groups:,} groups")
    print(f"- copied: {verify['copied']['byte_identical']} of {verify['copied']['tensors']} byte-identical\n")
    print("## Accuracy, W4A16 against FP8 (2026-09-26-moe2-VS-FP8.json)\n")
    for name, label in (("2026-09-26-moe2-VS-FP8.json", "FP8 (ref) against TPU W4A16"),
                        ("2026-09-26-gpu-l4-VS-TPU.json", "TPU W4A16 (ref) against L4 W4A16")):
        print(f"### {label}\n")
        for arm in load(name).values():
            for group, g in arm.items():
                print(f"- {group}: ref {100 * g['acc_ref']:.1f}%, test {100 * g['acc_test']:.1f}%, "
                      f"difference {100 * g['diff']:+.1f} points ({100 * g['diff_lo']:+.1f} to "
                      f"{100 * g['diff_hi']:+.1f}); {g['ref_right_test_wrong']} right->wrong, "
                      f"{g['ref_wrong_test_right']} wrong->right")
        print()
    print("\n## Capacity and speed\n")
    print(f"- KV tokens: W4A16 {kv_w4:,}, FP8 {kv_fp8:,}, ratio {kv_w4 / kv_fp8:.1f}x")
    print(f"- output tok/s: W4A16 {w4}, FP8 {fp8}, ratio {w4 / fp8:.2f}x")
    kv_per_tok = 8 * 2 * 256 * 30  # heads x (K, V) x head dim x layers, 1-byte fp8
    print(f"- KV cost at fp8: {kv_per_tok / 1024:.0f} KiB/token; W4A16 pool {kv_w4 * kv_per_tok / 2**30:.2f} GiB")
    print("\n## Cost per million output tokens (arithmetic)\n")
    for name, rate in RATES.items():
        for build, tps in (("W4A16", w4), ("FP8", fp8)):
            print(f"- {name} ${rate:.2f}/chip-hour, {build}: ${rate / (tps * 3600) * 1e6:.2f}")


if __name__ == "__main__":
    main()
