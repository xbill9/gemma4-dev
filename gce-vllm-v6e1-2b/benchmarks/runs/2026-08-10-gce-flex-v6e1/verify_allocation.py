"""Verify what the v6e-1 deployment ACTUALLY allocated, from the boot log.

This exists because of the rule in `@QUANTIZATION.md`: the fp8 KV flag was once accepted at
the CLI, echoed in `non-default args`, praised in an engine log line, reported in `/metrics`,
and allocated a genuinely fp8 tensor — five independent signals it had worked — while
delivering nothing. **Verify from the boot allocation, never from the flag being accepted.**

So this script does not ask the server what it was configured with. It reads the allocation
the engine printed and checks it against arithmetic derived from `@MODELS.md` and
`@HARDWARE.md`:

  * E2B KV is 18,432 B/token at bf16 (12 sliding x 1,024 + 3 full x 2,048).
  * v6e-1 has 31.24 GiB total HBM; E2B weights are 8.97 GiB resident.
  * The only recorded v6e allocation to date is a 65,536-context run at 19.79 GiB / 1,151,744
    KV tokens. This rig now runs 32,768, and nothing has measured that.

Usage:
    python3 verify_allocation.py --log boot.log [--json out.json]
    ... | python3 verify_allocation.py --log -

Every check reports PASS / FAIL / UNKNOWN separately. UNKNOWN means the line was not found,
which is itself a finding — an absent allocation log is a cleaner negative than a suspicious
number, and this script never infers a value it did not read.
"""

import argparse
import json
import re
import sys
from typing import Any, Dict, Optional

# --- Constants: derived facts, not measurements. See @MODELS.md / @HARDWARE.md. ---
KV_BYTES_PER_TOKEN_BF16 = 18_432  # 12 sliding x 1 KV head x 256 x 2 x 2B + 3 full x 512 x 2 x 2B
KV_BYTES_PER_TOKEN_FP8 = 9_216  # what fp8 WOULD cost if it ever engaged
WEIGHTS_GIB_BF16 = 8.97
TOTAL_HBM_GIB_V6E1 = 31.24
EXPECTED_MAX_MODEL_LEN = 32_768
GIB = 1024**3

# Blocks-per-request is held at 512, so block_size scales with max_model_len:
# 16384 -> 32, 32768 -> 64, 65536 -> 128.
EXPECTED_BLOCK_SIZE = 64

PATTERNS = {
    "kv_cache_tokens": [
        r"GPU KV cache size:\s*([\d,]+)\s*tokens",
        r"kv_cache_size_tokens[=:\s]+([\d,]+)",
        r"KV cache size:\s*([\d,]+)\s*tokens",
    ],
    "num_gpu_blocks": [
        r"#\s*GPU blocks:\s*([\d,]+)",
        r"num_gpu_blocks[=:\s]+([\d,]+)",
    ],
    "total_hbm_used_gb": [
        r"total_hbm_used_gb[=:\s]+([\d.]+)",
        r"hbm=\[\(([\d.]+),",
    ],
    "total_hbm_avail_gb": [
        r"total_hbm_avail_gb[=:\s]+([\d.]+)",
    ],
    "max_model_len": [
        # vLLM prints this inside a dict repr — `'max_model_len': 16384` — so the quote and
        # colon must be part of the pattern. The original [=:\s]+ class missed it and this
        # check silently returned UNKNOWN on the 2026-08-10 run that WAS misconfigured;
        # block_size is what actually caught it. Keep both checks: they are independent
        # derivations of the same setting, and that redundancy is what saved the run.
        r"['\"]?max_model_len['\"]?\s*[=:]\s*([\d,]+)",
        r"--max[-_]model[-_]len[=\s]+([\d,]+)",
    ],
    "block_size": [
        r"block_size[=:\s]+([\d,]+)",
    ],
    "num_kv_cache_groups": [
        r"num_kv_cache_groups[=:\s]+(\d+)",
    ],
    "num_kv_cache_tensors": [
        r"num_kv_cache_tensors[=:\s]+(\d+)",
    ],
}


def find_first(text: str, patterns: list) -> Optional[str]:
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def to_num(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def extract(text: str) -> Dict[str, Optional[float]]:
    return {k: to_num(find_first(text, pats)) for k, pats in PATTERNS.items()}


def check(name: str, ok: Optional[bool], detail: str) -> Dict[str, Any]:
    verdict = "UNKNOWN" if ok is None else ("PASS" if ok else "FAIL")
    return {"check": name, "verdict": verdict, "detail": detail}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to the vLLM boot log, or '-' for stdin")
    ap.add_argument("--json", help="Write the structured result here")
    args = ap.parse_args()

    text = sys.stdin.read() if args.log == "-" else open(args.log, "r", errors="replace").read()
    v = extract(text)
    checks = []

    # 1. Did the retargeted max_model_len actually take effect?
    mml = v["max_model_len"]
    checks.append(
        check(
            "max_model_len is 32768 (the tpu.env value, not the 16384 fork default)",
            None if mml is None else int(mml) == EXPECTED_MAX_MODEL_LEN,
            f"read {int(mml):,}" if mml else "not found in log",
        )
    )

    # 2. block_size must be DERIVED, not set. 32768 / 512 blocks-per-request = 64.
    bs = v["block_size"]
    checks.append(
        check(
            "block_size derived to 64 at 32768 context",
            None if bs is None else int(bs) == EXPECTED_BLOCK_SIZE,
            f"read {int(bs)}, expected {EXPECTED_BLOCK_SIZE}" if bs else "not found in log",
        )
    )

    # 3. THE fp8 CHECK. Does the KV pool match bf16 arithmetic or fp8 arithmetic?
    #    The v6e demo logged "Automatically using fp8_e5m2 for FP8 KV cache" and allocated bf16.
    toks = v["kv_cache_tokens"]
    if toks:
        pool_bytes_bf16 = toks * KV_BYTES_PER_TOKEN_BF16
        pool_bytes_fp8 = toks * KV_BYTES_PER_TOKEN_FP8
        avail = v["total_hbm_avail_gb"]
        if avail:
            err_bf16 = abs(pool_bytes_bf16 / GIB - avail) / avail
            err_fp8 = abs(pool_bytes_fp8 / GIB - avail) / avail
            checks.append(
                check(
                    "KV pool matches bf16 arithmetic (fp8 did NOT engage)",
                    err_bf16 < 0.02,
                    f"{int(toks):,} tokens vs {avail:.2f} GiB avail -> "
                    f"bf16 model off by {err_bf16 * 100:.2f}%, fp8 model off by {err_fp8 * 100:.1f}%",
                )
            )
        else:
            checks.append(
                check(
                    "KV pool matches bf16 arithmetic (fp8 did NOT engage)",
                    None,
                    f"{int(toks):,} tokens found, but no total_hbm_avail_gb to check against; "
                    f"bf16 predicts {pool_bytes_bf16 / GIB:.2f} GiB, fp8 predicts {pool_bytes_fp8 / GIB:.2f} GiB",
                )
            )
    else:
        checks.append(check("KV pool matches bf16 arithmetic (fp8 did NOT engage)", None, "KV token count not found"))

    # 4. Weights resident should be the known bf16 figure. If this moved, a quant route engaged.
    used = v["total_hbm_used_gb"]
    checks.append(
        check(
            "weights resident at the bf16 figure (8.97 GiB) — no quantization engaged",
            None if used is None else abs(used - WEIGHTS_GIB_BF16) < 0.5,
            f"read {used:.2f} GiB, expected ~{WEIGHTS_GIB_BF16}" if used else "not found in log",
        )
    )

    # 5. The 2.9x: sliding windows are expected to be OFF (one cache group for all 15 tensors).
    groups, tensors = v["num_kv_cache_groups"], v["num_kv_cache_tensors"]
    checks.append(
        check(
            "sliding windows still disabled (num_kv_cache_groups=1) — the 2.9x is still forgone",
            None if groups is None else int(groups) == 1,
            f"groups={int(groups) if groups else '?'}, tensors={int(tensors) if tensors else '?'} "
            f"(groups>1 would mean the upstream TODO landed — a large win)",
        )
    )

    # 6. Did the engine claim fp8 in prose? Recorded, not trusted — see the module docstring.
    fp8_prose = bool(re.search(r"fp8|FP8", text))
    checks.append(
        check(
            "engine prose mentions fp8 (informational — prose is not evidence)",
            None,
            "yes — cross-check against the arithmetic above, this line has lied before"
            if fp8_prose
            else "no fp8 mention in log",
        )
    )

    result = {"extracted": v, "checks": checks}

    print("=" * 78)
    print("v6e-1 allocation verification — read from the boot log, not from the flags")
    print("=" * 78)
    for k, val in v.items():
        shown = f"{val:,.2f}".rstrip("0").rstrip(".") if isinstance(val, float) else str(val)
        print(f"  {k:24s} {shown if val is not None else 'NOT FOUND'}")
    print()
    for c in checks:
        mark = {"PASS": "PASS", "FAIL": "FAIL", "UNKNOWN": "????"}[c["verdict"]]
        print(f"  [{mark}] {c['check']}")
        print(f"         {c['detail']}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote {args.json}")

    return 1 if any(c["verdict"] == "FAIL" for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
