"""Aggregate `vllm bench serve --save-result` JSON from sweep.sh into results.csv.

One row per (label, input_len, concurrency): the mean of the three repeats, plus
the coefficient of variation of output throughput across them so a noisy cell
is visible rather than averaged away. Columns follow the g4dn twin's
results.csv, with `label`, `reps` and `output_tok_per_s_cv` added.

    python3 aggregate.py            # reads ./bf16 and ./qat, writes ./results.csv
"""

import csv
import glob
import json
import os
import re
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CELL = re.compile(r"c(\d+)-in(\d+)-out(\d+)\.rep(\d+)\.json$")
FIELDS = [
    "label", "input_len", "concurrency", "output_len", "reps", "status",
    "output_tok_per_s", "output_tok_per_s_cv", "per_stream_tok_per_s",
    "request_rate_rps", "ttft_ms_median", "tpot_ms_median", "itl_ms_median", "failed",
]


def main() -> None:
    cells = defaultdict(list)
    for label in ("bf16", "qat"):
        for path in glob.glob(os.path.join(HERE, label, "*.json")):
            m = CELL.search(path)
            if m:
                c, i, o, _ = map(int, m.groups())
                with open(path) as fh:
                    cells[(label, i, c, o)].append(json.load(fh))

    rows = []
    for (label, i, c, o), runs in sorted(cells.items()):
        tput = [r["output_throughput"] for r in runs]
        tpot = statistics.mean(r["median_tpot_ms"] for r in runs)
        failed = sum(r.get("failed", 0) for r in runs)
        rows.append({
            "label": label, "input_len": i, "concurrency": c, "output_len": o,
            "reps": len(runs), "status": "ok" if failed == 0 else "failed",
            "output_tok_per_s": round(statistics.mean(tput), 2),
            "output_tok_per_s_cv": round(statistics.pstdev(tput) / statistics.mean(tput) * 100, 2),
            "per_stream_tok_per_s": round(1000 / tpot, 2),
            "request_rate_rps": round(statistics.mean(r["request_throughput"] for r in runs), 4),
            "ttft_ms_median": round(statistics.mean(r["median_ttft_ms"] for r in runs), 1),
            "tpot_ms_median": round(tpot, 3),
            "itl_ms_median": round(statistics.mean(r["median_itl_ms"] for r in runs), 3),
            "failed": failed,
        })

    with open(os.path.join(HERE, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    for r in rows:
        print(
            f"{r['label']:4s} in={r['input_len']:<5d} c={r['concurrency']:<3d} n={r['reps']} "
            f"out={r['output_tok_per_s']:7.1f} (cv {r['output_tok_per_s_cv']:4.1f}%) "
            f"stream={r['per_stream_tok_per_s']:5.1f} ttft={r['ttft_ms_median']:8.0f}ms "
            f"tpot={r['tpot_ms_median']:6.1f}ms failed={r['failed']}"
        )
    print(f"worst-cell cv: {max(r['output_tok_per_s_cv'] for r in rows):.1f}%  cells: {len(rows)}")


if __name__ == "__main__":
    main()
