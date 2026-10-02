#!/usr/bin/env python3
"""Parses the raw `vllm bench serve` sweep output into a CSV.

The sweep is produced by running a concurrency x context grid inside the vllm-tpu
container on the TPU VM (see the sweep_remote.sh flow). Each grid point is delimited by
a `===POINT ...===` marker; this turns the human-readable benchmark blocks between the
markers into one row per point.

Usage:  python parse_sweep.py <raw_output_file> <output_csv>
"""

import csv
import re
import sys
from typing import Any

# Benchmark label -> CSV column. Only these are pulled out of each block.
METRICS = {
    "Successful requests": "successful_requests",
    "Failed requests": "failed_requests",
    "Benchmark duration (s)": "duration_s",
    "Total input tokens": "total_input_tokens",
    "Total generated tokens": "total_generated_tokens",
    "Request throughput (req/s)": "req_per_s",
    "Output token throughput (tok/s)": "output_tok_per_s",
    "Total token throughput (tok/s)": "total_tok_per_s",
    "Mean TTFT (ms)": "mean_ttft_ms",
    "Median TTFT (ms)": "median_ttft_ms",
    "P99 TTFT (ms)": "p99_ttft_ms",
    "Mean TPOT (ms)": "mean_tpot_ms",
    "Mean ITL (ms)": "mean_itl_ms",
}

COLUMNS = ["concurrency", "input_len", "num_prompts"] + list(METRICS.values())

POINT_RE = re.compile(r"===POINT concurrency=(\d+) input_len=(\d+) num_prompts=(\d+)===")


def parse(raw: str) -> list[dict]:
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in raw.splitlines():
        match = POINT_RE.search(line)
        if match:
            if current:
                rows.append(current)
            current = {
                "concurrency": int(match.group(1)),
                "input_len": int(match.group(2)),
                "num_prompts": int(match.group(3)),
            }
            continue
        if current is None:
            continue
        for label, column in METRICS.items():
            # Benchmark lines look like "Request throughput (req/s):   1.88"
            if line.strip().startswith(label):
                value = line.split(":", 1)[1].strip()
                try:
                    current[column] = float(value)
                except ValueError:
                    current[column] = None
                break
    if current:
        rows.append(current)
    return rows


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2

    with open(sys.argv[1], "r") as f:
        rows = parse(f.read())

    if not rows:
        print("❌ No grid points found — did the sweep produce ===POINT=== markers?")
        return 1

    with open(sys.argv[2], "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c) for c in COLUMNS})

    complete = [r for r in rows if r.get("req_per_s") is not None]
    print(f"✅ Wrote {len(rows)} grid points to {sys.argv[2]} ({len(complete)} with throughput data).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
