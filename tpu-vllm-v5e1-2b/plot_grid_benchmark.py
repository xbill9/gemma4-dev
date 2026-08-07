#!/usr/bin/env python3
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    csv_file = "grid_benchmark_results.csv"
    if not os.path.exists(csv_file):
        print(f"❌ Error: {csv_file} does not exist yet. Run the benchmark sweep first.")
        return

    df = pd.read_csv(csv_file)

    # Filter for successful runs only
    df_success = df[df["status"] == "success"].copy()

    if df_success.empty:
        print("No successful runs found in results file to plot.")
        return

    print(f"📊 Loading {len(df_success)} successful sweep data points...")

    # Plot 1: Heatmap of Throughput (Concurrency vs Context Window)
    plt.figure(figsize=(14, 10))
    pivot_throughput = df_success.pivot(index="concurrency", columns="context_len", values="throughput_req_sec")
    pivot_throughput = pivot_throughput.sort_index(ascending=False).sort_index(axis=1)

    sns.heatmap(pivot_throughput, annot=True, fmt=".2f", cmap="viridis", cbar_kws={"label": "Throughput (req/s)"})
    plt.title(
        "Gemma 4 v5e-1 Serving Throughput (req/s)\nConcurrency vs Context Window",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.ylabel("Concurrency (Users)", fontsize=12)
    plt.xlabel("Context Window (Tokens)", fontsize=12)
    plt.tight_layout()
    plt.savefig("throughput_heatmap.png", dpi=300)
    print("💾 Saved: throughput_heatmap.png")

    # Plot 2: Heatmap of Latency
    plt.figure(figsize=(14, 10))
    pivot_latency = df_success.pivot(index="concurrency", columns="context_len", values="avg_latency_s")
    pivot_latency = pivot_latency.sort_index(ascending=False).sort_index(axis=1)

    sns.heatmap(pivot_latency, annot=True, fmt=".3f", cmap="magma", cbar_kws={"label": "Avg Latency (s)"})
    plt.title(
        "Gemma 4 v5e-1 Average Latency (s)\nConcurrency vs Context Window", fontsize=16, fontweight="bold", pad=20
    )
    plt.ylabel("Concurrency (Users)", fontsize=12)
    plt.xlabel("Context Window (Tokens)", fontsize=12)
    plt.tight_layout()
    plt.savefig("latency_heatmap.png", dpi=300)
    print("💾 Saved: latency_heatmap.png")

    # Write a Markdown report summary
    report = []
    report.append("# 📈 Gemma 4 TPU Benchmarking Sweep Report\n")
    report.append("## 📊 Throughput (req/s) Table\n")

    # Format and append throughput table
    report.append(pivot_throughput.sort_index(ascending=True).to_markdown())
    report.append("\n\n## ⏱️ Average Request Latency (s) Table\n")

    # Format and append latency table
    report.append(pivot_latency.sort_index(ascending=True).to_markdown())

    report_file = "benchmark_tables.md"
    with open(report_file, "w") as f:
        f.write("\n".join(report))
    print(f"💾 Saved Markdown report to: {report_file}")

    # Optional second copy for an external viewer. The report is already written to the CWD
    # above, so this is off unless ARTIFACT_DIR is set. It used to be a hardcoded UUID path
    # inside another tool's scratch directory, so it only ever landed on one machine.
    artifact_dir = os.getenv("ARTIFACT_DIR")
    if artifact_dir:
        os.makedirs(artifact_dir, exist_ok=True)
        artifact_path = os.path.join(artifact_dir, report_file)
        with open(artifact_path, "w") as f:
            f.write("\n".join(report))
        print(f"💾 Report also saved as artifact: [{report_file}](file://{artifact_path})")


if __name__ == "__main__":
    main()
