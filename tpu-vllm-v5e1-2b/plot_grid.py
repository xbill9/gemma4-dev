import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_and_plot():
    csv_path = "benchmarks/runs/undated-vllm-grid-b-v6e1/grid_benchmark_results.csv"
    if not os.path.exists(csv_path):
        print("CSV results file not found!")
        return

    # Parse rows starting from the header match or manually filtering for 5-column format
    data = []
    with open(csv_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 5:
                if parts[0] == "concurrency":
                    continue
                try:
                    data.append(
                        {
                            "concurrency": int(parts[0]),
                            "context_len": int(parts[1]),
                            "throughput": float(parts[2]),
                            "avg_latency": float(parts[3]),
                            "status": parts[4],
                        }
                    )
                except ValueError:
                    pass

    df = pd.DataFrame(data)
    df_success = df[df["status"] == "success"].copy()

    if df_success.empty:
        print("No successful points to plot!")
        return

    # Drop duplicate grid entries (keep the latest one)
    df_success = df_success.drop_duplicates(subset=["concurrency", "context_len"], keep="last")

    # Where the heatmaps land. Defaults to the CWD, matching the convention that writers use
    # bare filenames in the working directory so an archived run dir is never overwritten.
    # This was a hardcoded UUID path inside another tool's scratch directory.
    artifact_dir = os.getenv("ARTIFACT_DIR", ".")
    os.makedirs(artifact_dir, exist_ok=True)

    # Style configuration
    sns.set_theme(style="whitegrid")

    # 1. Throughput Heatmap
    plt.figure(figsize=(12, 8))
    pivot_throughput = df_success.pivot(index="concurrency", columns="context_len", values="throughput")
    pivot_throughput = pivot_throughput.sort_index(ascending=False).sort_index(axis=1)

    sns.heatmap(pivot_throughput, annot=True, fmt=".1f", cmap="crest", cbar_kws={"label": "Throughput (req/s)"})
    plt.title(
        "Gemma 4 2B (TPU v6e-4) Serving Throughput\nConcurrency vs Context Window (req/s)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.ylabel("Concurrency (Users)", fontsize=12)
    plt.xlabel("Context Window Length (Tokens)", fontsize=12)
    plt.tight_layout()

    plt.savefig("throughput_heatmap.png", dpi=300)
    plt.savefig(os.path.join(artifact_dir, "throughput_heatmap.png"), dpi=300)
    plt.close()

    # 2. Latency Heatmap
    plt.figure(figsize=(12, 8))
    pivot_latency = df_success.pivot(index="concurrency", columns="context_len", values="avg_latency")
    pivot_latency = pivot_latency.sort_index(ascending=False).sort_index(axis=1)

    sns.heatmap(pivot_latency, annot=True, fmt=".3f", cmap="flare", cbar_kws={"label": "Avg Latency (s)"})
    plt.title(
        "Gemma 4 2B (TPU v6e-4) Average Latency\nConcurrency vs Context Window (seconds)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.ylabel("Concurrency (Users)", fontsize=12)
    plt.xlabel("Context Window Length (Tokens)", fontsize=12)
    plt.tight_layout()

    plt.savefig("latency_heatmap.png", dpi=300)
    plt.savefig(os.path.join(artifact_dir, "latency_heatmap.png"), dpi=300)
    plt.close()

    print("Plots generated successfully!")


if __name__ == "__main__":
    parse_and_plot()
