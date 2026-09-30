import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the JSON results
with open("benchmarks/runs/undated-vllm-grid-a-v6e1/benchmark_results.json", "r") as f:
    results = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(results)

# Filter for successful runs only for plotting throughput values
df_success = df[df["status"] == "success"]

if not df_success.empty:
    # Plot 1: Heatmap of Throughput (Context vs Concurrency)
    plt.figure(figsize=(12, 8))
    pivot_df = df_success.pivot(index="concurrency", columns="context", values="throughput")

    # Sort the index and columns just in case
    pivot_df = pivot_df.sort_index(ascending=False).sort_index(axis=1)

    sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={"label": "Throughput (req/s)"})
    plt.title("vLLM Gemma-4 2B Throughput (req/s)\nConcurrency vs Context Length")
    plt.ylabel("Concurrency")
    plt.xlabel("Context Length (tokens)")
    plt.tight_layout()
    plt.savefig("sweep_throughput_heatmap.png", dpi=300)
    print("Created sweep_throughput_heatmap.png")

    # Plot 2: Line plot of Throughput vs Context Length for different concurrencies
    plt.figure(figsize=(12, 8))
    for c in sorted(df_success["concurrency"].unique()):
        subset = df_success[df_success["concurrency"] == c].sort_values("context")
        plt.plot(subset["context"], subset["throughput"], marker="o", label=f"Concurrency {c}")

    plt.xscale("log", base=2)
    plt.xticks(sorted(df_success["context"].unique()), labels=sorted(df_success["context"].unique()))
    plt.title("Throughput vs Context Length by Concurrency")
    plt.xlabel("Context Length (tokens)")
    plt.ylabel("Throughput (req/s)")
    plt.legend(title="Concurrency", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig("sweep_throughput_lineplot.png", dpi=300)
    print("Created sweep_throughput_lineplot.png")

else:
    print("No successful data points found to plot.")
