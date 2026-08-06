import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Load v6e-4 data
with open("benchmark_results.json", "r") as f:
    v6e4_data = json.load(f)
df_v6e4 = pd.DataFrame(v6e4_data)
df_v6e4 = df_v6e4[df_v6e4["status"] == "success"]
df_v6e4 = df_v6e4.rename(columns={"context": "context_len"})
df_v6e4["deployment"] = "GCP TPU v6e-4"

# 2. Load v6e-1 data
df_v6e1 = pd.read_csv("../tpu-12B-v6e1-devops-agent/grid_benchmark_results.csv")
df_v6e1 = df_v6e1[["concurrency", "context_len", "throughput"]].dropna()
df_v6e1["deployment"] = "GCP TPU v6e-1"

# 3. Load L4 data
df_l4 = pd.read_csv("../gpu-12B-L4-devops-agent/benchmark_sweep_results.csv")
df_l4 = df_l4.rename(columns={"context_size": "context_len", "req_per_sec": "throughput"})
df_l4 = df_l4[["concurrency", "context_len", "throughput"]].dropna()
df_l4["deployment"] = "GCP GPU L4 (x8)"

# 4. Load QAT L4 data
df_qat = pd.read_csv("../gpu-12B-qat-L4-devops-agent/benchmark_sweep_results.csv")
df_qat = df_qat.rename(columns={"context_size": "context_len", "req_per_sec": "throughput"})
df_qat = df_qat[["concurrency", "context_len", "throughput"]].dropna()
df_qat["deployment"] = "GCP GPU QAT L4 (x8)"

# 5. Load AWS Inferentia QAT data
df_aws = pd.read_csv("/home/xbill/gemma4-tips-aws/gpu-12B-qat-inf-devops-agent/benchmark_sweep_results.csv")
df_aws = df_aws.rename(columns={"context_size": "context_len", "req_per_sec": "throughput"})
df_aws = df_aws[["concurrency", "context_len", "throughput"]].dropna()
df_aws["deployment"] = "AWS Inferentia QAT"

# 6. Load Azure ACA QAT data
df_azure = pd.read_csv("/home/xbill/gemma4-tips-azure/gpu-12B-qat-aca-devops-agent/benchmark_sweep_results.csv")
df_azure = df_azure.rename(columns={"context_size": "context_len", "req_per_sec": "throughput"})
df_azure = df_azure[["concurrency", "context_len", "throughput"]].dropna()
df_azure["deployment"] = "Azure ACA QAT"

# Combine data
df_all = pd.concat([df_v6e4, df_v6e1, df_l4, df_qat, df_aws, df_azure], ignore_index=True)

# We want to compare across specific context lengths, e.g., 128, 1024, 8192
target_contexts = [128, 1024, 8192]

fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)

for i, ctx in enumerate(target_contexts):
    ax = axes[i]
    subset = df_all[df_all["context_len"] == ctx]

    sns.lineplot(data=subset, x="concurrency", y="throughput", hue="deployment", marker="o", ax=ax)
    ax.set_title(f"Context Length: {ctx}")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Throughput (req/s)")
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", ls="--", alpha=0.5)

plt.suptitle("Gemma 4 2B Throughput: GCP (TPU vs L4) vs AWS (Inferentia) vs Azure (ACA)", fontsize=16)
plt.tight_layout()
plt.savefig("comparison_plot_v3.png", dpi=300)

# Generate markdown report
report = "# Gemma 4 2B Deployment Comparison (Cross-Cloud)\n\n"
report += "Comparing throughput (requests/second) across GCP hardware deployments against AWS (Inferentia QAT) and Azure (Container Apps QAT).\n\n"

for ctx in target_contexts:
    report += f"### Context Length: {ctx} tokens\n"
    subset = df_all[df_all["context_len"] == ctx]
    pivot = subset.groupby(["concurrency", "deployment"])["throughput"].mean().unstack()
    report += pivot.to_markdown() + "\n\n"

report += "![Comparison Plot](/home/xbill/.gemini/antigravity-cli/brain/c8b3fe3d-b59c-4eaa-91ed-44864e64c16b/comparison_plot_v3.png)\n"

with open(
    "/home/xbill/.gemini/antigravity-cli/brain/c8b3fe3d-b59c-4eaa-91ed-44864e64c16b/deployment_comparison_v3.md", "w"
) as f:
    f.write(report)

print("Comparison completed.")
