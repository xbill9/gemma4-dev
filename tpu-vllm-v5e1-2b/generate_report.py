#!/usr/bin/env python3
import csv
import os


def main():
    csv_file = "grid_benchmark_results.csv"
    if not os.path.exists(csv_file):
        print(f"❌ Error: {csv_file} does not exist.")
        return

    concurrencies = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    contexts = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    throughput_data = {}
    latency_data = {}

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            c = int(row["concurrency"])
            ctx = int(row["context_len"])
            status = row["status"]

            if status == "success":
                throughput_data[(c, ctx)] = f"{float(row['throughput_req_sec']):.2f}"
                latency_data[(c, ctx)] = f"{float(row['avg_latency_s']):.3f}s"
            elif "skipped" in status:
                throughput_data[(c, ctx)] = "Skip"
                latency_data[(c, ctx)] = "Skip"
            else:
                throughput_data[(c, ctx)] = "Fail"
                latency_data[(c, ctx)] = "Fail"

    # Helper to format markdown table
    def build_markdown_table(data_map, title):
        headers = ["Concurrency / Context"] + [f"{ctx}" for ctx in contexts]
        lines = []
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for c in concurrencies:
            row = [f"**{c} Users**"]
            for ctx in contexts:
                val = data_map.get((c, ctx), "-")
                row.append(val)
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        return "\n".join(lines)

    report = []
    report.append("# 📈 Gemma 4 TPU Benchmarking Sweep Report (v5e-1)\n")
    report.append(build_markdown_table(throughput_data, "Throughput (Request/sec)"))
    report.append("\n" + build_markdown_table(latency_data, "Average Request Latency (seconds)"))

    report_file = "benchmark_tables.md"
    with open(report_file, "w") as f:
        f.write("\n".join(report))
    print(f"💾 Saved Markdown report to: {report_file}")

    # Copy to artifacts directory
    artifact_file = "/home/xbill/.gemini/antigravity-cli/brain/c482e7a6-8d62-45be-a43a-ea5de1c760c4/benchmark_tables.md"
    with open(artifact_file, "w") as f:
        f.write("\n".join(report))
    print(f"💾 Report saved as artifact to: {artifact_file}")


if __name__ == "__main__":
    main()
