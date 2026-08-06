#!/usr/bin/env python3
import csv
import os


def load_data(filepath, is_v6e1=True):
    data: dict[tuple[int, int], dict[str, float]] = {}
    if not os.path.exists(filepath):
        return data
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            c = int(row["concurrency"])
            ctx = int(row["context_len"])

            if is_v6e1:
                status = row["status"]
                if status == "success":
                    data[(c, ctx)] = {
                        "throughput": float(row["throughput_req_sec"]),
                        "latency": float(row["avg_latency_s"]),
                    }
            else:
                throughput_val = row["throughput"]
                latency_val = row["avg_latency"]

                # Check for skipped or invalid entries
                if throughput_val and latency_val and float(throughput_val) > 0:
                    data[(c, ctx)] = {"throughput": float(throughput_val), "latency": float(latency_val)}
    return data


def main():
    v6e1_file = "grid_benchmark_results.csv"
    v6e4_file = "../tpu-2B-v6e4-devops-agent/grid_benchmark_results.csv"

    v6e1_data = load_data(v6e1_file, is_v6e1=True)
    v6e4_data = load_data(v6e4_file, is_v6e1=False)

    if not v6e1_data or not v6e4_data:
        print("❌ Error: Missing benchmark files.")
        return

    # Compare at context length 128
    ctx_to_compare = 128
    concurrencies = [1, 8, 32, 128, 256, 512, 1024, 2048]

    print(f"### Gemma 4 2B Performance Comparison (v6e-1 vs v6e-4) at Context Size = {ctx_to_compare}")
    print("| Concurrency | v6e-1 req/s | v6e-4 req/s | Throughput Scaling | v6e-1 Latency | v6e-4 Latency |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")

    for c in concurrencies:
        v1 = v6e1_data.get((c, ctx_to_compare))
        v4 = v6e4_data.get((c, ctx_to_compare))

        if v1 and v4:
            scale = v4["throughput"] / v1["throughput"]
            print(
                f"| **{c} Users** | {v1['throughput']:.2f} | {v4['throughput']:.2f} | **{scale:.2f}x** | {v1['latency']:.3f}s | {v4['latency']:.3f}s |"
            )
        else:
            v1_str = f"{v1['throughput']:.2f}" if v1 else "N/A"
            v4_str = f"{v4['throughput']:.2f}" if v4 else "N/A"
            print(f"| **{c} Users** | {v1_str} | {v4_str} | N/A | N/A | N/A |")


if __name__ == "__main__":
    main()
