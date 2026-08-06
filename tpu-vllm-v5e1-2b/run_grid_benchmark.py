#!/usr/bin/env python3
import os
import subprocess
import sys
import time

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "aisprint-491218")
ZONE = "europe-west4-a"


def get_tpu_ip():
    cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "describe",
        "vllm-gemma4-qr-node",
        f"--project={PROJECT_ID}",
        f"--zone={ZONE}",
        "--format=value(networkEndpoints[0].accessConfig.externalIp)",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        ip = res.stdout.strip()
        if not ip:
            # Fallback to internal IP
            cmd[-1] = "value(networkEndpoints[0].ipAddress)"
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            ip = res.stdout.strip()
        return ip
    except Exception as e:
        print(f"Error fetching TPU IP: {e}", file=sys.stderr)
        return None


def main():
    ip = get_tpu_ip()
    if not ip:
        print("❌ Error: Could not determine active TPU VM IP address. Make sure the resource is ACTIVE.")
        sys.exit(1)

    print(f"📡 Found active TPU VM endpoint at: http://{ip}:8000")

    concurrencies = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    contexts = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    results_file = "grid_benchmark_results.csv"

    # Write header if new file
    if not os.path.exists(results_file):
        with open(results_file, "w") as f:
            f.write("concurrency,context_len,throughput_req_sec,avg_latency_s,status\n")

    print(f"🚀 Starting grid benchmark sweep. Results will be saved dynamically to: {results_file}")

    for c in concurrencies:
        for ctx in contexts:
            # Skip extremely high memory configurations that will definitely OOM to save time
            # For example, 2048 concurrent users with 16K context = 33 million active tokens in KV cache,
            # which exceeds the physical KV cache size of 789,760 tokens and will prompt capacity blocks or OOM.
            if c * ctx > 789760:
                print(f"⏭️ Skipping concurrency={c}, context={ctx} (Exceeds max KV cache limit of 789,760 tokens)")
                with open(results_file, "a") as f:
                    f.write(f"{c},{ctx},0.0,0.0,skipped_capacity_limit\n")
                continue

            print(f"🏃 Concurrency: {c:4d} | Context Window: {ctx:5d} tokens ... ", end="", flush=True)
            num_prompts = max(c, 10)

            # Limit prompts for extremely high concurrency to speed up execution
            if c > 256:
                num_prompts = c

            cmd = (
                f"vllm bench serve "
                f"--host {ip} "
                f"--port 8000 "
                f"--model google/gemma-4-E2B-it "
                f"--dataset-name random "
                f"--num-prompts {num_prompts} "
                f"--random-input-len {ctx} "
                f"--random-output-len 10 "  # Short output length for speed
                f"--max-concurrency {c}"
            )

            start_run = time.time()
            try:
                # 90 second timeout per benchmark run
                res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=90)
                elapsed = time.time() - start_run

                if res.returncode == 0:
                    output = res.stdout
                    throughput = 0.0
                    avg_latency = 0.0

                    for line in output.splitlines():
                        if "Request throughput (req/s):" in line:
                            try:
                                throughput = float(line.split(":")[1].strip())
                            except Exception:
                                pass
                        if "Average latency (s):" in line or "Average request latency (s):" in line:
                            try:
                                avg_latency = float(line.split(":")[1].strip())
                            except Exception:
                                pass

                    print(f"✅ Success | {throughput:6.2f} req/s | Latency: {avg_latency:6.3f}s (Took {elapsed:.1f}s)")
                    with open(results_file, "a") as f:
                        f.write(f"{c},{ctx},{throughput:.3f},{avg_latency:.3f},success\n")
                else:
                    err_msg = res.stderr.strip().replace("\n", " ")[:50]
                    print(f"❌ Failed (Code: {res.returncode}) | {err_msg}")
                    with open(results_file, "a") as f:
                        f.write(f"{c},{ctx},0.0,0.0,failed_{res.returncode}\n")
            except subprocess.TimeoutExpired:
                print("⏳ Timeout")
                with open(results_file, "a") as f:
                    f.write(f"{c},{ctx},0.0,0.0,timeout\n")
            except Exception as e:
                print(f"💥 Error: {e}")
                with open(results_file, "a") as f:
                    f.write(f"{c},{ctx},0.0,0.0,error_{type(e).__name__}\n")

    print("\n🏁 Sweep completed!")


if __name__ == "__main__":
    main()
