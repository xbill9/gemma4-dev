import json
import subprocess

concurrencies = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
contexts = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

results = []

for c in concurrencies:
    for ctx in contexts:
        print(f"Running benchmark for concurrency {c} and context {ctx}")
        try:
            num_prompts = max(c, 10)

            cmd = f"vllm bench serve --host 35.204.78.225 --port 8000 --model google/gemma-4-E2B-it --dataset-name random --num-prompts {num_prompts} --random-input-len {ctx} --random-output-len 100 --max-concurrency {c}"
            process = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)

            if process.returncode == 0:
                print("Success")
                output = process.stdout

                throughput = None
                for line in output.split("\n"):
                    if "Request throughput (req/s):" in line:
                        throughput = float(line.split(":")[1].strip())

                results.append({"concurrency": c, "context": ctx, "throughput": throughput, "status": "success"})
            else:
                print(f"Benchmark failed for {c} users, {ctx} context. Return code: {process.returncode}")
                print(f"Stderr: {process.stderr}")
                results.append(
                    {"concurrency": c, "context": ctx, "throughput": None, "status": f"failed: {process.returncode}"}
                )
        except subprocess.TimeoutExpired:
            print(f"Benchmark timeout for {c} users, {ctx} context")
            results.append({"concurrency": c, "context": ctx, "throughput": None, "status": "timeout"})
        except Exception as e:
            print(f"Benchmark exception for {c} users, {ctx} context: {e}")
            results.append({"concurrency": c, "context": ctx, "throughput": None, "status": f"exception: {e}"})

with open("benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Benchmark completed. Results saved to benchmark_results.json")
