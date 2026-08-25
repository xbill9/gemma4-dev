import asyncio
import csv
import os
import statistics
import sys
import time

import httpx

# Configuration.
#
# No default host. The IP that used to sit here belonged to an instance the PARENT rig held
# during one E2B sweep; flex-start instances are deleted at MAX_RUN_DURATION and their
# addresses are recycled, so a stale default benchmarks a stranger rather than failing.
_HOST = os.getenv("VLLM_HOST", "")
if not _HOST:
    sys.exit("Set VLLM_HOST to this rig's live instance IP (see get_vllm_endpoint).")
URL = f"http://{_HOST}:8000/v1/chat/completions"
MODEL = os.getenv("MODEL_NAME", "google/gemma-4-31B-it")
RESULTS_FILE = "grid_benchmark_results.csv"

# Cells above this many (concurrency x context) tokens cannot be resident at once and are
# skipped rather than run into an OOM. THE VALUE IS CHECKPOINT-SPECIFIC and was 789,760 while
# this rig served E2B — 4.3x too generous for 31B, which pays 960 KiB/token at TP=8 against
# E2B's 18 KiB (see tpu.env). 184,000 is the arithmetic KV-token count for a 31B on a v6e-8,
# NOT a measurement: derive it again from the boot log's KV pool before trusting a skip.
KV_TOKEN_LIMIT = int(os.getenv("KV_TOKEN_LIMIT", "184000"))


# Generate a prompt of approximately `ctx` words to simulate context length
def generate_prompt(ctx):
    return "word " * ctx


async def send_request(client, semaphore, prompt):
    async with semaphore:
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 10,
            "temperature": 0.0,
        }
        start = time.perf_counter()
        try:
            response = await client.post(URL, json=payload, timeout=90)
            if response.status_code == 200:
                return time.perf_counter() - start
            else:
                return None
        except Exception:
            return None


async def run_benchmark(concurrency, context_len):
    prompt = generate_prompt(context_len)
    num_requests = max(concurrency, 4)
    # Capping num_requests to avoid spamming at high concurrency
    if concurrency > 128:
        num_requests = concurrency

    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:
        start_time = time.perf_counter()
        tasks = [send_request(client, semaphore, prompt) for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time

        latencies = [latency for latency in results if latency is not None]
        if not latencies:
            return 0.0, 0.0, "failed"

        throughput = len(latencies) / total_time
        avg_latency = statistics.mean(latencies)
        return throughput, avg_latency, "success"


async def main():
    print(f"📡 Running fast native sweep against: {URL}")

    concurrencies = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    contexts = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concurrency", "context_len", "throughput_req_sec", "avg_latency_s", "status"])

    for c in concurrencies:
        for ctx in contexts:
            # Skip combinations exceeding the physical KV cache limits to prevent OOM
            if c * ctx > KV_TOKEN_LIMIT:
                print(f"⏭️ Skipping c={c}, ctx={ctx} (Exceeds KV cache limit)")
                with open(RESULTS_FILE, "a", newline="") as f:
                    csv.writer(f).writerow([c, ctx, 0.0, 0.0, "skipped_capacity_limit"])
                continue

            print(f"🏃 Concurrency: {c:4d} | Context: {ctx:5d} tokens ... ", end="", flush=True)
            throughput, avg_latency, status = await run_benchmark(c, ctx)

            if status == "success":
                print(f"✅ Success | {throughput:6.2f} req/s | Latency: {avg_latency:6.3f}s")
            else:
                print("❌ Failed")

            with open(RESULTS_FILE, "a", newline="") as f:
                csv.writer(f).writerow([c, ctx, throughput, avg_latency, status])


if __name__ == "__main__":
    asyncio.run(main())
