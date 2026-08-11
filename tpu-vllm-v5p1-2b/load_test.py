import argparse
import asyncio
import statistics
import time

import httpx


# Configuration through command-line arguments
async def main():
    parser = argparse.ArgumentParser(description="Load testing script for vLLM endpoint.")
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="The vLLM endpoint URL, e.g. http://<TPU_VM_IP>:8000/v1/completions (get it with `make endpoint`).",
    )
    parser.add_argument(
        "--model", type=str, default="google/gemma-4-E2B-it", help="The model to use for the load test."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain the architecture of TPU v5p and why it is optimized for JAX.",
        help="The prompt to send to the model.",
    )
    parser.add_argument("--num-requests", type=int, default=20, help="The total number of requests to send.")
    parser.add_argument("--concurrency", type=int, default=4, help="The number of concurrent requests.")
    args = parser.parse_args()

    print(f"🚀 Starting load test on {args.url}")
    print(f"   Model: {args.model}")
    print(f"   Concurrent Workers: {args.concurrency}")
    print(f"   Total Requests: {args.num_requests}\n")

    start_test = time.time()
    async with httpx.AsyncClient() as client:
        semaphore = asyncio.Semaphore(args.concurrency)
        tasks = [send_request(client, semaphore, i, args) for i in range(args.num_requests)]
        results = await asyncio.gather(*tasks)

    # Filter out failures
    latencies = [latency for latency in results if latency is not None]
    total_test_time = time.time() - start_test

    if latencies:
        avg_latency = statistics.mean(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        throughput = len(latencies) / total_test_time

        print("\n✅ Load Test Results:")
        print("   -------------------")
        print(f"   Total Successes:  {len(latencies)}/{args.num_requests}")
        print(f"   Total Time:       {total_test_time:.2f}s")
        print(f"   Average Latency:  {avg_latency:.2f}s")
        print(f"   P95 Latency:      {p95_latency:.2f}s")
        print(f"   Throughput:       {throughput:.2f} req/s")
    else:
        print("\n❌ All requests failed. Check if the endpoint is reachable.")


async def send_request(client, semaphore, request_id, args):
    async with semaphore:
        print(f"  [#{request_id}] Sending request...")
        start = time.time()
        try:
            response = await client.post(
                args.url,
                json={
                    "model": args.model,
                    "messages": [{"role": "user", "content": args.prompt}],
                    "max_tokens": 128,
                    "temperature": 0.2,
                },
                timeout=60,
            )
            response.raise_for_status()
            latency = time.time() - start
            print(f"  [#{request_id}] Completed in {latency:.2f}s")
            return latency
        except Exception as e:
            print(f"  [#{request_id}] Failed: {e}")
            return None


if __name__ == "__main__":
    asyncio.run(main())
