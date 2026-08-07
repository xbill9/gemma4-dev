import argparse
import asyncio
import statistics
import time
from datetime import datetime
from typing import Any, Dict

import httpx
import pandas as pd


class vLLMBenchmarkSuite:
    def __init__(self, base_url: str, model: str, output_file: str):
        self.base_url = f"{base_url.rstrip('/')}/v1/completions"
        self.model = model
        self.output_file = output_file
        self.results: list[dict[str, Any]] = []

    async def send_request(self, client: httpx.AsyncClient, prompt: str, max_tokens: int) -> Dict[str, Any]:
        payload = {"model": self.model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0, "stream": False}

        start_time = time.perf_counter()
        try:
            response = await client.post(self.base_url, json=payload, timeout=120)
            end_time = time.perf_counter()

            if response.status_code == 200:
                latency = end_time - start_time
                data = response.json()
                # Try to extract tokens if available in usage
                tokens = data.get("usage", {}).get("completion_tokens", max_tokens)
                return {"success": True, "latency": latency, "tokens": tokens}
            else:
                return {"success": False, "error": f"Status {response.status_code}"}
        except httpx.RequestError as e:
            return {"success": False, "error": str(e)}

    async def run_batch(self, concurrency: int, num_requests: int, prompt: str, max_tokens: int):
        print(f"  🏎️  Running Sweep: Concurrency={concurrency}, Total Requests={num_requests}...")

        async with httpx.AsyncClient() as client:
            # We use a semaphore to control concurrency
            sem = asyncio.Semaphore(concurrency)

            async def wrapped_req():
                async with sem:
                    return await self.send_request(client, prompt, max_tokens)

            start_batch = time.perf_counter()
            results = await asyncio.gather(*[wrapped_req() for _ in range(num_requests)])
            total_time = time.perf_counter() - start_batch

            successes = [r for r in results if r["success"]]
            latencies = [r["latency"] for r in successes]

            if not latencies:
                print("    ❌ Batch failed entirely.")
                return

            avg_lat = statistics.mean(latencies)
            p95_lat = sorted(latencies)[int(len(latencies) * 0.95)]
            throughput = len(successes) / total_time
            tokens_per_sec = sum(r["tokens"] for r in successes) / total_time

            print(f"    ✅ Throughput: {throughput:.2f} req/s | Avg Latency: {avg_lat:.2f}s | P95: {p95_lat:.2f}s")

            self.results.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "concurrency": concurrency,
                    "total_requests": num_requests,
                    "success_rate": len(successes) / num_requests,
                    "avg_latency": avg_lat,
                    "p95_latency": p95_lat,
                    "req_per_sec": throughput,
                    "tokens_per_sec": tokens_per_sec,
                }
            )

    def save_results(self):
        df = pd.DataFrame(self.results)
        df.to_csv(self.output_file, index=False)
        print(f"\n📊 Results saved to {self.output_file}")

        # Print a summary table
        print("\n### 📈 Benchmark Summary Table")
        print(df[["concurrency", "req_per_sec", "tokens_per_sec", "avg_latency", "p95_latency"]].to_string(index=False))


async def main():
    parser = argparse.ArgumentParser(description="Gemma 4 TPU Benchmarking Suite")
    parser.add_argument("--url", type=str, required=True, help="vLLM Endpoint URL (e.g. http://IP:8000)")
    parser.add_argument("--model", type=str, default="google/gemma-4-E2B-it-qat-q4_0-unquantized")
    parser.add_argument("--requests", type=int, default=20, help="Requests per sweep")
    parser.add_argument("--tokens", type=int, default=128, help="Max tokens per request")
    parser.add_argument("--output", type=str, default="benchmark_results.csv")
    args = parser.parse_args()

    prompt = "Explain the importance of Site Reliability Engineering for large scale AI deployments."

    suite = vLLMBenchmarkSuite(args.url, args.model, args.output)

    print(f"🚀 Starting Gemma 4 Performance Sweep on {args.url}")
    print(f"📝 Prompt length: ~{len(prompt.split())} words")

    # Warmup
    print("🔥 Warming up model...")
    async with httpx.AsyncClient() as client:
        await suite.send_request(client, prompt, args.tokens)

    # Sweep through concurrencies
    concurrencies = [1, 2, 4, 8, 16]
    for c in concurrencies:
        await suite.run_batch(c, args.requests, prompt, args.tokens)

    suite.save_results()


if __name__ == "__main__":
    asyncio.run(main())
