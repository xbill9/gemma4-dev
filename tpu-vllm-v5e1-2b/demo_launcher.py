# Grand Demo: TPU vLLM DevOps Agent (Gemma 4)

import asyncio

from server import (
    analyze_cloud_logging,
    get_vllm_deployment_config,
    get_vllm_endpoint,
)


async def devops_demo():
    print("🚀 TPU Sprint Demo: Self-Hosted vLLM DevOps Agent (Gemma 4)")
    print("=" * 60)

    # Step 1: Discovery
    print("\n[Step 1] Discovering TPU vLLM Endpoint...")
    try:
        endpoint = await get_vllm_endpoint()
        print(f"  FOUND: {endpoint}")
    except Exception as e:
        print(f"  NOTICE: {e} (Discovery failed, using simulation mode)")

    # Step 2: Log Analysis
    print("\n[Step 2] Analyzing Cloud Logging errors (severity=ERROR)...")
    # In a real demo, this queries the self-hosted Gemma 4 on TPU
    analysis = await analyze_cloud_logging(minutes=60)
    print(f"  ANALYSIS: {analysis[:300]}...")

    # Step 3: Deployment Config & TPU instructions
    print("\n[Step 3] Generating TPU v5e (v5litepod) Deployment Config...")
    config = await get_vllm_deployment_config(
        service_name="tpu-2B-v5e1-devops-agent",
        model_name="google/gemma-4-E2B-it",
    )
    print(config)

    print("\n" + "=" * 60)
    print("✅ DevOps Agent Demo Complete: Gemma 4 on TPU v5e ready!")


if __name__ == "__main__":
    asyncio.run(devops_demo())
