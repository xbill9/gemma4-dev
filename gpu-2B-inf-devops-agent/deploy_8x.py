import asyncio, os, sys, boto3
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import server

async def try_deploy(region, market):
    print(f"\n⚡ inf2.8xlarge {market.upper()} in {region}", flush=True)
    os.environ["AWS_DEFAULT_REGION"]=region; os.environ["AWS_REGION"]=region
    server.AWS_REGION=region; boto3.setup_default_session(region_name=region)
    try:
        res=await server.deploy_vllm(service_name="inferentia-2b-devops-agent",
            model_path="google/gemma-4-E2B-it", key_name="alinux",
            instance_type="inf2.8xlarge", market_type=market)
        print(res, flush=True)
        return "Failed to deploy" not in res and "Instance ID" in res, res
    except Exception as e:
        print(f"❌ {region}/{market}: {e}", flush=True)
    return False, None

async def main():
    for ln in open(".aws_creds"):
        if "=" in ln and not ln.strip().startswith("#"):
            k,v=ln.strip().split("=",1); os.environ[k]=v
    regions=["us-east-1","us-east-2","us-west-2","us-west-1","eu-west-1","ap-southeast-1","ap-northeast-1","eu-central-1"]
    for market in ("spot","on-demand"):
        print(f"\n===== {market.upper()} CASCADE (inf2.8xlarge) =====", flush=True)
        for region in regions:
            ok,msg=await try_deploy(region, market)
            if ok:
                print(f"\n🎉 SUCCESS: inf2.8xlarge {market} in {region}!", flush=True)
                open("active_deployment_region.txt","w").write(region)
                return
    print("\n❌ inf2.8xlarge unavailable across all regions (spot+on-demand).", flush=True)
    sys.exit(1)

if __name__=="__main__":
    asyncio.run(main())
