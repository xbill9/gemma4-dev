import asyncio,sys,time,httpx; sys.path.insert(0,'/home/xbill/gemma4-dev/gpu-jax-g5g-2b')
import server
IID='i-021f15b2b45e13793'; URL='http://3.238.21.83:8000'
PROMPT='Name three primary colours.'; MAXTOK=48

async def wait_ready(timeout=420):
    t0=time.time()
    async with httpx.AsyncClient(timeout=10) as c:
        while time.time()-t0 < timeout:
            try:
                if (await c.get(URL+"/health")).status_code==200: return time.time()-t0
            except Exception: pass
            await asyncio.sleep(5)
    raise TimeoutError("never became ready")

async def timed_request():
    async with httpx.AsyncClient(timeout=240) as c:
        t=time.time()
        r=await c.post(URL+"/v1/chat/completions", json={
            "model":"google/gemma-4-E2B-it",
            "messages":[{"role":"user","content":PROMPT}],"max_tokens":MAXTOK})
        dt=time.time()-t; u=r.json().get("usage",{})
        return dt, u.get("completion_tokens"), u.get("cold_shape", u.get("cold"))

async def phase(label, restore):
    cmd = "systemctl stop jax-g5g; rm -rf /opt/jax-cache; mkdir -p /opt/jax-cache; "
    if restore:
        cmd += ("aws s3 sync s3://vllm-models-bucket/jax-cache/gpu-jax-g5g-2b/ /opt/jax-cache "
                "--only-show-errors || true; ")
    cmd += "find /opt/jax-cache -type f | wc -l; systemctl start jax-g5g"
    n = (await server._ssm(IID, cmd)).strip().splitlines()[-1]
    load = await wait_ready()
    dt, toks, cold = await timed_request()
    print(f"{label:32s} cache_before_start={n:>4s}  load={load:5.1f}s  "
          f"first_request={dt:6.2f}s  tokens={toks}  cold={cold}", flush=True)
    return dt, load

async def main():
    a, la = await phase("A control: empty cache", False)
    b, lb = await phase("B treatment: restored from S3", True)
    print(f"\nfirst-request: {a:.2f}s -> {b:.2f}s  ({a/b:.1f}x)", flush=True)
    print(f"time-to-ready: {la:.1f}s -> {lb:.1f}s", flush=True)
asyncio.run(main())
