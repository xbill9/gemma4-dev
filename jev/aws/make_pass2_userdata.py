"""Build the cloud-init for run 2026-09-23-l4-latency (see PREREGISTRATION.md addendum).

    python3 aws/make_pass2_userdata.py > /tmp/pass2-userdata.sh

The instance runs every pass itself against localhost:8000, so no AWS call is
needed after launch: when the passes finish it serves the results on :8000
(open to one IP by the security group) and terminates itself. Code comes in as
a base64 tarball; data is rebuilt with build_eval_set.py and checked against
the committed hashes; the proxy is fetched from vLLM at the PR merge commit and
checked against the vendored copy's hash.
"""

import base64
import hashlib
import io
import os
import tarfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ["run_eval.py", "tasks.py", "build_eval_set.py"]
IMAGE = "vllm/vllm-openai:nightly-e9757321527ca1ecd514c07c1418dd2c53da3d19"
PROXY_URL = (
    "https://raw.githubusercontent.com/vllm-project/vllm/"
    "1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8/examples/features/structured_diffusion/structured_server.py"
)


def sha(path):
    return hashlib.sha256(open(os.path.join(HERE, path), "rb").read()).hexdigest()


buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    for f in FILES:
        tar.add(os.path.join(HERE, f), arcname=f)
bundle = base64.b64encode(buf.getvalue()).decode()
data_sums = "\n".join(
    f"{sha('data/' + t + '.jsonl')}  data/{t}.jsonl" for t in ("ag_news", "emotion", "irony", "sst2")
)
proxy_sum = sha("vendor/structured_server.py")

print(f"""#!/usr/bin/env bash
set -uxo pipefail
shutdown -h +150 || true      # hard stop; InstanceInitiatedShutdownBehavior=terminate
systemctl enable --now docker
if [ "$(awk '/MemTotal/{{print int($2/1048576)}}' /proc/meminfo)" -lt 30 ] && ! swapon --show --noheadings | grep -q /swapfile; then
  fallocate -l 16G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
fi
W=/opt/jev/work; OUT=$W/out; mkdir -p $W/vendor $OUT /opt/hf
log() {{ echo "[$(date -Is)] $*" | tee -a $OUT/pass2.log; }}
echo '{bundle}' | base64 -d | tar xzf - -C $W
curl -sf -o $W/vendor/structured_server.py '{PROXY_URL}'
echo '{proxy_sum}  vendor/structured_server.py' > $W/proxy.sha256
IMAGE={IMAGE}
docker pull $IMAGE
log image pulled

serve() {{
  local model="$1"; shift
  docker rm -f vllm >/dev/null 2>&1 || true
  set +x
  HF_TOKEN=$(aws secretsmanager get-secret-value --region us-east-1 --secret-id vllm/hf-token --query SecretString --output text 2>/dev/null || true)
  docker run -d --name vllm --ipc=host --gpus all -e HF_TOKEN="$HF_TOKEN" \\
    -v /opt/hf:/root/.cache/huggingface -v $W:/work -p 127.0.0.1:8000:8000 $IMAGE \\
    --model "$model" --served-model-name "$model" --host 0.0.0.0 --port 8000 \\
    --attention-backend TRITON_ATTN --gpu-memory-utilization 0.92 \\
    --max-model-len 2048 --max-num-seqs 16 --max-logprobs 32 \\
    --enforce-eager --enable-prefix-caching "$@"
  set -x
  for i in $(seq 1 120); do curl -sf localhost:8000/v1/models | grep -q '"id"' && {{ log "ready $model after $((i*10))s"; return 0; }}; sleep 10; done
  log "TIMEOUT $model"; docker logs --tail 60 vllm > $OUT/fail-$(echo $model | tr / _).log 2>&1; return 1
}}
client() {{ docker exec -w /work vllm python3 "$@" >> $OUT/pass2.log 2>&1; }}

serve cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit && {{
  client build_eval_set.py --per-task 300
  (cd $W && sha256sum -c proxy.sha256 && printf '%s\\n' '{data_sums}' | sha256sum -c) > $OUT/hashes.txt 2>&1
  log "hash check: $(grep -c ': OK' $OUT/hashes.txt) of 5 OK"
  client run_eval.py --arm autoregressive --upstream http://localhost:8000 --model cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit --run 2026-09-23-l4-latency --limit 100 --concurrency 1
}}
serve cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4 --diffusion-config '{{"canvas_length": 64}}' && \\
  client run_eval.py --arm diffusion --upstream http://localhost:8000 --model cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4 --run 2026-09-23-l4-latency --limit 100 --concurrency 1 --reads 4 --keep-top 5
for m in E4B E2B; do
  serve google/gemma-4-$m-it --dtype bfloat16 && {{
    client run_eval.py --arm autoregressive --upstream http://localhost:8000 --model google/gemma-4-$m-it --run 2026-09-23-l4-$(echo $m | tr A-Z a-z) --concurrency 8
    client run_eval.py --arm autoregressive --upstream http://localhost:8000 --model google/gemma-4-$m-it --run 2026-09-23-l4-$(echo $m | tr A-Z a-z) --concurrency 8 --variant reversed
  }}
done
{{ nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  docker inspect --format 'image: {{{{.Config.Image}}}} digest: {{{{.Image}}}}' vllm
  docker exec vllm python3 -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__)"
  TOKEN=$(curl -s -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')
  echo "instance-type: $(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-type)"
  echo "az: $(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/placement/availability-zone)"
}} > $OUT/host.txt 2>&1
docker rm -f vllm || true
cp -r $W/results $OUT/ 2>/dev/null || true
cp -r $W/data $OUT/ 2>/dev/null || true
log DONE
touch $OUT/DONE
(cd /opt/jev/work && tar czf out.tgz out)
cd /opt/jev/work && timeout 3600 python3 -m http.server 8000 --bind 0.0.0.0
shutdown -h now
""")
