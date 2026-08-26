
# Gemma 4 on GKE with Cloud TPU v6e: step by step with MCP and an Agent CLI

This article provides a step by step deployment guide for Gemma 4 to a Google Kubernetes Engine cluster
backed by a Cloud TPU v6e (Trillium) chip. A suite of Python MCP tools is built to simplify management of
the vLLM hosted Gemma 4 deployment from an agent CLI.

Everything below was run end to end, twice, from an empty project. Every command and every output shown is
real.

#### What is this project trying to Do?

We want one command per lifecycle step, and no remembered command lines. The agent should be able to
provision the hardware, deploy the model, tell us whether it is actually serving, benchmark it and price it.

The deployment is deliberately the smallest useful shape on GKE:

- **One** zonal cluster
- **One** single-host TPU node pool
- **One** `ct6e-standard-1t` node — a single v6e chip
- **One** vLLM Pod serving `google/gemma-4-E2B-it`

Small enough that every line on the bill is visible, and every failure has exactly one place to be.

#### Where do I start?

At this point you should have:

- A GCP project with billing enabled, and **Compute Engine CT6E quota in the region you plan to use** — this
  is the one that trips people up, and it is covered below
- `gcloud` installed and authenticated
- A Hugging Face token with access to Gemma 4
- The rig checked out: `git clone https://github.com/xbill9/gemma4-dev`

Then `cd gemma4-dev/gke-vllm-v6e1-2b`. Everything from here happens in that directory.

#### Quota: check this before anything else

**The two TPU control planes meter against completely different pools, and holding one buys nothing on the
other.** Our project holds 512 v6e chips in `us-east5` under the Cloud TPU API quota — and zero Compute
Engine CT6E quota in that same region. GKE spends the **Compute Engine** pools.

```bash
gcloud compute machine-types list --filter="name=ct6e-standard-1t" --format="value(zone)"
```

That tells us where the machine type is published. Quota is the other half, and there are two ids that do
not behave the same way:

```
Provisioning model   Quota id                                Default when unset
-------------------  --------------------------------------  ------------------
On-demand            TPUS-PER-TPU-FAMILY-per-project-region                   0
Spot and flex-start  PREEMPTIBLE-TPU-V6E-per-project-region                1536
```

Reading only the first one writes off regions that have plenty of flex-start headroom. We use
`europe-west4`, where this project has both quota and actual hardware.

**Quota is a ceiling, not an allocation.** Every zone we probed held 1536 chips of preemptible quota and
four of five had no capacity at all. A spot create is the cheap probe — it fails fast with `reason:
stockout` where flex-start silently queues.

#### Setup the Basic Environment

GKE needs two client tools that a plain VM deployment does not:

```bash
sudo apt-get install -y kubectl google-cloud-cli-gke-gcloud-auth-plugin
```

Then authenticate both ways. The tools need a user credential for gcloud and application default
credentials for the client libraries:

```bash
gcloud auth login
gcloud auth application-default login
```

Store the Hugging Face token in Secret Manager once. The deployment reads it from there, so it never lands
in a manifest or a shell history:

```bash
echo -n "hf_your_token_here" | gcloud secrets create hf-token --data-file=- --project=YOUR_PROJECT
```

Or once the MCP server is registered, from the agent:

```
save_hf_token with token hf_your_token_here
```

Now check everything at once:

```bash
make gke-preflight
```

```
✅ kubectl, gke-gcloud-auth-plugin, envsubst and gcloud credentials all present
```

That target exists because every one of those four fails *late* otherwise — halfway through a deploy, with
an error that does not name the missing piece.

#### tpu.env is the source of truth

The directory name is documentation. `tpu.env` is configuration, and it is committed:

```bash
GOOGLE_CLOUD_PROJECT=aisprint-491218
GOOGLE_CLOUD_ZONE=europe-west4-a
MODEL_NAME=google/gemma-4-E2B-it
MACHINE_TYPE=ct6e-standard-1t
TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN=32768

GKE_CLUSTER_NAME=gke-vllm-v6e1-2b
GKE_NODE_POOL=tpu-v6e-1
GKE_LOCATION=europe-west4-a
TPU_TOPOLOGY=1x1
GKE_NUM_NODES=1
GKE_NODE_PROVISIONING=on-demand
```

A real environment variable always beats the file, so `make gke-status GKE_LOCATION=us-east5-b` works as a
one-off without editing anything.

**Never copy a value out of the directory name into a command.** `v6e1` is for humans; gcloud wants
`ct6e-standard-1t`, and it lives in `MACHINE_TYPE`.

#### Model Management Tool with MCP Stdio Transport

The MCP server is a single Python file built on FastMCP. The simplest transport the SDK supports is stdio,
which connects a locally running process — the agent CLI spawns the server and talks to it over pipes.

```python
mcp = FastMCP(MCP_SERVER_NAME)
```

The server name is not cosmetic. It is the key the client registers under, and that key prefixes every tool:
`mcp__gke-vllm-v6e1-2b__deploy_vllm`. With sibling rigs loaded at the same time, that prefix is the only
thing that tells the GKE rig's tools from the Compute Engine rig's, so we derive it from the directory name
rather than typing it.

#### Running the Python Code

Install and test before wiring anything up:

```bash
make install
make test
```

```
Ran 48 tests in 0.113s

OK
```

The tests are offline — they mock the MCP module and the Google Cloud clients before importing the server —
so this passes without touching a cloud API. Then lint:

```bash
make lint
```

```
mypy .
Success: no issues found in 18 source files
```

And start the server by hand once, just to see it come up:

```bash
make run
```

#### Agent CLI mcp_config.json

Register the server with the agent CLI:

```json
{
  "mcpServers": {
    "gke-vllm-v6e1-2b": {
      "command": "./mcp-run.sh",
      "args": [],
      "env": {}
    }
  }
}
```

`mcp-run.sh` exports only variables that are not already set, so `tpu.env` fills the gaps without ever
overriding the environment we are running in.

#### Validation with the Agent CLI

Ask the agent what it can do. The `get_help` tool renders itself from the live tool list, so it cannot go
stale:

```
get_help
```

Thirty tools, grouped by what they are for:

- **Provisioning** — `create_gke_cluster`, `create_tpu_node_pool`, `provision_gke_tpu`, `find_tpu`
- **Deployment** — `deploy_vllm`, `manage_vllm_deployment`, `get_vllm_deployment_config`
- **Validation** — `get_system_status`, `verify_model_health`, `get_vllm_endpoint`, `query_queued_gemma4`
- **Diagnostics** — `get_vllm_pod_logs`, `get_tpu_node_diagnostics`, `get_cloud_logging_logs`
- **Measurement** — `run_vllm_benchmark`, `estimate_deployment_cost`
- **Teardown** — `destroy_tpu_node_pool`, `destroy_gke_cluster`

#### Getting Started with Gemma 4 on TPU

Before spending anything, ask what it will cost:

```
estimate_deployment_cost with provisioning_model on-demand
```

```
### 💸 Estimated Cost: `$2.97` for `1h` on `1` chip `v6e` (on-demand) in `europe-west4`
- **Rate:** `$2.9700` per chip-h × `1` chips × `1h`
- **SKU:** TpuV6e running in Netherlands
- ⚠️ On-demand has no run bound — this bills until `destroy_tpu_node_pool`.
- Not counted here: the `e2-standard-4` system node and the GKE cluster management fee.
```

That number is read live from the Cloud Billing Catalog at call time. There is no rate table in this
project, because a hardcoded rate is wrong eventually and wrong silently.

#### Create the GKE Cluster

One call, and it is idempotent — run it twice and the second run tells us the cluster is already there:

```
create_gke_cluster
```

Under the hood:

```bash
gcloud container clusters create gke-vllm-v6e1-2b \
    --location=europe-west4-a \
    --release-channel=rapid \
    --num-nodes=1 --machine-type=e2-standard-4 --disk-size=50
```

```
✅ Cluster `gke-vllm-v6e1-2b` created in europe-west4-a (rapid channel).

NAME              LOCATION        MASTER_VERSION      MACHINE_TYPE   NUM_NODES  STATUS
gke-vllm-v6e1-2b  europe-west4-a  1.36.3-gke.1537000  e2-standard-4  1          RUNNING
```

Two decisions in that command are worth knowing:

- **Pin the release channel, never a version.** TPU v6e needs a recent control plane, and any version we pin
  goes stale. Rapid gave us 1.36.3-gke.1537000.
- **The `e2-standard-4` default pool exists only for system workloads.** Keeping kube-dns and
  metrics-server off the TPU node is the whole point — a chip billed by the hour must not be kept alive by
  CoreDNS after the model Pod is gone.

This step takes about nine minutes.

#### Create the TPU Node Pool

This is the step that gets us a chip:

```
create_tpu_node_pool
```

```bash
gcloud container node-pools create tpu-v6e-1 \
    --cluster=gke-vllm-v6e1-2b --location=europe-west4-a \
    --node-locations=europe-west4-a \
    --machine-type=ct6e-standard-1t \
    --num-nodes=1 --disk-size=200
```

```
✅ Node pool `tpu-v6e-1` created in `gke-vllm-v6e1-2b` (europe-west4-a): 1x ct6e-standard-1t, on-demand.
```

**Do not add `--tpu-topology=1x1` to that command.** Our first attempt did, and the API refused it:

```
ERROR: (gcloud.container.node-pools.create) ResponseError: code=400,
message=TPU topology can't be specified with single-host TPU slice pool;
please remove the tpu_topology from the node pool creation request
```

A one-chip machine type at one node **is** the slice — there is no topology to describe. The flag belongs to
multi-host slices. What makes this a trap rather than a typo is that GKE then labels the node
`cloud.google.com/gke-tpu-topology=1x1` **anyway**, so the value is real as a Pod selector and rejected as a
create flag. We keep them in two separate config keys for exactly that reason.

Both steps in one call, if we prefer:

```
provision_gke_tpu
```

#### What Actually Got Created

Here is the thing nobody tells you: **the cluster created its own TPU, and what it created is a Compute
Engine VM.**

```bash
gcloud compute instances list
```

```
NAME                                             MACHINE_TYPE      STATUS
gke-gke-vllm-v6e1-2b-default-pool-8e8b988c-7d71  e2-standard-4     RUNNING
gke-tpu-bcdb7fb0-bw0t                            ct6e-standard-1t  RUNNING
```

```bash
gcloud compute instance-groups managed list
```

```
NAME                                          SIZE
gke-gke-vllm-v6e1-2b-tpu-v6e-1-cfc04f31-grp   1
```

The node pool is a **managed instance group**, and the node inside it carries the same machine type we would
pass to `gcloud compute instances create` on a VM deployment. Same silicon, same attachment. What changed is
who calls create and who owns the lifecycle.

Three consequences, each of which costs an afternoon if we learn it the hard way:

- **`gcloud compute instances delete` on that node is not teardown.** It succeeds, and the instance group
  rebuilds the node minutes later under a new name, still billing. Delete the **pool**.
- **The node is cattle.** It is replaced on upgrade and repair, so anything we write to its disk is gone.
- **We do not choose the node's name.** Across two runs it came back as `gke-tpu-cfc04f31-8h14` and then
  `gke-tpu-bcdb7fb0-bw0t`.

Check what the node is advertising:

```bash
kubectl get node gke-tpu-bcdb7fb0-bw0t -o jsonpath='{.status.allocatable}'
```

```
{"cpu":"43820m","google.com/tpu":"1","memory":"170630440Ki","pods":"110"}
```

**A Ready node can advertise `google.com/tpu: 0` for a window** before the device plugin registers. A Pod
scheduled in that window fails with `Insufficient google.com/tpu`, which reads exactly like a quota problem
and is not one. If we see that, we wait thirty seconds.

#### Deploy The Model

```
deploy_vllm
```

```
🚀 Deployed `vllm-gemma4` to cluster `gke-vllm-v6e1-2b` (europe-west4-a).
- model: `google/gemma-4-E2B-it`, max-model-len 32768, TP 1
- image: `vllm/vllm-tpu:nightly`, service type `LoadBalancer`

deployment.apps/vllm-gemma4 created
service/vllm-gemma4 created

👉 The image pull, weight load and XLA precompile take about ten minutes.
```

That tool does three things with no VM equivalent: fetches cluster credentials, copies the Hugging Face
token from Secret Manager into a Kubernetes Secret, and applies the manifest.

The Pod spec is where the chip actually gets claimed:

```yaml
      nodeSelector:
        cloud.google.com/gke-tpu-accelerator: tpu-v6e-slice
        cloud.google.com/gke-tpu-topology: 1x1
      tolerations:
        - key: google.com/tpu
          operator: Exists
          effect: NoSchedule
      containers:
        - name: vllm
          image: vllm/vllm-tpu:nightly
          command: ["vllm", "serve", "google/gemma-4-E2B-it"]
          args:
            - --max-model-len=32768
            - --tensor-parallel-size=1
            - --max_num_batched_tokens=4096
            - --enable-auto-tool-choice
            - --tool-call-parser=gemma4
            - --reasoning-parser=gemma4
          resources:
            limits:
              google.com/tpu: "1"
          startupProbe:
            httpGet: { path: /health, port: 8000 }
            periodSeconds: 15
            failureThreshold: 120
```

Four things in there are load-bearing:

- **Both node selectors.** Drop one and the Pod schedules onto the `e2-standard-4` system node and fails
  there, which reads as a vLLM problem rather than a placement one.
- **The `google.com/tpu: 1` limit.** Without it the device plugin never attaches the chip.
- **The toleration.** GKE taints TPU nodes `google.com/tpu=present:NoSchedule` automatically.
- **The startup probe budget.** 15 s × 120 = 30 minutes. The load took ten; a budget shorter than that gets
  the container killed and restarted forever, one ten-minute load at a time.

No `privileged: true` is needed. On GKE the device plugin handles device access.

We can print the exact manifest without applying it:

```
get_vllm_deployment_config
```

#### Checking System Status

```
get_system_status
```

```
### 🌀 System Status (europe-west4-a)
- **Cluster:** ✅ `gke-vllm-v6e1-2b`
- **TPU node pool:** ✅ tpu-v6e-1
- **vLLM Health:** 🟢 Online at http://34.91.103.13:8000 (serving `google/gemma-4-E2B-it`)
**🖥️ TPU nodes:**
- `gke-tpu-bcdb7fb0-bw0t` — Ready, google.com/tpu: 1, pool `tpu-v6e-1`
**📦 vLLM pod:** `vllm-gemma4-858cfc589f-swh95` — Running, ready, on `gke-tpu-bcdb7fb0-bw0t`
**👉 Next Step:** Use `query_queued_gemma4` to interact with the model.
```

Five separate claims there, and each can be healthy while the next is not. **"Running" is three steps weaker
than "serving" on this path.** A GKE node is Ready the moment the kubelet registers — before the Pod is
scheduled, before the image is pulled, before the weights load, before XLA compiles.

While it loads, watch the actual work:

```
get_vllm_pod_logs with tail 20
```

#### Cross Check The Deployed Model

Never trust one layer. Go under the tools and ask Kubernetes directly:

```bash
kubectl get pods -l app=vllm-gemma4 -o wide
kubectl get svc vllm-gemma4
```

```
NAME                           READY   STATUS    RESTARTS   AGE
vllm-gemma4-858cfc589f-swh95   1/1     Running   0          47m
```

Then bypass Kubernetes too, and ask the model server:

```bash
curl -s http://34.91.103.13:8000/v1/models | python3 -m json.tool
```

**The endpoint is a Service, not a machine.** This is the most portable mistake from the VM guides: a GKE
node *does* appear in `gcloud compute instances list`, so reading its external IP succeeds and returns the
wrong address. Across a full teardown and rebuild, our Service came back on the **same** IP while the node
behind it changed name entirely.

#### Review the Model

```
verify_model_health
```

```
✅ Model health check PASSED.
Response: 'Hello! Yes, I am working. I am...'
Latency: 0.77 seconds.
```

```
query_queued_gemma4 with prompt "In one short sentence, what is a TPU node pool?"
```

```
A TPU node pool is a group of dedicated TPU hardware resources for running machine learning workloads.
```

Or from the shell, without the agent:

```bash
make query PROMPT="What is Site Reliability Engineering?"
```

**Use the chat endpoint, not raw completions.** `/v1/completions` returns an empty completion on `-it`
models, so an empty result there is expected rather than a broken deploy.

#### How long did all that take?

Measured twice from an empty project:

```
Step                        Tool                           Time
--------------------------  ----------------------  -----------
Cluster + TPU node pool     provision_gke_tpu          8 m 52 s
Apply Deployment + Service  deploy_vllm                     6 s
Pod -> model answering      poll get_system_status    11 m 09 s
Nothing -> first token                              ~20 minutes
Full teardown               destroy_gke_cluster        6 m 24 s
```

That eleven-minute load is billed chip time **every time the Pod is rescheduled**, which is a good argument
against letting a scheduler move a model Pod around casually.

#### Benchmark the Model

```
run_vllm_benchmark with max_concurrency 8 and save_result true
```

The benchmark runs `vllm bench serve` **inside the serving Pod** via `kubectl exec`, against localhost.
There is no SSH here and no second container; the load is TPU-bound and the client is not, so sharing the
Pod's CPU does not distort the numbers.

**Concurrency sweep**, 1024 input / 128 output tokens:

```
Concurrency  Output tok/s  TTFT p99  TPOT mean  Per-stream tok/s
-----------  ------------  --------  ---------  ----------------
          1  199              17 ms     4.9 ms               203
          2  380              23 ms     5.2 ms               194
          4  693              29 ms     5.6 ms               177
          8  1,180            45 ms     6.6 ms               152
         16  #1 1,744        118 ms     8.8 ms               115
         32  1,675           216 ms    18.5 ms                54
         64  #1 2,134        392 ms    28.5 ms                35
```

**Throughput is not monotonic, and the dip is real.** Concurrency 32 comes in *below* concurrency 16. That
looked like noise, so we ran each point three more times:

```
Concurrency     Run 1     Run 2     Run 3  Spread
-----------  --------  --------  --------  ------
         16  1,777.00  1,780.35  1,778.87    0.2%
         32  1,694.35  1,696.15  1,694.24    0.1%
         64  2,350.99  2,353.04  2,351.38    0.1%
```

Run-to-run spread is two tenths of a percent. The shape reproduces — concurrency 32 really is about 4%
slower than 16 here. The plausible cause is TPU static-shape padding: vLLM compiles a set of batch shapes
and pads to the nearest, so a batch straddling a bucket boundary spends compute on padding. **That is a
hypothesis and we have not verified it** against the compiled shape list, and it is recorded as unverified
in the report. The operational lesson stands on its own: **measure your own concurrency ladder**, because
the arithmetic answer is wrong here.

**Context sweep**, concurrency 8:

```
Input tokens  Output tok/s  TTFT p99  Per-stream tok/s
------------  ------------  --------  ----------------
         512         1,185     35 ms               152
       1,024         1,180     45 ms               152
       2,048         1,083    133 ms               148
       4,096           986    216 ms               140
       8,192           795    399 ms               122
      16,384           487    878 ms                80
```

A 32× longer prompt costs **2.4× more per output token and 25× the time to first token**. Prefill dominates
as context grows. Note that memory is not the limit — v6e's 32 GB of HBM leaves roughly 19.8 GiB for KV
after this model's weights, so 16K at concurrency 8 sits well inside the budget and still nearly halves
throughput. The ceiling is time.

Each point comes back as a schema-conformant `throughput.sweep[]` entry, so the run files itself as a
validated report rather than a screenshot.

#### Cost Breakdowns

Every price here was read live from the Cloud Billing Catalog for `europe-west4`.

**The same chip has three prices, and they are three products — not three discounts:**

```
                  $/chip-hour  What we are buying
----------------  -----------  -----------------------------------------------------  --
Flex-start (DWS)      $1.3500  Queued grant, then uninterrupted for a bounded window  #1
Spot                  $1.7820  Same hardware, reclaimable at any moment               #2
On-demand             $2.9700  Starts instantly, runs unbounded, nothing preempts it  #3
```

We tested the cheap one rather than quoting it. A flex-start node pool came up with **zero nodes** — on GKE
flex-start is an autoscaling shape, not a fixed pool, so an idle flex-start pool costs nothing at all. We
scaled the Deployment to two replicas, the second Pod went `Pending`, and **4 minutes 18 seconds later DWS
granted a chip**: a node joined, the Pod landed, and ten minutes later the replica was serving.

The whole 55% saving costs about four minutes of queue.

**The bill is not only the chip.** Two line items exist here with no VM equivalent:

```
Line item                               Catalog SKU                                    $/hour
--------------------------------------  --------------------------------------------  -------
System node e2-standard-4               E2 Instance Core x4 + E2 Instance Ram x16 GB  $0.1475
Cluster management fee                  Zonal Kubernetes Clusters                     $0.1000
Fixed overhead, any provisioning model                                                $0.2475
```

Google's free tier credits cover roughly one zonal cluster's fee. The system node is not covered.

**All-in, one chip serving:**

```
               Chip  Overhead  Total/hour  Total/month (730 h)
----------  -------  --------  ----------  -------------------  --
Flex-start  $1.3500   $0.2475     $1.5975               $1,166  #1
Spot        $1.7820   $0.2475     $2.0295               $1,482  #2
On-demand   $2.9700   $0.2475     $3.2175               $2,349  #3
```

The overhead is 7.7% of an on-demand bill but **15.5% of a flex-start one**. The cheaper the capacity, the
more the fixed cost of running Kubernetes matters — that, and not the cluster fee in isolation, is the real
cost argument against a cluster for a small deployment.

**Cost per million output tokens**, all-in:

```
Concurrency  Output tok/s  On-demand   Spot  Flex-start
-----------  ------------  ---------  -----  ----------  --
          1           199      $4.48  $2.83       $2.23  #3
          8         1,180      $0.76  $0.48       $0.38
         16         1,744      $0.51  $0.32       $0.25  #2
         64         2,134      $0.42  $0.26       $0.21  #1
```

**Concurrency is worth more than procurement.** One stream to sixty-four divides unit cost by 10.7×;
on-demand to flex-start divides it by 2.0×. Both together: **21×**, on identical silicon running the
identical model. The trade is visible in the sweep — at concurrency 64 each caller sees 35 tokens per second
instead of 203.

**The line nobody budgets:** with the TPU node pool deleted, the cluster and its system node still bill
$0.2475/hour. That is **$181 a month for an empty cluster.**

#### Tearing It Down

**Nothing here stops billing on its own.** A Compute Engine instance can carry `--max-run-duration` with
`--instance-termination-action=DELETE` and delete itself. A node pool has **no run bound at all** — not even
under flex-start, where the setting caps how long capacity is *granted*, not how long it is *billed*.

Release the chip and keep the cluster:

```
destroy_tpu_node_pool
```

```
🗑️ Node pool `tpu-v6e-1` deleted from `gke-vllm-v6e1-2b` (europe-west4-a). The chip is released.
```

Or take it all:

```
destroy_gke_cluster
```

From the shell, the same two:

```bash
make destroy          # node pool only
make destroy-cluster  # everything
```

#### Five things that broke, and what they had in common

Every one was a correct habit imported from the Compute Engine version of this deployment.

1. **`--tpu-topology=1x1`** — refused for a single-host slice, while GKE sets the matching label anyway.
2. **`gcloud secrets versions access` without `--project`** — silently resolved to the workstation's default
   project, an expired lab, and failed with a permission error naming a project this rig never mentions.
3. **A dollar sign in a template comment** — the manifest renders through `envsubst` from the shell path and
   `string.Template` from the MCP tool. The first tolerates it, the second raises. So the shell path worked
   and the tool path did not, which is the worst way to find a bug.
4. **Vocabulary drift** — the shell scripts said `ondemand` and the tools said `on-demand`. A test caught it,
   not a run.
5. **The cost tool repeated the VM's story** — "flex-start self-terminates at `--max-run-duration`, capping
   the bill." False on a node pool: a confident wrong statement about money, in the one tool written to
   avoid exactly that.

The defence that worked was not more careful reading. It was a test that greps the **whole** server module
for the old control plane's commands rather than checking one function — because an earlier version of this
project asserted it had migrated and verified that claim on a single call path, while four tools quietly
kept using `tpus tpu-vm ssh`.

#### Summary

The strategy for using MCP for Gemma 4 TPU deployment on GKE was validated with an incremental step by step
approach. A minimal stdio transport MCP server was started from Python source, registered with an agent CLI
in the same local environment, and then used to provision a cluster, attach a TPU node pool, deploy vLLM,
validate the endpoint, benchmark it and price it — each step checked against the live system rather than
against the previous step's assumptions.

What we can state, having run it twice end to end:

- A single-chip TPU deployment on GKE takes **about twenty minutes** from nothing to a served token.
- **The cluster provisions its own TPU**, as a Compute Engine VM inside a managed instance group it owns and
  we do not. Deleting that VM is not teardown; deleting the pool is.
- **Flex-start is the default worth reaching for** on bounded work: 55% off the chip rate for a four-minute
  queue, with a pool that costs nothing while idle.
- **Concurrency beats procurement** as a cost lever, five to one — provided the latency budget can absorb it.

For one small model with one replica, a plain VM is still the simpler answer and the cost table above says
so. Kubernetes earns its overhead when there are replicas to schedule, versions to roll, or an application
already living in the cluster. When it does, the twenty minutes above is the whole of it.
