# Gemma 4 on GKE with a TPU v6e-1: a step by step deployment with MCP

This article provides a step by step deployment guide for Gemma 4 to a Google Kubernetes Engine cluster
backed by a single Cloud TPU v6e (Trillium) chip. A suite of Python MCP tools is built to provision the
cluster, deploy the model server, validate it, benchmark it and price it — driven from an agent CLI over
stdio transport.

The deployment is small on purpose: **one cluster, one node, one chip, one Pod.** That is the smallest
useful shape on GKE, and it makes every cost line visible rather than buried in a fleet.

Two questions get answered along the way that are specific to this path and not to the GPU-on-a-VM guides
in this series:

- **Does the cluster create its own TPU, or does a VM have to be provisioned first?** It creates its own —
  and what it creates is still a Compute Engine VM. That distinction turns out to matter for teardown.
- **Why is on-demand capacity $2.97 per chip-hour when flex-start is $1.35?** The two-and-a-bit multiple is
  not a discount, it is a different product. The cost section breaks it down with live catalog prices.

#### 🎯 What is this project trying to Do?

The goal is a reproducible, agent-driven deployment of `google/gemma-4-E2B-it` on TPU hardware, where every
step is a tool call rather than a remembered command line:

- **Provision** a GKE cluster and a single-host TPU v6e node pool
- **Deploy** vLLM as a Kubernetes Deployment with the chip attached to the Pod
- **Validate** health, endpoint and a real completion
- **Benchmark** with a concurrency sweep, emitting a schema-conformant report
- **Price** the result from the live Cloud Billing Catalog, not from a rate table

This rig is one of three that serve the identical model on the identical chip and differ **only** in which
control plane provisions the hardware: the Cloud TPU API, Compute Engine, and — here — GKE. Everything about
the serving configuration is held constant so the provisioning path is the only variable.

#### 🧱 Why GKE at all for one small model?

Worth answering honestly before spending an hour on it, because for a single chip the answer is often "you
don't need it."

A single TPU chip serving a 2B-class model runs perfectly well as one VM with `docker run`. GKE earns its
keep when there is more than one replica behind an endpoint, when the deployment must survive a bad boot
without a human, when rolling a new model version with no downtime matters, or when the rest of the
application already lives in Kubernetes.

Two things push the other way, and both are sharper for **small** models than for large ones:

- **A TPU chip is allocated whole.** The node advertises `google.com/tpu: 1` and one Pod takes it. There is
  no MIG/MPS-style partitioning, so the classic small-model argument — bin-pack a dozen little services onto
  one accelerator — does not apply here at all.
- **Fixed overhead is proportionally worse the cheaper the accelerator.** The cluster fee plus the system
  node is about $0.25/hour whatever else happens. Against a v6e that is noise; against a cheap GPU it is a
  third of the bill.

The counterintuitive part: **small models are the easy case for Kubernetes, large ones are the hard case.**
A model that fits in one chip scales out as replicas, which is exactly what a Deployment does well. A model
that needs eight chips spans a multi-host slice, and that needs gang scheduling and a slice that fails as a
unit.

#### 🛠️ Setup the Basic Environment

The rig is a single-directory project. Its `tpu.env` is the source of truth for every value below, and it is
committed — the directory name is documentation, the env file is configuration.

```
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

GKE needs two client-side tools that a Compute Engine deployment does not, and both fail late and
confusingly if missing:

```bash
sudo apt-get install -y kubectl google-cloud-cli-gke-gcloud-auth-plugin
gcloud auth login
gcloud auth application-default login
make gke-preflight
```

`make gke-preflight` checks `kubectl`, the auth plugin, `envsubst` and a live gcloud token, and prints the
install command rather than failing halfway through a deploy.

**Zone choice is a quota decision, not a latency one.** The two control planes meter against completely
different pools: this project holds 512 v6e chips in `us-east5` under the Cloud TPU API quota, and **zero**
Compute Engine CT6E quota in that region. GKE spends the Compute Engine pools. `europe-west4` is where this
project's Compute Engine quota and actual capacity overlap, so that is where the cluster goes.

#### 🐍 Model Management Tool with MCP Stdio Transport

The MCP server is a single Python file built on FastMCP. The simplest transport the SDK supports is stdio,
which connects a locally running process — the agent CLI spawns the server and speaks to it over pipes.

```python
mcp = FastMCP(MCP_SERVER_NAME)
```

The server name matters more than it looks. It is the key the client registers the server under, and that
key prefixes every tool — `mcp__gke-vllm-v6e1-2b__find_tpu`. With sibling rigs loaded at the same time, the
prefix is the only thing distinguishing a call to the GKE rig from a call to the Compute Engine one, so the
name is derived from the rig directory rather than typed in.

Thirty tools are exposed. The ones that carry the deployment:

- **Provisioning** — `create_gke_cluster`, `create_tpu_node_pool`, `provision_gke_tpu`, `find_tpu`
- **Deployment** — `deploy_vllm`, `manage_vllm_deployment`, `get_vllm_deployment_config`
- **Validation** — `get_system_status`, `verify_model_health`, `get_vllm_endpoint`, `query_queued_gemma4`
- **Diagnostics** — `get_vllm_pod_logs`, `get_tpu_node_diagnostics`, `get_cloud_logging_logs`
- **Measurement** — `run_vllm_benchmark`, `estimate_deployment_cost`
- **Teardown** — `destroy_tpu_node_pool`, `destroy_gke_cluster`

Every subprocess call goes through one helper using `asyncio.create_subprocess_exec` with an argument list —
never `shell=True` — and every tool returns markdown with an emoji status prefix, because the output is read
by a human through an agent transcript.

#### ⚙️ Agent CLI mcp_config.json

Registering the server is four lines:

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

`mcp-run.sh` exports only variables that are not already set, so a real environment variable always beats
`tpu.env`. That is what makes `make gke-status GKE_LOCATION=us-east5-b` work as a one-off without editing
committed config.

#### 🚀 Model Lifecycle Management via MCP

The lifecycle is four tool calls, and the wall-clock cost of each is worth knowing before starting:

| Step | Tool | Time |
| :--- | :--- | ---: |
| Cluster + TPU node pool | `provision_gke_tpu` | 8 m 52 s |
| Apply Deployment + Service | `deploy_vllm` | 6 s |
| Pod → model answering | (poll `get_system_status`) | 11 m 09 s |
| Teardown | `destroy_gke_cluster` | 6 m 24 s |

**Cold start is about 20 minutes from nothing to a served token**, 26 minutes if a teardown comes first.
Measured twice on separate runs, from a deliberate cold start with no cluster in the project.

#### 🏗️ Provision the Cluster and the TPU Node Pool

`provision_gke_tpu` wraps two calls. The first creates a zonal cluster with a small default pool that exists
only to run system workloads:

```bash
gcloud container clusters create gke-vllm-v6e1-2b \
    --location=europe-west4-a \
    --release-channel=rapid \
    --num-nodes=1 --machine-type=e2-standard-4 --disk-size=50
```

Keeping kube-dns and metrics-server off the TPU node is the point of that pool. A v6e node is billed by the
chip and must not be kept alive by CoreDNS after the model Pod is gone.

**Pin the release channel, never a version string.** TPU v6e needs a recent control plane, and every pinned
version goes stale; rapid resolved to 1.36.3-gke.1537000.

The second call creates the node pool that carries the chip:

```bash
gcloud container node-pools create tpu-v6e-1 \
    --cluster=gke-vllm-v6e1-2b --location=europe-west4-a \
    --node-locations=europe-west4-a \
    --machine-type=ct6e-standard-1t \
    --num-nodes=1 --disk-size=200
```

**There is no `--tpu-topology` flag in that command, and adding one is an error.** The first attempt at this
deployment included `--tpu-topology=1x1` and was refused by the API:

```
ERROR: (gcloud.container.node-pools.create) ResponseError: code=400,
message=TPU topology can't be specified with single-host TPU slice pool;
please remove the tpu_topology from the node pool creation request
```

`ct6e-standard-1t` at one node **is** the slice, so there is no topology to describe — the flag belongs to
multi-host slices, where it says how several nodes are wired into one. The trap is that GKE then labels the
node `cloud.google.com/gke-tpu-topology=1x1` anyway. The value is real as a Pod selector and rejected as a
create flag, so the two live in separate config keys in `tpu.env`.

#### 🖥️ What actually got created

This is the answer to "does the cluster make its own TPU?" — it does, and the thing it makes is a Compute
Engine VM:

```
$ gcloud compute instances list
NAME                                             MACHINE_TYPE      STATUS
gke-gke-vllm-v6e1-2b-default-pool-8e8b988c-7d71  e2-standard-4     RUNNING
gke-tpu-bcdb7fb0-bw0t                            ct6e-standard-1t  RUNNING

$ gcloud compute instance-groups managed list
NAME                                          SIZE
gke-gke-vllm-v6e1-2b-tpu-v6e-1-cfc04f31-grp   1
```

The node pool is implemented as a **managed instance group**, and the node inside it carries the same
`ct6e-standard-1t` machine type a Compute Engine deployment would pass to `instances create`. Same silicon,
same attachment mechanism. What changes is who calls create and who owns the lifecycle.

**Consequences that cost time if learned the hard way:**

- **`gcloud compute instances delete` on that node is not teardown.** It succeeds, and the MIG rebuilds the
  node within minutes under a new name, still billing. The pool is the unit of teardown.
- **The node is cattle.** It is replaced on upgrade and repair; anything written to its disk goes with it.
  State belongs in the Pod spec.
- **The node's name is not yours to choose.** Across two runs of this deployment the node came back as
  `gke-tpu-cfc04f31-8h14` and then `gke-tpu-bcdb7fb0-bw0t`.

The node registers with three labels and a taint that together make the scheduling work:

```
cloud.google.com/gke-tpu-accelerator=tpu-v6e-slice
cloud.google.com/gke-tpu-topology=1x1
cloud.google.com/gke-nodepool=tpu-v6e-1
taint: google.com/tpu=present:NoSchedule
allocatable: google.com/tpu: 1
```

**A Ready node can advertise zero chips.** Observed on a freshly created pool: `google.com/tpu: 0` on a node
already reporting Ready, because the device plugin had not finished registering. A Pod scheduled in that
window fails with `Insufficient google.com/tpu`, which reads like a quota problem and is not one.

#### 📦 Deploy The Model

`deploy_vllm` does three things that have no equivalent in a VM deployment: fetch cluster credentials,
materialise the Hugging Face token as a Kubernetes Secret, and apply the manifest.

The Pod spec is where the chip is claimed:

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
```

**All three of the selector, the limit and the toleration are load-bearing.** Drop a selector and the Pod
schedules onto the `e2-standard-4` system node and fails there, which reads as a vLLM problem rather than a
placement one. Drop the limit and the device plugin never attaches the chip. No `privileged: true` is
needed — on GKE the device plugin handles device access.

**The startup probe has to cover the whole load.** Image pull, weight load and XLA precompile took ten
minutes; a probe budget shorter than that gets the container killed and restarted forever, one load at a
time:

```yaml
          startupProbe:
            httpGet: { path: /health, port: 8000 }
            periodSeconds: 15
            failureThreshold: 120
```

**The token never goes through a command line.** `kubectl create secret --from-literal=token=...` puts the
Hugging Face token in the process table for every user on the machine. The tool writes a base64 Secret
manifest to a 0600 temp file and applies that instead.

#### ✅ Validate the Deployment

`get_system_status` is the dashboard, and it reports each layer separately because each can be healthy while
the next is not:

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

**"Running" is three steps weaker than "serving" on this path.** A Queued Resource reached ACTIVE with a
node up. A Compute Engine instance was RUNNING when the VM booted. A GKE node is Ready the moment the
kubelet registers — before the Pod is scheduled, before the image is pulled, before the weights load, before
XLA compiles. Readiness is what `verify_model_health` reports, and nothing else is evidence:

```
✅ Model health check PASSED.
Response: 'Hello! Yes, I am working. I am...'
Latency: 0.77 seconds.
```

**The endpoint is a Service, not a machine.** This is the single most portable mistake from the VM guides:
a GKE node *does* appear in `gcloud compute instances list`, so reading its external IP succeeds and returns
the wrong address. The model is behind a `LoadBalancer` Service on `:8000`, and only the Service knows where
it is. Across a full teardown and rebuild the Service came back on the same IP, `34.91.103.13`, while the
node behind it changed name.

#### 📊 Benchmark the Model

`run_vllm_benchmark` runs vLLM's own `vllm bench serve` **inside the serving Pod** via `kubectl exec`,
against localhost — there is no SSH here and no second container. The load is TPU-bound and the client is
not, so sharing the Pod's CPU does not distort the result the way sharing the chip would.

**Concurrency sweep**, 1024 input / 128 output tokens:

| Concurrency | Output tok/s | TTFT p99 | TPOT mean | Per-stream tok/s | $/M output (on-demand, all-in) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 199 | 17 ms | 4.9 ms | 203 | $4.48 |
| 2 | 380 | 23 ms | 5.2 ms | 194 | $2.35 |
| 4 | 693 | 29 ms | 5.6 ms | 177 | $1.29 |
| 8 | 1,180 | 45 ms | 6.6 ms | 152 | $0.76 |
| 16 | **1,744** | 118 ms | 8.8 ms | 115 | $0.51 |
| 32 | 1,675 | 216 ms | 18.5 ms | 54 | $0.53 |
| 64 | **2,134** | 392 ms | 28.5 ms | 35 | **$0.42** |

**Throughput is not monotonic, and the dip is real.** Aggregate output falls from 1,744 tok/s at
concurrency 16 to 1,675 at 32, then climbs to 2,134 at 64. That looked like measurement noise, so it was
re-run three times at each of the three points:

| Concurrency | Run 1 | Run 2 | Run 3 | Spread |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 1,777.00 | 1,780.35 | 1,778.87 | 0.2% |
| 32 | 1,694.35 | 1,696.15 | 1,694.24 | 0.1% |
| 64 | 2,350.99 | 2,353.04 | 2,351.38 | 0.1% |

Run-to-run spread is a fifth of a percent, so **the shape reproduces**: concurrency 32 really is slower than
16 on this configuration. The plausible explanation is TPU static-shape padding — vLLM compiles a set of
batch shapes and pads to the nearest, so a batch that straddles a bucket boundary wastes compute on padding.
**That is a hypothesis, not a finding**: it was not verified against the compiled shape list, and it is
recorded in the report as unverified. What is safe to take away is operational rather than causal — *measure
your own concurrency ladder rather than assuming it rises monotonically*, because the arithmetic answer
(more concurrency, more throughput) is wrong here by 4%.

**Context sweep**, concurrency 8, output 128:

| Input tokens | Output tok/s | TTFT p99 | Per-stream tok/s | $/M output (on-demand, all-in) |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 1,185 | 35 ms | 152 | $0.75 |
| 1,024 | 1,180 | 45 ms | 152 | $0.76 |
| 2,048 | 1,083 | 133 ms | 148 | $0.83 |
| 4,096 | 986 | 216 ms | 140 | $0.91 |
| 8,192 | 795 | 399 ms | 122 | $1.12 |
| 16,384 | 487 | 878 ms | 80 | $1.84 |

**A 32× longer prompt costs 2.4× more per output token and 25× the time to first token.** Prefill dominates
as context grows, which is the cost nobody models when they budget by output tokens alone. The 32 GB of HBM
on a v6e leaves roughly 19.8 GiB for KV after this model's weights, so the ceiling here is time, not memory —
16K context at concurrency 8 is comfortably inside the KV budget and still nearly halves throughput.

Each point is emitted as a `throughput.sweep[]` entry conforming to the project's serving-report schema, so
the run files itself as a validated report rather than a screenshot.

#### 💸 Cost Analysis

Every price below was read live from the Cloud Billing Catalog API for `europe-west4`, not from a rate
table. `estimate_deployment_cost` does the lookup at call time for exactly this reason: a hardcoded rate is
wrong eventually and silently.

**The chip, three ways:**

| Provisioning model | Catalog SKU | $/chip-hour | vs on-demand |
| :--- | :--- | ---: | ---: |
| On-demand | `TpuV6e running in Netherlands` | **$2.9700** | — |
| Spot | `TpuV6e attached to Spot Preemptible VMs` | **$1.7820** | −40% |
| Flex-start (DWS) | `DWS Defined Duration V6e` | **$1.3500** | **−55%** |

**On-demand is 2.2× flex-start, and that gap is not a discount — it is three different products.** What the
extra money buys is control over *when* and *for how long*:

- **On-demand** starts the instant you ask and runs until you stop it. Nothing preempts it, nothing bounds
  it. You are paying for unbounded, uninterruptible, immediately-available capacity.
- **Spot** is the same hardware with the guarantee removed. It can be reclaimed at any moment, so it suits
  work that checkpoints or retries — and a serving endpoint is not naturally that.
- **Flex-start** is a *scheduled* grant through Dynamic Workload Scheduler. The request queues until capacity
  exists, then runs uninterrupted for a bounded window. You give up "start now" and get "cheap and
  uninterrupted once started."

For a benchmark run, a demo, or an overnight batch, flex-start is the obvious default — the work is bounded
anyway, so the only thing surrendered is a few minutes at the start.

**Measured on this cluster, end to end.** A flex-start node pool was created alongside the on-demand one and
came up with **zero nodes** — the flex-start shape is an autoscaling pool, not a fixed one, so an idle
flex-start pool costs nothing at all. Scaling the Deployment to two replicas left the second Pod `Pending`,
and **4 minutes 18 seconds later DWS had granted a chip**: node `gke-tpu-49b36129-bxrn` joined, the Pod
landed on it, and about ten minutes after that the second replica was serving. The whole 55% saving costs
roughly four minutes of queue.

That scale-from-zero behaviour is worth more than the rate cut for anything bursty. An on-demand pool bills
from the moment it exists; a flex-start pool bills from the moment a Pod needs it.

**The bill is not only the chip.** Two lines exist on GKE that have no equivalent in a VM deployment:

| Line item | Catalog SKU | $/hour |
| :--- | :--- | ---: |
| System node, `e2-standard-4` | `E2 Instance Core` ×4 + `E2 Instance Ram` ×16 GB | $0.1475 |
| Cluster management fee | `Zonal Kubernetes Clusters` | $0.1000 |
| **Fixed overhead, any provisioning model** | | **$0.2475** |

Google's free tier provides $74.40 in monthly credits per billing account against zonal cluster fees, which
covers roughly one cluster running continuously — so for a single-cluster project the $0.10 line may net to
zero. The system node does not.

**All-in hourly, one chip serving:**

| | Chip | Overhead | **Total/hour** | **Total/month (730 h)** |
| :--- | ---: | ---: | ---: | ---: |
| On-demand | $2.9700 | $0.2475 | **$3.2175** | **$2,349** |
| Spot | $1.7820 | $0.2475 | **$2.0295** | **$1,482** |
| Flex-start | $1.3500 | $0.2475 | **$1.5975** | **$1,166** |

The overhead is 7.7% of an on-demand bill and 15.5% of a flex-start one. **The cheaper the capacity, the
more the fixed cost of running Kubernetes matters** — which is the real cost argument against a cluster for
small deployments, not the cluster fee in isolation.

**Cost per million output tokens**, all-in (chip + overhead), at four points on the ladder:

| Concurrency | Output tok/s | On-demand | Spot | Flex-start |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 199 | $4.48 | $2.83 | $2.23 |
| 8 | 1,180 | $0.76 | $0.48 | $0.38 |
| 16 | 1,744 | $0.51 | $0.32 | $0.25 |
| 64 | 2,134 | $0.42 | $0.26 | **$0.21** |

**Concurrency is worth more than procurement.** Moving from one stream to sixty-four divides the unit cost
by 10.7×; moving from on-demand to flex-start divides it by 2.0×. Doing both is **21×** — $4.48 per million
output tokens becomes $0.21 on identical hardware. A deployment serving one request at a time on on-demand
capacity is paying twenty-one times the achievable rate.

The trade is latency, and the sweep prices it: at concurrency 64 each individual caller sees 35 tokens per
second instead of 203, and a p99 first token at 392 ms instead of 17 ms.

**The line nobody budgets: idle.** With the TPU node pool deleted, the cluster and its system node still bill
$0.2475/hour — **$181 a month for an empty cluster.** Flex-start's scale-to-zero pool is the mitigation:
the pool exists at zero nodes, costs nothing until a Pod demands one, and pulls capacity in ~4 minutes when
one does.

**Cold start has a price too.** The ~11 minutes between the container starting and the model answering is
billed chip time: about $0.54 on-demand, $0.25 on flex-start, every time the Pod is rescheduled. That is an
argument against letting a scheduler move a model Pod around casually.

**Nothing here stops billing on its own.** A Compute Engine instance can carry `--max-run-duration` with
`--instance-termination-action=DELETE` and delete itself. **A node pool has no run bound at all** — not even
under flex-start, where the flag caps how long capacity is *granted*, not how long it is *billed*. The only
thing that stops the meter is deleting the pool:

```bash
make destroy          # deletes the TPU node pool, releases the chip
make destroy-cluster  # deletes everything
```

#### 🐛 What went wrong, and what each cost

Five things broke on the way, and the pattern is worth more than any of them individually: **every one was a
correct habit imported from the Compute Engine version of this deployment.**

1. **`--tpu-topology=1x1`** — refused by the API for a single-host slice, while GKE labels the node `1x1`
   regardless. One failed create.
2. **`gcloud secrets versions access` without `--project`** — resolved to the workstation's default project,
   an expired lab, and failed with a permission error naming a project the rig never mentions.
3. **A dollar sign in a template comment** — the manifest is rendered by `envsubst` from the shell path and
   by `string.Template` from the MCP tool. The former tolerates it, the latter raises. The shell path worked
   and the tool path did not, which is the worst way to find a bug.
4. **The provisioning-model vocabulary drifted** between the shell scripts (`ondemand`) and the tools
   (`on-demand`). Caught by a test, not by a run.
5. **The cost tool told the VM's story** — "flex-start self-terminates at `--max-run-duration`, capping the
   bill." False on a node pool, and a confident wrong statement about money in the one tool written to avoid
   exactly that.

The defence that worked was not more careful reading. It was a test that greps the whole server for
`compute instances create`, `tpu-vm` and `queued-resources` rather than checking one function — because an
earlier version of this project asserted "we are off the old control plane" about the codebase and verified
it on a single call path, while four tools quietly kept using it.

#### 🏁 Conclusion

The strategy for using MCP for Gemma 4 TPU deployment on GKE was validated with an incremental, step by step
approach: provision, deploy, validate, benchmark, price — each step a tool call, each result checked against
the live system rather than against the previous step's assumptions.

What the exercise establishes:

- **A single-chip TPU deployment on GKE works and costs about 20 minutes of cold start.** Two full runs from
  an empty project produced 26 m 31 s and ~28 m end to end, including teardown.
- **The cluster provisions its own TPU**, as a Compute Engine VM inside a managed instance group that it
  owns and you do not.
- **Flex-start is the default worth reaching for on bounded work** — 55% off the chip rate for a ~4 minute
  queue wait, with a pool that costs nothing while idle.
- **Concurrency beats procurement.** Serving at concurrency 32 on flex-start capacity costs $0.29 per
  million output tokens; serving one stream at a time on on-demand costs $4.49 for the same hardware.

For a single small model with one replica, a VM is still the simpler answer and this article's own cost
table says so. GKE earns the overhead when there are replicas to schedule, versions to roll, or an
application already living in the cluster — and when it does, the deployment above is the whole of it.
