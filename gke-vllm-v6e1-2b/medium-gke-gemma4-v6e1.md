# Gemma 4 on GKE with Cloud TPU v6e: step by step with MCP and an Agent CLI

This article provides a step by step deployment guide for Gemma 4 to a Google Kubernetes Engine cluster
backed by a Cloud TPU v6e (Trillium) chip. A suite of Python MCP tools is built to simplify management of
the vLLM hosted Gemma 4 deployment from an agent CLI.

Everything below was run end to end, twice, from an empty project. Every command and output shown is real.

[FIGURE 1 — cover: the `get_system_status` output showing cluster, node pool, TPU node, Pod and endpoint all
green. Caption: One cluster, one node, one chip, one Pod — the smallest useful shape on GKE.]

#### What is this project trying to Do?

We want one command per lifecycle step and no remembered command lines. The agent should provision the
hardware, deploy the model, tell us whether it is actually serving, benchmark it, and price it.

The deployment is deliberately the smallest useful shape on GKE: one zonal cluster, one single-host TPU node
pool, one `ct6e-standard-1t` node carrying a single v6e chip, and one vLLM Pod serving
`google/gemma-4-E2B-it`. Small enough that every line on the bill is visible and every failure has exactly
one place to be.

#### Where do I start?

At this point you should have a GCP project with billing enabled and Compute Engine CT6E quota in the region
you plan to use, `gcloud` installed and authenticated, a Hugging Face token with access to Gemma 4, and the
rig checked out from GitHub. Everything below happens inside the rig directory.

#### Quota: check this before anything else

The two TPU control planes meter against completely different pools, and holding one buys nothing on the
other. Our project holds 512 v6e chips in us-east5 under the Cloud TPU API quota, and zero Compute Engine
CT6E quota in that same region. GKE spends the Compute Engine pools.

Find where the machine type is published:

`gcloud compute machine-types list --filter="name=ct6e-standard-1t" --format="value(zone)"`

Quota is the other half, and the two ids do not behave the same way. On-demand draws on
`TPUS-PER-TPU-FAMILY-per-project-region`, which defaults to **zero** where unset. Spot and flex-start draw
on `PREEMPTIBLE-TPU-V6E-per-project-region`, which defaults to **1536**. Reading only the first writes off
regions with plenty of flex-start headroom.

And quota is a ceiling, not an allocation. Every zone we probed held 1536 chips of preemptible quota, and
four of five had no capacity at all. A spot create is the cheap probe — it fails fast with a stockout where
flex-start silently queues.

#### Setup the Basic Environment

GKE needs two client tools a plain VM deployment does not:

`sudo apt-get install -y kubectl google-cloud-cli-gke-gcloud-auth-plugin`

Then authenticate both ways — a user credential for gcloud, and application default credentials for the
client libraries:

`gcloud auth login`

`gcloud auth application-default login`

Store the Hugging Face token in Secret Manager once, so it never lands in a manifest or a shell history:

`echo -n "hf_your_token" | gcloud secrets create hf-token --data-file=- --project=YOUR_PROJECT`

Then check all of it at once:

`make gke-preflight`

That target exists because every one of those four pieces fails *late* otherwise — halfway through a deploy,
with an error that does not name the missing piece.

[FIGURE 2 — the `tpu.env` configuration block. Caption: The directory name is documentation; the env file is
configuration, and it is committed.]

A real environment variable always beats the file, so a one-off run against another zone needs no edit at
all. And never copy a value out of the directory name into a command: v6e1 is for humans, gcloud wants
`ct6e-standard-1t`, and that string lives in the env file.

#### Model Management Tool with MCP Stdio Transport

The MCP server is a single Python file built on FastMCP. The simplest transport the SDK supports is stdio,
which connects a locally running process — the agent CLI spawns the server and talks to it over pipes.

`mcp = FastMCP(MCP_SERVER_NAME)`

The server name is not cosmetic. It is the key the client registers under, and that key prefixes every tool.
With sibling rigs loaded at once, the prefix is the only thing telling the GKE rig's tools from the Compute
Engine rig's, so we derive it from the directory name rather than typing it.

#### Running the Python Code

Install and test before wiring anything up:

`make install`

`make test`

The tests are offline — they mock the MCP module and the Google Cloud clients before importing the server —
so all 48 pass without touching a cloud API. Then `make lint` runs ruff and mypy, and `make run` starts the
server by hand once, just to watch it come up.

[FIGURE 3 — the agent CLI mcp_config.json registering the server. Caption: Four lines. The registration key
is what prefixes every tool.]

#### Validation with the Agent CLI

Ask the agent what it can do. The `get_help` tool renders itself from the live tool list, so it cannot go
stale. Thirty tools come back, grouped by purpose: provisioning, deployment, validation, diagnostics,
measurement and teardown.

#### Getting Started with Gemma 4 on TPU

Before spending anything, ask what it will cost. The `estimate_deployment_cost` tool reads the Cloud Billing
Catalog live at call time — there is no rate table in this project, because a hardcoded rate is wrong
eventually and wrong silently.

[FIGURE 4 — `estimate_deployment_cost` output for on-demand. Caption: $2.97 per chip-hour, read live, with
the warning that nothing stops the bill but teardown.]

#### Create the GKE Cluster

One call, and it is idempotent — run it twice and the second run says the cluster is already there.

[GIST 1 + FIGURE 5 — the `gcloud container clusters create` command and its output. Caption: A zonal cluster
with a small system pool. About nine minutes.]

Two decisions in that command are worth knowing. Pin the release channel, never a version: TPU v6e needs a
recent control plane, and any version we pin goes stale — rapid gave us 1.36.3-gke.1537000. And the small
default pool exists only for system workloads, because a chip billed by the hour must not be kept alive by
CoreDNS after the model Pod is gone.

#### Create the TPU Node Pool

This is the step that gets us a chip.

[GIST 2 + FIGURE 6 — the `gcloud container node-pools create` command and its output. Caption: One
ct6e-standard-1t node. Note what is absent from the command.]

Do not add a topology flag to that command. Our first attempt did, and the API refused it outright: TPU
topology cannot be specified for a single-host TPU slice pool. A one-chip machine type at one node *is* the
slice — there is nothing to describe, and the flag belongs to multi-host slices.

What makes it a trap rather than a typo is that GKE then labels the node with that same 1x1 topology
anyway. The value is real as a Pod selector and rejected as a create flag, so we keep them in two separate
configuration keys.

#### What Actually Got Created

Here is the part nobody tells you: the cluster created its own TPU, and what it created is a Compute Engine
VM.

[FIGURE 7 — `gcloud compute instances list` and `gcloud compute instance-groups managed list` output.
Caption: The node pool is a managed instance group; the node is an ordinary Compute Engine VM the cluster
owns.]

The node inside that group carries the same machine type we would pass to `gcloud compute instances create`
on a VM deployment. Same silicon, same attachment mechanism. What changed is who calls create and who owns
the lifecycle.

Three consequences follow, each of which costs an afternoon if learned the hard way. Deleting that instance
directly is not teardown — it succeeds, and the instance group rebuilds the node minutes later under a new
name, still billing; delete the pool instead. The node is cattle, replaced on upgrade and repair, so
anything written to its disk is gone. And we do not choose the node's name: across two runs it came back
as two entirely different ones.

One more surprise worth knowing before it bites: a node reporting Ready can advertise zero TPU chips for a
window, because the device plugin has not finished registering. A Pod scheduled in that window fails with
insufficient TPU resources, which reads exactly like a quota problem and is not one. Wait thirty seconds.

#### Deploy The Model

The `deploy_vllm` tool does three things with no VM equivalent: fetches cluster credentials, copies the
Hugging Face token from Secret Manager into a Kubernetes Secret, and applies the manifest.

[GIST 3 + FIGURE 8 — the Pod spec: nodeSelector, toleration, TPU resource limit, vLLM args and startup
probe. Caption: Four things in here are load-bearing.]

Both node selectors are load-bearing — drop one and the Pod schedules onto the small system node and fails
there, which reads as a model-server problem rather than a placement one. The TPU resource limit is what
makes the device plugin attach the chip. The toleration matches a taint GKE applies automatically. And the
startup probe budget of thirty minutes exists because the load took ten: a shorter budget gets the container
killed and restarted forever, one ten-minute load at a time.

No privileged container is needed. On GKE the device plugin handles device access.

#### Checking System status

[FIGURE 9 — the `get_system_status` dashboard. Caption: Five separate claims, and each can be healthy while
the next is not.]

Running is three steps weaker than serving on this path. A GKE node is Ready the moment the kubelet
registers — before the Pod is scheduled, before the image is pulled, before the weights load, before the
compiler runs. While it loads, `get_vllm_pod_logs` shows the actual work.

#### Cross Check The Deployed Model

Never trust one layer. Go under the tools and ask Kubernetes directly with `kubectl get pods` and
`kubectl get svc`, then bypass Kubernetes too and ask the model server:

`curl -s http://EXTERNAL_IP:8000/v1/models`

The endpoint is a Service, not a machine, and this is the most portable mistake from the VM guides: a GKE
node *does* appear in the Compute Engine instance list, so reading its external IP succeeds and returns the
wrong address. Across a full teardown and rebuild our Service came back on the same IP while the node behind
it changed name entirely.

#### Review the Model

[FIGURE 10 — `verify_model_health` and a `query_queued_gemma4` completion. Caption: Health check at 0.77
seconds, and the model answering a question about its own hardware.]

One caution: use the chat endpoint, not raw completions. `/v1/completions` returns an empty completion on
instruction-tuned models, so an empty result there is expected rather than a broken deploy.

#### How long did all that take?

[FIGURE 11 — the timing table: provision 8m52s, deploy 6s, load 11m09s, teardown 6m24s. Caption: About
twenty minutes from nothing to a served token, measured twice.]

That eleven-minute load is billed chip time every time the Pod is rescheduled, which is a good argument
against letting a scheduler move a model Pod around casually.

#### Benchmark the Model

The benchmark runs vLLM's own tool inside the serving Pod against localhost — no SSH, no second container.
The load is TPU-bound and the client is not, so sharing the Pod's CPU does not distort the numbers.

[FIGURE 12 — concurrency sweep, 1 to 64 at 1024 input / 128 output. Caption: Aggregate throughput, tail
latency and per-stream speed across the ladder.]

Throughput is not monotonic, and the dip is real. Concurrency 32 comes in below concurrency 16. That looked
like noise, so we ran each of those points three more times.

[FIGURE 13 — the repeat table. Caption: 0.1 to 0.2 percent run-to-run spread. The shape reproduces.]

Concurrency 32 really is about four percent slower than 16 here. The plausible cause is TPU static-shape
padding — vLLM compiles a set of batch shapes and pads to the nearest, so a batch straddling a bucket
boundary spends compute on padding. That is a hypothesis and we have not verified it against the compiled
shape list; it is recorded as unverified. The operational lesson stands on its own: measure your own
concurrency ladder, because the arithmetic answer is wrong here.

[FIGURE 14 — context sweep, 512 to 16,384 input at concurrency 8. Caption: A 32x longer prompt costs 2.4x
per output token and 25x the time to first token.]

Prefill dominates as context grows. Memory is not the limit — the chip's 32 GB of HBM leaves roughly 19.8
GiB for KV cache after this model's weights, so 16K at concurrency 8 sits well inside the budget and still
nearly halves throughput. The ceiling is time.

#### Cost Breakdowns

Every price here was read live from the Cloud Billing Catalog for europe-west4.

[FIGURE 15 — the three chip rates with catalog SKU names and medal ranking. Caption: Flex-start $1.35, spot
$1.782, on-demand $2.97 per chip-hour — three products, not three discounts.]

On-demand starts the instant we ask, runs unbounded, and nothing preempts it. Spot is the same hardware with
the guarantee removed. Flex-start is a scheduled grant: the request queues until capacity exists, then runs
uninterrupted for a bounded window.

We tested the cheap one rather than quoting it. A flex-start node pool came up with zero nodes — on GKE
flex-start is an autoscaling shape rather than a fixed pool, so an idle flex-start pool costs nothing at
all. We scaled the Deployment to two replicas, the second Pod went pending, and four minutes eighteen
seconds later the scheduler granted a chip: a node joined, the Pod landed, and ten minutes after that the
replica was serving. The entire fifty-five percent saving costs about four minutes of queue.

[FIGURE 16 — the overhead table: system node $0.1475/h, cluster fee $0.10/h, total $0.2475/h. Caption: Two
line items with no VM equivalent, and they do not scale down.]

The overhead is 7.7 percent of an on-demand bill but 15.5 percent of a flex-start one. The cheaper the
capacity, the more the fixed cost of running Kubernetes matters — that, and not the cluster fee in
isolation, is the real cost argument against a cluster for a small deployment. Google's free tier credits
cover roughly one zonal cluster's fee; the system node is not covered.

[FIGURE 17 — all-in hourly and monthly by provisioning model, with medals. Caption: $1,166 to $2,349 a
month for the same chip serving the same model.]

[FIGURE 18 — cost per million output tokens across concurrency and provisioning model. Caption: $4.48 to
$0.21 per million output tokens on identical hardware.]

Concurrency is worth more than procurement. One stream to sixty-four divides unit cost by 10.7; on-demand to
flex-start divides it by 2.0. Both together is a factor of 21, on identical silicon running the identical
model. The trade is visible in the sweep: at concurrency 64 each caller sees 35 tokens per second instead of
203.

And the line nobody budgets — with the TPU node pool deleted, the cluster and its system node still bill
about $181 a month for an empty cluster.

#### Tearing It Down

Nothing here stops billing on its own. A Compute Engine instance can carry a maximum run duration and delete
itself. A node pool has no run bound at all — not even under flex-start, where the setting caps how long
capacity is granted, not how long it is billed.

Release the chip and keep the cluster:

`make destroy`

Or take it all:

`make destroy-cluster`

The same two are available as `destroy_tpu_node_pool` and `destroy_gke_cluster` from the agent.

#### Five things that broke, and what they had in common

Every one was a correct habit imported from the Compute Engine version of this deployment. A topology flag
that a single-host slice refuses, while the label it sets appears on the node anyway. A secrets call without
an explicit project, which silently resolved to a stale default project. A dollar sign in a template
comment, tolerated by one renderer and fatal to the other, so the shell path worked and the tool path did
not. A vocabulary drift between scripts and tools that only a test caught. And a cost tool that confidently
repeated the VM's story about self-terminating capacity — a wrong statement about money, in the one tool
written to avoid exactly that.

The defence that worked was not more careful reading. It was a test that greps the whole server module for
the old control plane's commands rather than checking one function, because an earlier version of this
project asserted it had migrated and verified that claim on a single call path — while four tools quietly
kept using the old one.

#### Summary

The strategy for using MCP for Gemma 4 TPU deployment on GKE was validated with an incremental step by step
approach. A minimal stdio transport MCP server was started from Python source, registered with an agent CLI
in the same local environment, and then used to provision a cluster, attach a TPU node pool, deploy vLLM,
validate the endpoint, benchmark it and price it — each step checked against the live system rather than
against the previous step's assumptions.

Having run it twice end to end: a single-chip TPU deployment on GKE takes about twenty minutes from nothing
to a served token. The cluster provisions its own TPU as a Compute Engine VM inside a managed instance group
it owns and we do not, so deleting that VM is not teardown and deleting the pool is. Flex-start is the
default worth reaching for on bounded work — fifty-five percent off the chip rate for a four-minute queue,
with a pool that costs nothing while idle. And concurrency beats procurement as a cost lever by five to one,
provided the latency budget can absorb it.

For one small model with one replica, a plain VM remains the simpler answer and the cost tables above say
so. Kubernetes earns its overhead when there are replicas to schedule, versions to roll, or an application
already living in the cluster. When it does, the twenty minutes above is the whole of it.

---

## Assets to render before importing

Medium's importer drops markdown tables entirely and flattens multi-line code blocks, so each item below has
to be an image. **Single-line commands are left as real code blocks** — those import fine. The prose is
written to carry the argument if a reader skims every figure.

| Asset | Type | Source |
| --- | --- | --- |
| FIGURE 1 | screenshot | `get_system_status` — **first image in the body becomes the cover** |
| FIGURE 2 | code image | the `tpu.env` GKE block |
| FIGURE 3 | code image | `mcp_config.json` |
| FIGURE 4 | screenshot | `estimate_deployment_cost` output |
| FIGURE 5 + GIST 1 | code image + gist | `clusters create` command and output |
| FIGURE 6 + GIST 2 | code image + gist | `node-pools create` command and output |
| FIGURE 7 | code image | `compute instances list` + `instance-groups managed list` |
| FIGURE 8 + GIST 3 | code image + gist | the Pod spec from the manifest template |
| FIGURE 9 | screenshot | `get_system_status` dashboard |
| FIGURE 10 | screenshot | `verify_model_health` + a completion |
| FIGURE 11 | table image | timing table |
| FIGURE 12 | table image | concurrency sweep 1–64 |
| FIGURE 13 | table image | repeat runs at c=16/32/64 |
| FIGURE 14 | table image | context sweep 512–16,384 |
| FIGURE 15 | table image | three chip rates + SKUs |
| FIGURE 16 | table image | overhead breakdown |
| FIGURE 17 | table image | all-in hourly and monthly |
| FIGURE 18 | table image | $/M output tokens |

Three import rules that each cost a wasted import if forgotten: a link inside a figcaption makes Medium drop
the whole figure silently, so captions must be plain text; the importer caches by URL and ignores query
strings, so the staged page needs a content-addressed filename; and any canonical link tag must be stripped
from the copy handed to the importer, or Medium serves the canonical URL's cached copy instead. All headings
are `####` because Medium renders `#` and `##` identically as its single big heading size.
