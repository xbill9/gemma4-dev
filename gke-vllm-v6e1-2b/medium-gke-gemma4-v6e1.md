# One TPU chip, one Kubernetes cluster: what deploying Gemma 4 to GKE actually costs

A step-by-step deployment of Gemma 4 to Google Kubernetes Engine on a single Cloud TPU v6e chip, driven
entirely by MCP tools — and a cost breakdown that explains why the same chip is priced at $2.97 and $1.35
per hour depending on how you ask for it.

[FIGURE 1 — cover: the terminal showing `get_system_status` with the cluster, node pool, TPU node and pod
all green. Caption: One cluster, one node, one chip, one Pod — the smallest useful shape on GKE.]

The deployment below is deliberately minimal: one zonal cluster, one single-host TPU node pool, one node,
one chip, one vLLM Pod serving `google/gemma-4-E2B-it`. Small enough that every line on the bill is visible
instead of buried in a fleet.

Two questions get answered on the way, and neither has an obvious answer from the GPU-on-a-VM guides:

Does the cluster create its own TPU, or does a virtual machine have to exist first? It creates its own — and
what it creates is still a Compute Engine VM, which turns out to matter enormously for teardown.

Why is on-demand capacity $2.97 per chip-hour when flex-start is $1.35 for the identical silicon? Because
they are not the same product. The difference is measured, not assumed, further down.

#### What this is trying to do

The goal is an agent-driven deployment where every step is a tool call rather than a remembered command
line: provision a cluster and a TPU node pool, deploy vLLM as a Kubernetes Deployment with the chip attached
to the Pod, validate it, benchmark it with a two-dimensional sweep, and price the result from the live Cloud
Billing Catalog rather than from a rate table that goes stale silently.

This is one of three rigs that serve the identical model on the identical chip and differ only in which
control plane provisions the hardware — the Cloud TPU API, Compute Engine, and GKE. Holding the serving
configuration constant is what makes the provisioning path the only variable.

#### Is a cluster even the right answer for a small model?

Worth asking before spending an hour, because for one chip the honest answer is often no.

A single TPU chip serving a 2B-class model runs perfectly well as one VM with `docker run`. Kubernetes earns
its keep when there is more than one replica behind an endpoint, when a bad boot has to be survived without
a human, when a model version has to roll with no downtime, or when the rest of the application already
lives in a cluster.

Two things push against it, and both are sharper for small models than for large ones. A TPU chip is
allocated whole — the node advertises one chip and one Pod takes it, with no MIG-style partitioning — so the
classic small-model argument of bin-packing a dozen services onto one accelerator simply does not apply.
And the fixed overhead of running a cluster is proportionally worse the cheaper the accelerator: about
$0.25 an hour regardless of what else happens, which is noise against a TPU and a third of the bill against
a cheap GPU.

The counterintuitive part is that small models are the easy case for Kubernetes and large ones are the hard
case. A model that fits in one chip scales out as replicas, which is precisely what a Deployment does well.
A model needing eight chips spans a multi-host slice, and that needs gang scheduling and a slice that fails
as a unit.

#### Setting up

GKE needs two client-side tools a Compute Engine deployment does not, and both fail late and confusingly
when missing:

`sudo apt-get install -y kubectl google-cloud-cli-gke-gcloud-auth-plugin`

A preflight target checks those two plus a live gcloud token and the template renderer, and prints the
install command rather than dying halfway through a deploy.

Zone choice here is a quota decision, not a latency one. The two control planes meter against entirely
different pools: this project holds 512 v6e chips in us-east5 under the Cloud TPU API quota and zero
Compute Engine CT6E quota in that same region. GKE spends the Compute Engine pools, so the cluster goes
where the Compute Engine quota and the actual hardware overlap.

[FIGURE 2 — image of the `tpu.env` configuration block. Caption: The env file is configuration; the
directory name is only documentation.]

#### The MCP server

The server is a single Python file on FastMCP, speaking stdio — the simplest transport the SDK supports,
where the agent CLI spawns the server and talks to it over pipes.

The server's name matters more than it looks: it is the key the client registers under, and that key
prefixes every tool. With sibling rigs loaded at once, the prefix is the only thing distinguishing a call to
the GKE rig from a call to the Compute Engine one, so it is derived from the rig directory rather than
typed by hand.

Thirty tools are exposed, covering provisioning, deployment, validation, diagnostics, measurement and
teardown. Every subprocess call goes through one helper using an argument list rather than a shell string,
and every tool returns markdown with an emoji status prefix, because a human reads the output through an
agent transcript.

#### Provisioning: what the cluster actually builds

Two calls. The first creates a zonal cluster with a small default pool that exists only to run system
workloads — keeping kube-dns and metrics-server off the TPU node, so a chip billed by the hour is never
kept alive by CoreDNS after the model Pod is gone. The second creates the node pool that carries the chip.

[GIST 1 + FIGURE 3 — the two gcloud commands. Caption: Cluster first, then the node pool that carries the
chip. Note what is absent from the second command.]

What is absent from that second command is the interesting part. There is no topology flag, and adding one
is an error. The first attempt at this deployment passed a 1x1 topology and was refused outright: TPU
topology cannot be specified for a single-host slice. A one-chip machine type at one node *is* the slice, so
there is nothing to describe — the flag belongs to multi-host slices, where it says how several nodes are
wired into one.

The trap is that GKE then labels the node with that same 1x1 topology anyway. The value is real as a Pod
selector and rejected as a create flag, which is why they live in two separate configuration keys.

And then the answer to the opening question:

[FIGURE 4 — `gcloud compute instances list` and `instance-groups managed list` output showing the
`ct6e-standard-1t` node and its managed instance group. Caption: The node pool is a managed instance group;
the node is an ordinary Compute Engine VM the cluster owns.]

The node pool is implemented as a managed instance group, and the node inside it carries the same machine
type a Compute Engine deployment would pass to instance creation. Same silicon, same attachment mechanism.
What changes is who calls create, and who owns the lifecycle.

Three consequences follow, each of which costs an afternoon if learned the hard way. Deleting that instance
directly is not teardown — it succeeds, and the instance group rebuilds the node within minutes under a new
name, still billing. The node is cattle: replaced on upgrade and repair, so anything written to its disk
vanishes at the next replacement. And the node's name is not yours to choose; across two runs of this
deployment it came back as two different names.

One more thing that surprises people: a node reporting Ready can advertise zero chips for a window, because
the device plugin has not finished registering. A Pod scheduled in that window fails with insufficient TPU
resources, which reads exactly like a quota problem and is not one.

#### Deploying the model

Three things happen that have no VM equivalent: fetch cluster credentials, materialise the Hugging Face
token as a Kubernetes Secret, and apply the manifest.

[GIST 2 + FIGURE 5 — the Pod spec: nodeSelector, toleration, the TPU resource limit and the vLLM args.
Caption: All three of the selector, the limit and the toleration are load-bearing.]

Drop a node selector and the Pod schedules onto the small system node and fails there, which reads as a
model-server problem rather than a placement one. Drop the resource limit and the device plugin never
attaches the chip. No privileged container is needed — on GKE the device plugin handles device access.

The startup probe has to cover the entire load. Image pull, weight load and XLA precompile took ten minutes;
a probe budget shorter than that gets the container killed and restarted forever, one load at a time.

The token never goes through a command line. Creating a Kubernetes secret from a literal argument puts the
token in the process table for every user on the machine, so the tool writes a base64 Secret manifest to a
private temp file and applies that instead.

#### Validating: Running is three steps weaker than serving

[FIGURE 6 — the `get_system_status` dashboard output. Caption: Cluster, node pool, node, Pod and endpoint
are five separate claims, and each can be healthy while the next is not.]

A Queued Resource reached ACTIVE with a node up. A Compute Engine instance was RUNNING when the VM booted.
A GKE node is Ready the moment the kubelet registers — before the Pod is scheduled, before the image is
pulled, before the weights load, before the compiler runs. Only a completion is evidence, and the health
check reports one at 0.77 seconds.

The endpoint is a Service, not a machine. This is the most portable mistake from the VM guides: a GKE node
*does* appear in the Compute Engine instance list, so reading its external IP succeeds and returns the wrong
address. Across a full teardown and rebuild the Service came back on the same IP while the node behind it
changed name entirely.

#### Cold start: about twenty minutes

Provisioning the cluster and node pool took 8 minutes 52 seconds. Applying the Deployment took 6 seconds.
The Pod then took 11 minutes 9 seconds to answer — image pull, weight load, XLA precompile. Tearing the
whole thing down afterwards took 6 minutes 24 seconds. Measured twice from an empty project: about twenty
minutes from nothing to a served token, twenty-six with a teardown first.

That eleven-minute load is billed chip time on every reschedule, which is an argument against letting a
scheduler move a model Pod around casually.

#### Benchmarking, and a dip that turned out to be real

The benchmark runs inside the serving Pod against localhost — no SSH, no second container. The load is
TPU-bound and the client is not, so sharing the Pod's CPU does not distort it.

[FIGURE 7 — concurrency sweep table, 1 to 64 at 1024 input / 128 output. Caption: Aggregate throughput,
tail latency and per-stream speed across the concurrency ladder.]

Aggregate output throughput climbs from 199 tokens per second at concurrency 1 to 1,744 at concurrency 16 —
and then *falls* to 1,675 at concurrency 32, before climbing again to 2,134 at 64.

That looked like noise, so each of the three points was re-run three times. Run-to-run spread came in at one
to two tenths of a percent: 1,777 / 1,780 / 1,779 at concurrency 16, and 1,694 / 1,696 / 1,694 at 32. The
shape reproduces. Concurrency 32 really is about four percent slower than concurrency 16 on this
configuration.

The plausible explanation is TPU static-shape padding — vLLM compiles a set of batch shapes and pads to the
nearest, so a batch straddling a bucket boundary spends compute on padding. That is a hypothesis, not a
finding; it was not verified against the compiled shape list and is recorded as unverified. The safe
takeaway is operational rather than causal: measure your own concurrency ladder instead of assuming it rises
monotonically, because the arithmetic answer is wrong here.

[FIGURE 8 — context sweep table, 512 to 16,384 input at concurrency 8. Caption: A 32x longer prompt costs
2.4x more per output token and 25x the time to first token.]

Prefill dominates as context grows. Throughput falls from 1,185 tokens per second at 512-token prompts to
487 at 16,384, while p99 time to first token goes from 35 milliseconds to 878. This is the cost nobody
models when budgeting by output tokens alone — and note that memory is not the constraint here. The chip's
32 GB of HBM leaves roughly 19.8 GiB for KV cache after this model's weights, so a 16K context at
concurrency 8 sits comfortably inside the budget and still nearly halves throughput. The ceiling is time.

#### What it costs, and why the same chip has three prices

Every price below was read live from the Cloud Billing Catalog for europe-west4.

[FIGURE 9 — the three chip rates with their catalog SKU names. Caption: On-demand $2.97, spot $1.782,
flex-start $1.35 per chip-hour — three products, not three discounts.]

On-demand at $2.97 per chip-hour starts the instant you ask, runs until you stop it, and nothing preempts
it. You are paying for unbounded, uninterruptible, immediately available capacity.

Spot at $1.782 is the same hardware with the guarantee removed — reclaimable at any moment, which suits work
that checkpoints or retries, and a serving endpoint is not naturally that.

Flex-start at $1.35 is a scheduled grant through Dynamic Workload Scheduler. The request queues until
capacity exists, then runs uninterrupted for a bounded window. You give up "start now" and get "cheap and
uninterrupted once started."

That last one was a catalog number rather than a tested path, so it was tested. A flex-start node pool
created alongside the on-demand one came up with *zero nodes* — on GKE, flex-start is an autoscaling shape
rather than a fixed pool, so an idle flex-start pool costs nothing at all. Scaling the Deployment to two
replicas left the second Pod pending, and four minutes eighteen seconds later the scheduler had granted a
chip: a node joined, the Pod landed on it, and ten minutes after that the second replica was serving.

The entire 55 percent saving costs roughly four minutes of queue. For anything bursty, that scale-from-zero
behaviour is worth more than the rate cut — an on-demand pool bills from the moment it exists, a flex-start
pool from the moment a Pod needs one.

[FIGURE 10 — the overhead table: e2-standard-4 system node $0.1475/h, zonal cluster fee $0.10/h, total
$0.2475/h. Caption: Two line items with no VM equivalent, and they do not scale down.]

The bill is not only the chip. A system node and a cluster management fee add about twenty-five cents an
hour whatever the provisioning model — 7.7 percent of an on-demand bill, but 15.5 percent of a flex-start
one. The cheaper the capacity, the more the fixed cost of running Kubernetes matters, and that is the real
cost argument against a cluster for small deployments. Google's free tier credits cover roughly one zonal
cluster's fee; the system node is not covered.

It is also the line nobody budgets. With the TPU node pool deleted, the cluster and its system node still
bill about $181 a month for an empty cluster.

[FIGURE 11 — cost per million output tokens: concurrency 1/8/16/64 across on-demand, spot and flex-start.
Caption: $4.48 to $0.21 per million output tokens on identical hardware.]

Putting the two together produces the number that matters. At concurrency 1 on on-demand capacity, a million
output tokens costs $4.48. At concurrency 64 on flex-start, the same million costs $0.21.

Concurrency divides the unit cost by 10.7. Procurement divides it by 2.0. Together they are a factor of 21 —
on identical silicon, running the identical model. A deployment serving one request at a time on on-demand
capacity is paying twenty-one times the achievable rate, and the trade it is buying is visible in the sweep:
at concurrency 64 each caller sees 35 tokens per second instead of 203.

And nothing here stops billing on its own. A Compute Engine instance can carry a maximum run duration and
delete itself. A node pool has no run bound at all — not even under flex-start, where the setting caps how
long capacity is granted, not how long it is billed. Deleting the pool is the only thing that stops the
meter.

#### Five things that broke, and what they have in common

Every one was a correct habit imported from the Compute Engine version of this deployment.

A topology flag that a single-host slice refuses, while the label it sets appears on the node anyway. A
secrets call without an explicit project, which silently resolved to a stale default project and failed with
a permission error naming a project that appears nowhere in the deployment. A dollar sign in a template
comment, tolerated by one renderer and fatal to the other, so the shell path worked and the tool path did
not. A vocabulary drift between scripts and tools that only a test caught. And a cost tool that confidently
repeated the VM's story about self-terminating capacity — a wrong statement about money, in the one tool
written to avoid exactly that.

The defence that worked was not more careful reading. It was a test that greps the entire server for the old
control plane's commands rather than checking one function, because an earlier version of this project
asserted it had migrated and verified that claim on a single call path — while four tools quietly kept using
the old one.

#### Conclusion

The deployment validates end to end: provision, deploy, validate, benchmark, price — each step a tool call,
each result checked against the live system rather than against the previous step's assumptions.

A single-chip TPU deployment on GKE works and costs about twenty minutes of cold start. The cluster
provisions its own TPU as a Compute Engine VM inside a managed instance group it owns and you do not.
Flex-start is the default worth reaching for on bounded work — 55 percent off the chip rate for a four-minute
queue, with a pool that costs nothing while idle. And concurrency beats procurement by five to one as a cost
lever, provided the latency budget can absorb it.

For one small model with one replica, a VM remains the simpler answer, and the cost table above says so.
Kubernetes earns the overhead when there are replicas to schedule, versions to roll, or an application
already living in the cluster — and when it does, the deployment above is the whole of it.

---

## Assets to render before importing

Medium's importer drops markdown tables entirely and flattens multi-line code blocks, so every one below has
to be an image. Prose above is written to carry the argument if a reader skims the figures.

| Asset | Type | Source |
| --- | --- | --- |
| FIGURE 1 | screenshot | `get_system_status` output — **first image in the body becomes the cover** |
| FIGURE 2 | code image | the `tpu.env` GKE block |
| FIGURE 3 + GIST 1 | code image + gist | `clusters create` and `node-pools create` commands |
| FIGURE 4 | code image | `compute instances list` + `instance-groups managed list` |
| FIGURE 5 + GIST 2 | code image + gist | the Pod spec from `gke/vllm-gemma4.yaml.tmpl` |
| FIGURE 6 | screenshot | `get_system_status` dashboard |
| FIGURE 7 | table image | concurrency sweep, 1–64 |
| FIGURE 8 | table image | context sweep, 512–16,384 |
| FIGURE 9 | table image | the three chip rates + SKU names |
| FIGURE 10 | table image | overhead breakdown |
| FIGURE 11 | table image | $/M output tokens × provisioning model |

Import rules that cost a wasted import each if forgotten: captions must be plain text with no links (a link
inside a figcaption makes Medium drop the whole figure silently); the importer caches by URL and ignores
query strings, so the staged page needs a content-addressed filename; and any canonical link tag must be
stripped from the copy handed to the importer, or Medium serves the canonical URL's cached copy instead.
All headings here are `####` because Medium renders `#` and `##` identically as its one big heading size.
