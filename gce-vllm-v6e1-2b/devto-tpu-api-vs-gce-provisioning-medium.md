
# The unofficial TPU migration guide: Cloud TPU API to Compute Engine

The Cloud TPU API is no longer under active development. Bug fixes and security updates only, and that includes its gcloud CLI and the Cloud Client Libraries.

The official word is in two places:

- [Introduction to Cloud TPU](https://docs.cloud.google.com/tpu/docs/intro-to-tpu)
- [Cloud TPU resources in Compute Engine](https://docs.cloud.google.com/tpu/docs/tpus-in-compute-engine)

No sunset date is published, so nothing breaks on a deadline. But TPU7x (Ironwood) and everything after it is Compute Engine or GKE only. The API you are on today is the API your next chip will not support.

So I moved a rig over — a v6e-1 (Trillium) chip serving `gemma-4-E2B-it` under vLLM, rebuilt on `gcloud compute instances`. Same chip, same checkpoint, same serving flags, only the control plane changed.

The flag mapping took an afternoon. Everything else took the rest of the week, because **almost nothing on this path fails loudly.** What follows is what changes, what bit me, and how to tell one failure from another.

## What actually changes

The short version, so the rest makes sense.

Old:

```
gcloud alpha compute tpus queued-resources create vllm-gemma4-qr \
  --node-id=vllm-gemma4-qr-node \
  --zone=us-east5-b \
  --accelerator-type=v6e-1 \
  --runtime-version=v2-alpha-tpuv6e \
  --provisioning-model=flex-start \
  --valid-until-duration=2h
```

New:

```
gcloud compute instances create gce-vllm-v6e1-2b \
  --zone=europe-west4-a \
  --machine-type=ct6e-standard-1t \
  --image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e \
  --image-project=ubuntu-os-accelerator-images \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=200GB \
  --scopes=cloud-platform \
  --metadata-from-file=startup-script=/tmp/startup.sh \
  --provisioning-model=FLEX_START \
  --request-valid-for-duration=2h \
  --max-run-duration=4h \
  --instance-termination-action=DELETE
```

```
Cloud TPU API                       Compute Engine
----------------------------------  -------------------------------------------------
--accelerator-type=v6e-1            --machine-type=ct6e-standard-1t
--runtime-version=v2-alpha-tpuv6e   --image-family=ubuntu-accel-... + --image-project
--valid-until-duration              --request-valid-for-duration
--provisioning-model=flex-start     --provisioning-model=FLEX_START
QR produces a node named <id>-node  the instance is the node
gcloud compute tpus tpu-vm list     gcloud compute instances list
gcloud compute tpus tpu-vm ssh      gcloud compute ssh
```

One documentation trap before you start. [Request TPU Flex-start VMs](https://docs.cloud.google.com/tpu/docs/request-using-flex-start) states that "You must use the queued resources API to use TPU Flex-start VMs." Ignore it — that page sits inside the deprecated API's own doc set and describes flex-start within that API. The Compute Engine [provisioning models](https://docs.cloud.google.com/compute/docs/instances/provisioning-models) page lists v5p, v6e and TPU7x as flex-start machine series, `instances create` takes `FLEX_START` as a first-class value, and `--request-valid-for-duration` is its wait knob. Every flex-start instance in this article was created that way.

Two genuine wins while we are here. `--max-run-duration` is no longer flex-start-only, so spot and on-demand can self-terminate too — pair it with `--instance-termination-action=DELETE` and a demo VM cleans up after itself. And the two-object lifecycle collapses: no queued resource that owns a node you did not name, no reconciling the two, no `--force` on teardown.

Serving does not change at all. Same chip, same engine build, same flags, the KV cache allocation came out at **1,151,744 tokens on both control planes** — the same integer, not merely close — and throughput matched to 0.6% on the control cells. **This is not a performance decision.** Plan it as a refactor.

Now the parts that cost real time.

## Not every chip has a Compute Engine path

Check before you plan anything, and do not check by looking in the machine-type catalog, because it will tell you yes when the answer is no.

v5e looks fine there:

```
$ gcloud compute machine-types list --filter="name~ct5lp"
NAME              ZONE           CPUS  MEMORY_GB  GUEST_ACCELERATOR_TYPE
ct5lp-hightpu-1t  us-central1-a  24    48.00      ['ct5lp']
ct5lp-hightpu-4t  us-central1-a  112   192.00     ['ct5lp']
ct5lp-hightpu-8t  us-central1-a  224   384.00     ['ct5lp']
```

Three shapes, 26 zones. Now create one:

```
ERROR: (gcloud.compute.instances.create) Could not fetch resource:
 - This user agent is not allowed to use the machine type [ct5lp-hightpu-1t].
```

Refused outright. Not a quota error, not a does-not-exist error.

Those machine types are in the catalog because **GKE node pools are created with exactly those strings** — that is the consumer, not you. Same explanation for the other things that look like a v5e path: the image family is literally named `ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e`, and there is a `compute.googleapis.com` quota metric called `TPU-LITE-PODSLICE-V5-per-project-zone`. The Cloud TPU API and GKE are both built on Compute Engine, so Compute-Engine-shaped artifacts exist for v5e whether or not you can create one.

The public docs agree, if you read them closely. [TPU machines in the accelerator-optimized family](https://docs.cloud.google.com/compute/docs/tpus/tpu-machines) lists the supported versions as TPU7x, v6e and v5p — no v5e. And the [TPU v5e](https://docs.cloud.google.com/tpu/docs/v5e) page says v5e "is supported using Google Kubernetes Engine and the Cloud TPU API", with Compute Engine absent from that list.

**Catalog presence is not creatability.** Testing costs nothing: pick a zone where your quota is zero and try the create. A rejection is free and conclusive.

## Quota is the gate, and it is not the quota you expect

Three separate things went wrong for me here, so take them in order.

### 1. Your TPU API quota does not come with you

The two control planes meter against completely disjoint pools. My project holds **512 v6e chips in us-east5 on the TPU API** and, on Compute Engine, held **nothing at all** in the same region for the same silicon. That is why my first create failed in the zone my rig had used happily for months.

### 2. Which quota you spend depends on the provisioning model

There are two v6e quotas on Compute Engine, and picking the wrong one to check is the easiest mistake to make:

```
Provisioning model    Quota id it spends
--------------------  ------------------------------------------------------------------------
FLEX_START            PREEMPTIBLE-TPU-V6E-per-project-region, falling back to the family quota
SPOT                  PREEMPTIBLE-TPU-V6E-per-project-region
STANDARD (on-demand)  TPUS-PER-TPU-FAMILY-per-project-region, tpu_family=CT6E
```

**Flex-start spends the preemptible pool.** That is counterintuitive — flex-start is not preemptible in behaviour, once granted it runs uninterrupted for up to seven days — and nothing in the flag names hints at it. The [provisioning models](https://docs.cloud.google.com/compute/docs/instances/provisioning-models) page says so, and the second sentence matters as much as the first:

> When you create a Flex-start VM, preemptible quota is consumed. If your project lacks preemptible quota, then standard quota is consumed.

So for flex-start, **a region is usable if either pool has room.** Check the preemptible metric first, because that is what gets spent, but do not write a region off on one listing alone.

I spent a day trying to establish this experimentally before finding it documented — and then, having found it, still got it wrong by quoting only the first sentence. Read the whole entry.

Note there is **no non-preemptible v6e id at all** — no `TPU-V6E-per-project-region` exists — which is why on-demand falls back to the generic family quota. v4, v5e and v5p each publish their own dedicated pair, so this fallback is a v6e and TPU7x quirk rather than a rule.

### 3. The obvious command does not show either of them

This is the part that cost me the most, because it answers confidently and wrongly:

```
$ gcloud compute regions describe us-east5 --format="value(quotas.list())" | tr ',' '\n' | grep TPU
TPU_LITE_DEVICE_V5               0.0
PREEMPTIBLE_TPU_LITE_DEVICE_V5   0.0
TPU_LITE_PODSLICE_V5             32.0
PREEMPTIBLE_TPU_LITE_PODSLICE_V5 1536.0
```

Four metrics, all v5e, **none of which governs v6e.** The regional quota view only carries the older metrics. v6e lives in the newer Cloud Quotas API and has to be asked for by name — once per metric:

```
$ gcloud alpha quotas info describe PREEMPTIBLE-TPU-V6E-per-project-region \
    --service=compute.googleapis.com          # flex-start and spot

$ gcloud alpha quotas info describe TPUS-PER-TPU-FAMILY-per-project-region \
    --service=compute.googleapis.com          # on-demand
```

**Read both, because their defaults are opposite.** A region absent from the family listing inherits **0**. A region absent from the preemptible listing inherits **1536**. So a region that looks dead in one listing may have plenty of headroom in the other — which is exactly the mistake I made, writing off regions as unusable when only their on-demand path was.

An unset value also reads identically to a zero one, so a blank does not tell you the hardware is missing. Check `machine-types list` for that.

### What my project actually holds

After the requests below, for the twelve regions that publish `ct6e-standard-1t`:

```
region                                                                                                                      flex-start / spot  on-demand
--------------------------------------------------------------------------------------------------------------------------  -----------------  ---------
europe-west4, asia-east1, asia-northeast1, asia-south1, asia-southeast1, southamerica-east1, southamerica-west1, us-south1               1536         32
us-east1                                                                                                                                 1536         32
us-central1, us-west1                                                                                                                    1536          0
us-east5                                                                                                                                   32         32
us-east4                                                                                                                                    0          0
```

Two things worth reading off that.

`us-central1` and `us-west1` look unusable if you only check on-demand, and are in fact fine for flex-start — they hold the full 1536 on the pool flex-start actually spends. That is the opposite-defaults trap doing real damage: I wrote both off for a day.

`us-east5` sits at **32** where every other live region has 1536. I put it there by asking for 32, not realising the preemptible metric defaults to 1536. When I noticed and went back to ask for 1536, **that request was denied** — so the 32 was not the self-inflicted ceiling it looked like. us-east5 simply is not giving out more today, whatever number you put in the form.

### Requesting more: what was accepted and what was refused

One command per metric. The dimension keys differ — the family quota takes `region` **and** `tpu_family`, the preemptible one takes `region` alone:

```
gcloud quotas preferences create \
  --service=compute.googleapis.com --project=YOUR_PROJECT \
  --quota-id=PREEMPTIBLE-TPU-V6E-per-project-region \
  --dimensions="region=us-east5" \
  --preferred-value=32 \
  --preference-id=preemptible-tpu-v6e-us-east5 \
  --justification="..."
```

I attempted eleven of these across five regions. Every decision came back within seconds:

```
Attempt                                                                                   Outcome
----------------------------------------------------------------------------------------  ---------------------
us-east5 preemptible -> 32, us-east5 family -> 32, us-east1 family -> 32                  approved
us-central1 family, us-west1 family, us-east4 family, us-east4 preemptible — all 0 -> 32  denied
us-east5 preemptible, 32 -> 1536                                                          denied
us-central1 / us-east1 / us-west1 preemptible -> 32                                       refused at submission
```

Look at the first two rows together: **the same request, 0 → 32, was approved in two regions and denied in three.** Nothing about the form differs. So there is no "correct" number that gets you approved — whether you are granted anything is a property of the region, and the only way to find out is to ask. Asking is free and the answer arrives in seconds, so ask.

The three refusals were my own fault and worth knowing about. Those regions already sat at the 1536 default, so asking for 32 was a *decrease*:

```
FAILED_PRECONDITION: The quota override ... decreases effective quota unsafely
```

**Read the current value per metric before asking.** Because the two metrics carry different defaults, one blanket number for both is wrong roughly half the time.

The denials came back as fast as the approvals, with `quotaConfig.stateDetail` reading `Quota request denied` and no human apparently involved. They clustered in the busier regions, which suggests capacity pressure rather than policy — though that is inference, not something I can confirm. Check any request with:

```
gcloud quotas preferences list --project=YOUR_PROJECT
```

### Quota is a ceiling, not an allocation

Worth saying plainly, because everything above is about getting quota and none of it gets you a chip: **holding quota does not mean the hardware is there.** Capacity for single v6e chips was tight in every region I touched.

I probed five zones with a spot create, which fails fast and names the reason:

```
zone            quota held  result
--------------  ----------  --------------------------------------------
europe-west4-a        1536  provisioned
us-central1-a         1536  reason: stockout
us-central1-b         1536  provisioned, then stocked out a minute later
us-central1-c         1536  reason: stockout
us-west1-c            1536  reason: stockout
```

Every one of those zones had 1536 chips of quota. Four of five had no hardware:

```
reason: stockout
zonesAvailable: ''
message: The zone '.../zones/us-central1-a' does not have enough resources
  available to fulfill the request.
```

`us-central1-b` is the instructive one. A spot instance came up there, I deleted it, and a flex-start request a minute later was refused for stockout. **Availability moves faster than you can run a controlled test against it**, let alone plan around it.

So treat quota as permission to ask, not as reserved hardware. If you need a chip at a particular moment, flex-start's queue is the mechanism for that — it waits for capacity rather than failing — and the seven-day maximum run duration exists precisely because getting hold of one is the hard part.

And when a request denial arrives, capacity is the likely reason rather than policy: my denials clustered in exactly the regions that later refused spot creates for stockout.

## PENDING means two different things

This is the one most likely to cost you an afternoon.

A flex-start create with **no quota** does not error. It is accepted and queues:

```
NAME                       ZONE        STATUS   PROVISIONING_MODEL
v6e-quota-probe-delete-me  us-east5-b  PENDING  FLEX_START
```

A flex-start create with **no capacity** does exactly the same thing. I produced both, separately, and they are indistinguishable from outside.

It is not even consistent. In a third zone the same flex-start create came back immediately with an explicit `reason: stockout` instead of queueing. So flex-start sometimes queues and sometimes fails fast on the same condition, and you cannot infer the cause from which behaviour you get.

The practical consequence: **you cannot use "did the create succeed" as a quota test**, and a stuck request tells you nothing about why it is stuck.

**The trick: fire a SPOT create at the same zone.** Spot does not queue. It fails immediately with an explicit reason, which makes it a capacity probe you can run in seconds and which costs nothing when it fails:

```
$ gcloud compute instances create probe --zone=us-central1-a \
    --machine-type=ct6e-standard-1t --provisioning-model=SPOT ...

reason: stockout
zonesAvailable: ''
message: The zone '.../zones/us-central1-a' does not have enough resources
  available to fulfill the request.
```

Stockout means your flex-start request is queued behind real scarcity and more quota will not help. If spot provisions instead, capacity exists and quota is the thing to go check. (Spot may draw on a different pool than flex-start, so this probes the zone, not your entitlement.)

## The image is not the runtime version

The first boot of my migrated rig died 100 seconds in, for a reason no flag mapping would have caught:

```
+ sudo docker pull vllm/vllm-tpu:nightly
sudo: docker: command not found
...
ERROR: Failed to pull vLLM Docker image after multiple retries. Exiting.
```

**`ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e` has no `docker` on PATH at first boot.** The TPU API runtime versions — `v2-alpha-tpuv6e` and friends — do. My startup script came across verbatim and went straight for the pull.

Install it first:

```
if ! command -v docker > /dev/null 2>&1; then
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker.io
  sudo systemctl enable --now docker
fi
```

And fix it in three places, not one: the startup script, any Docker command your tooling runs over SSH (the recovery tool you grab after a failed boot must not fail the same way), and any copy-pasteable deploy one-liner you emit.

**The general form:** your startup script was written against a runtime version that gave you things for free. Mine assumed Docker. Whatever yours assumes, the instance will sit there reporting `RUNNING` while it fails.

## RUNNING does not mean ready

This is the most misleading signal on the new path.

A queued resource reached `ACTIVE` only once its node was up. **An instance is `RUNNING` the moment the VM boots** — before the startup script has pulled an image, loaded a model, or done anything at all. During the entire failed boot above, the instance list said:

```
NAME              ZONE            MACHINE_TYPE      STATUS
gce-vllm-v6e1-2b  europe-west4-a  ct6e-standard-1t  RUNNING
```

It said that indefinitely. Nothing distinguishes a dead boot from a healthy one except reading the startup log or curling the port. Any readiness check you ported that trusted `ACTIVE` is now wrong — by several minutes on a good day, and forever on a bad one.

## Discovery and SSH move, and the old calls go quiet

A `ct6e-*` instance is an ordinary Compute Engine instance that happens to carry a TPU, so the old API cannot see it:

```
$ gcloud compute instances list --filter="name=gce-vllm-v6e1-2b"
NAME              ZONE            MACHINE_TYPE      STATUS
gce-vllm-v6e1-2b  europe-west4-a  ct6e-standard-1t  RUNNING

$ gcloud compute tpus tpu-vm list --zone=europe-west4-a
$
```

Empty. No error, no warning — your tooling simply believes nothing is running. Two field shapes move with it: status is `status: RUNNING` rather than `state: READY`, and the external IP moves from `networkEndpoints[].accessConfig.externalIp` to `networkInterfaces[].accessConfigs[].natIP`. Copy the old status check across and it will not throw; it will just sort every healthy instance to the bottom of your ranking, which you notice the day you have two.

SSH moves too, and this is the call site people miss:

```
gcloud compute tpus tpu-vm ssh <node>   # old
gcloud compute ssh <instance>           # new
```

Everything that manages your container, tails logs, reads journalctl or runs a benchmark has to move — and those are precisely the tools you reach for *when something has already gone wrong*.

I had a test asserting my rig was off the old API. It covered the discovery function. Four other tools were still calling `tpu-vm ssh` behind its back, plus several Makefile targets. **Grep for the old command; do not trust one test over one function.**

## Three flags that fail late

**`--scopes=cloud-platform`** — required if your startup script reads a secret. Mine pulls a Hugging Face token from Secret Manager at boot. Without the scope the VM boots fine and then spins for 30 minutes before giving up, so the symptom is a slow startup followed by what looks like a token problem.

**`--boot-disk-size`** — the image default is 10 GB, which will not hold a vLLM TPU image. Queued resources gave you a far larger disk by default, so this is easy to miss. Fails after a clean boot, mid-pull.

**`--maintenance-policy=TERMINATE`** — required, because a TPU instance cannot live-migrate.

Two more worth knowing, both from the [provisioning models](https://docs.cloud.google.com/compute/docs/instances/provisioning-models) page. Flex-start instances run for a minimum of 10 minutes and **a maximum of seven days**, so set `--max-run-duration` explicitly rather than discovering the boundary. And you cannot suspend one — a standalone flex-start instance can be stopped, but suspend and recreate are unavailable, and anything created through a MIG resize request cannot be stopped either. Keep state you care about on a separate disk or in GCS.

## Troubleshooting quick reference

```
Symptom                                                 Likely cause                                           Check
------------------------------------------------------  -----------------------------------------------------  -------------------------------------------------------------------------------
PENDING for hours                                       quota or capacity — identical from outside             fire a SPOT create at the same zone; stockout means capacity, and usually it is
This user agent is not allowed to use the machine type  that generation has no Compute Engine path             use the Cloud TPU API for that chip
RUNNING but nothing serves                              startup script died                                    read the startup log; curl the port
docker: command not found                               the CE image ships no Docker                           install docker.io before pulling
Out of disk mid-pull                                    10 GB image default                                    --boot-disk-size
Secret access hangs 30 min                              missing --scopes=cloud-platform                        recreate with the scope
VM vanished after 24h                                   flex-start default duration                            set --max-run-duration explicitly
tpu-vm list returns nothing                             wrong API for a ct6e-* instance                        gcloud compute instances list
SSH says not found                                      wrong SSH surface                                      gcloud compute ssh, not tpus tpu-vm ssh
Quota looks fine but nothing works                      reading regions describe, which shows v5 metrics only  Cloud Quotas API, by metric name
decreases effective quota unsafely                      requesting less than you hold                          read the current value first
gcloud alpha reported missing but works                 apt install, component manager disabled by design      ignore it; alpha ships in the base package
```

## The short version

Check whether your chip has a Compute Engine path at all, by trying a create rather than reading the catalog. Check your quota through the Cloud Quotas API rather than `regions describe`, and read it as an intersection with machine-type availability. Translate the flags. Then rewrite discovery and SSH instead of filtering them, and grep for the old commands afterwards.

And assume nothing fails loudly. A stuck request, a dead boot, a blind discovery helper and a missing SSH surface all present as silence or as a cheerful `RUNNING`. The flag mapping is the part gcloud checks for you; everything in this article is the part it does not.

There is nothing as constant as change. TPU7x is already Compute Engine only, so this will not be the last migration any of us does — but the next one should be cheaper, because the hard part was never the flags.
