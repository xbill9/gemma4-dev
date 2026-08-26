# The GKE path — one cluster, one node, one v6e chip

One zonal cluster, one single-host `ct6e-standard-1t` node pool, one node, one chip, one vLLM Pod.

**Two front ends, one path.** These scripts and the MCP tools in `server.py` do the same thing with the same
commands: they render the same manifest template, read the same `tpu.env`, and use the same four
provisioning-model names. Use the scripts when you want to read or paste the exact gcloud invocation; use
the tools when an agent is driving. Running `deploy_vllm` after `make gke-deploy` reports `unchanged`, which
is the check that the two have not drifted.

```
make gke-preflight   # kubectl + gke-gcloud-auth-plugin + envsubst + live gcloud creds
make gke-up          # cluster + TPU node pool          (~18 min cold, idempotent)
make gke-deploy      # credentials + HF secret + manifest
make gke-status      # node, pod, service
make gke-logs        # follow the vLLM pod
make gke-endpoint    # LoadBalancer external IP
make gke-query       # chat-completions smoke test
make gke-down-pool   # release the chip, keep the cluster
make gke-down        # delete the cluster
```

MCP equivalents: `provision_gke_tpu`, `deploy_vllm`, `get_system_status`, `get_vllm_pod_logs`,
`get_vllm_endpoint`, `query_queued_gemma4`, `destroy_tpu_node_pool`, `destroy_gke_cluster`, plus `find_tpu`
for a zone sweep and `get_tpu_node_diagnostics` when a pod will not schedule.

`GKE_NODE_PROVISIONING` takes `on-demand` (default), `spot`, `flex-start` or `reservation-bound` — the same
four names `server.py` uses, deliberately.

## What actually gets created

| Object | Why |
| --- | --- |
| Zonal cluster `$GKE_CLUSTER_NAME`, rapid channel | TPU v6e needs a recent control plane; the rapid channel avoids pinning a version string that goes stale. Zonal, because a TPU node pool has to sit in the zone that has the chips |
| Default pool, 1× `e2-standard-4` | kube-dns and metrics-server have to run somewhere. Keeping them off the TPU node means system pods can never be the reason a chip stays alive |
| TPU pool `$GKE_NODE_POOL`, 1× `ct6e-standard-1t`, **no** `--tpu-topology` | The chip. The machine type alone makes it a TPU pool; `--tpu-topology` is refused for a single-host slice (see below) |
| Secret `hf-token` | Copied from Secret Manager at deploy time, so the committed manifest holds no token |
| Deployment `vllm-gemma4` | `vllm/vllm-tpu:nightly`, `google.com/tpu: 1`, the same serve flags as the two sibling rigs |
| Service `vllm-gemma4` | `LoadBalancer` on `:8000` — the same network path as the GCE twin, which is what keeps a benchmark comparable |

## The parts that differ from the GCE twin, and bite

- **`google.com/tpu: 1` plus the two node-selector labels are what bind the pod to the chip.**
  Drop the selectors and the pod schedules onto the `e2-standard-4` system node and fails there,
  looking like a vLLM problem rather than a placement one.
- **`RUNNING` is weaker here than it was even on Compute Engine.** A node is Ready as soon as the
  kubelet registers; the pod then pulls a multi-GB image and compiles for TPU. `make gke-status`
  showing `Running` is not readiness — that is what the startup probe on `/health` is for, and
  `make gke-logs` is the only honest progress indicator.
- **The endpoint is a Service, not a `natIP`.** Nothing in `server.py` can find it: those tools
  resolve a Compute Engine instance's `networkInterfaces[].accessConfigs[].natIP`, and a GKE node's
  IP is not where the model is listening. Use `make gke-endpoint` or `make gke-port-forward`.
- **The LoadBalancer is unauthenticated and public**, exactly like the twin's `:8000`. That is
  deliberate for comparability. Set `GKE_SERVICE_TYPE=ClusterIP` in `tpu.env` and use
  `make gke-port-forward` if it is not wanted.
- **Quota is the Compute Engine pool, not the TPU API's.** GKE node pools consume
  `TPUS-PER-TPU-FAMILY-per-project-region` (on-demand) and `PREEMPTIBLE-TPU-V6E-per-project-region`
  (spot, and flex-start first) — the same ids the `gce` rigs spend and the same regional
  misalignment documented in `../CLAUDE.md`. Holding TPU-API quota buys nothing here.
- **Three things bill, not one:** the v6e chip, the `e2-standard-4` system node, and the cluster
  management fee. `make gke-down-pool` stops the expensive one and keeps the rest.

## The node is a VM you did not create, and cannot delete

The node pool is a **managed instance group**; the node inside it is an ordinary Compute Engine VM with the
same `ct6e-standard-1t` machine type the `gce` rig passes to `instances create`. It shows up in
`gcloud compute instances list` under a name GKE chose (`gke-tpu-cfc04f31-8h14`).

**`gcloud compute instances delete` on that node is not teardown.** It succeeds, and the MIG recreates the
node minutes later under a new name, still billing. Delete the *pool*: `make gke-down-pool`, or
`destroy_tpu_node_pool`. Likewise, do not hand-fix a node — it is replaced on upgrade and repair, and
anything written to its disk goes with it.

## If the pod can't see the chip

In order of likelihood: the node-selector labels don't match the pool (`kubectl get nodes
--show-labels`); the pool was created without `--tpu-topology`, so it is an ordinary VM pool; the
`google.com/tpu` limit is missing, so the device plugin never attaches the chip. Only after those
three is it worth trying `securityContext.privileged: true` — on GKE the device plugin is supposed
to make that unnecessary, which is why the manifest does not set it.

## `--tpu-topology` is a multi-host flag, and 1x1 is not a small version of it

The first `node-pools create` from these scripts (2026-08-25) passed `--tpu-topology=1x1` and was
refused at the API:

```
ResponseError: code=400, message=TPU topology can't be specified with single-host TPU slice pool;
please remove the tpu_topology from the node pool creation request
```

`ct6e-standard-1t` at one node **is** the slice, so there is no topology to describe — the flag
exists for multi-host slices, where several nodes are wired into one. What makes this trap quiet is
that **GKE then labels the node `cloud.google.com/gke-tpu-topology=1x1` anyway**, so the value is
real as a *selector* and refused as a *create flag*. `tpu.env` keeps the two apart: `TPU_TOPOLOGY`
is the pod selector, `GKE_TPU_TOPOLOGY` (unset here) is the multi-host create flag.

## Status — proven on hardware 2026-08-25

First run end to end, `europe-west4-a`, on-demand:

| | |
| --- | --- |
| Cluster | `gke-vllm-v6e1-2b`, 1.36.3-gke.1537000, rapid channel |
| TPU node | `gke-tpu-*`, `ct6e-standard-1t`, 43.8 CPU / 170 GiB / **`google.com/tpu: 1`** allocatable, taint `google.com/tpu=present:NoSchedule` |
| Labels on it | `gke-tpu-accelerator=tpu-v6e-slice`, `gke-tpu-topology=1x1`, `machine-family=ct6e` — the manifest's selectors matched with no change |
| Pod | `vllm-gemma4`, Ready ~10 min after apply (image pull → weight load → shape precompile) |
| Endpoint | LoadBalancer `:8000`, chat-completions answered in 0.40 s (10 prompt / 15 completion tokens) |
| vLLM | `0.26.1rc1.dev994+gd626108b1` from `vllm/vllm-tpu:nightly` |

Nothing needed `privileged`, and no quota increase was requested — the on-demand CT6E family quota
in europe-west4 covered it.

`server.py` was ported off Compute Engine the same day, so the rig and its name now agree: the MCP tool
round trip (create pool → list → status → destroy) was verified on 2026-08-25, and `deploy_vllm` re-applying
the shell path's manifest returned `unchanged`.
