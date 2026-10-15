# First provision through GKE — v6e-1, europe-west4-a, 2026-08-25

**What this records: a provisioning path, not a serving benchmark.** There is no
`benchmarks/reports/*.json` here on purpose — nothing was swept, so nothing should be counted by
`benchmarks/rollup.py`. The one serving number below is a smoke test, not a measurement.

## What was created

| | |
| --- | --- |
| Cluster | `gke-vllm-v6e1-2b`, zonal `europe-west4-a`, rapid channel → **1.36.3-gke.1537000** |
| System pool | 1× `e2-standard-4` (kube-dns, metrics-server) |
| TPU pool | `tpu-v6e-1`, 1× `ct6e-standard-1t`, on-demand, 200 GB disk |
| Node | `gke-tpu-cfc04f31-8h14` — 43.8 CPU / 170 GiB allocatable, **`google.com/tpu: 1`**, taint `google.com/tpu=present:NoSchedule` |
| Node labels | `gke-tpu-accelerator=tpu-v6e-slice`, `gke-tpu-topology=1x1`, `machine-family=ct6e` |
| Workload | Deployment + Service `vllm-gemma4`, `vllm/vllm-tpu:nightly` → vLLM `0.26.1rc1.dev994+gd626108b1` |
| Endpoint | LoadBalancer `34.91.103.13:8000` |

## Timings

Wall clock, single run, no repeats — treat as an order of magnitude, not a measurement:

| Phase | Time | Source |
| --- | --- | --- |
| `clusters create` (incl. system node) | ~12 min | the system node was 12m old when the TPU node first registered |
| `node-pools create` → node Ready | ~6 min | script wall clock; node was 20 s old at first `get nodes` |
| pod scheduled → container running | 55 s | 13:04:09 → 13:05:04 (image pull) |
| container running → `/health` Ready | **10 min 09 s** | 13:05:04 → 13:15:13 (weight load + XLA precompile) |
| **nothing → first token** | **~28 min** | sum of the above |

First completion: 0.40 s for 10 prompt / 15 completion tokens on `/v1/chat/completions`.

## What this cost, in corrections

1. **`--tpu-topology=1x1` is refused for a single-host slice.** The first `node-pools create` failed with
   `TPU topology can't be specified with single-host TPU slice pool`. GKE labels the node `1x1` anyway, so
   the value is a valid *selector* and an invalid *create flag*. Now split in `tpu.env` as `TPU_TOPOLOGY`
   (selector) vs `GKE_TPU_TOPOLOGY` (multi-host create flag, unset).
2. **`gcloud secrets versions access` without `--project`** resolved to the workstation's default project —
   an expired qwiklabs lab — and failed with a permission error naming a project this rig never mentions.
   Only the MCP tool hit it; the shell path always passed `--project`.
3. **A dollar sign in a template comment** broke `string.Template` rendering while envsubst rendered it
   happily, so the shell path worked and the MCP path did not.
4. **A Ready TPU node can advertise `google.com/tpu: 0`** for a window after joining, before the device
   plugin registers. A pod scheduled then fails `Insufficient google.com/tpu`, which reads like a quota
   problem and is not one. Observed on the second (spot) pool.

## MCP tool round trip

Verified the same day, after the port off `gcloud compute instances`: `provision_gke_tpu` (idempotent),
`create_tpu_node_pool` → a second `tpu-v6e-probe` spot pool, `list_tpu_node_pools`, `get_system_status`,
`destroy_tpu_node_pool`. `deploy_vllm` re-applied the manifest the shell path had already applied and
Kubernetes reported `deployment.apps/vllm-gemma4 unchanged` — which is the check that the two render paths
have not drifted.

## Not measured

Throughput, latency under concurrency, KV headroom, and any comparison against `tpu-vllm-v6e1-2b` or
`gce-vllm-v6e1-2b`. The provisioning-path comparison this rig exists for needs the other two timed the same
way, and neither has been.

---

## Second run: the whole flow, MCP tools only

Re-run the same day to validate the ported tools, from a deliberate cold start — `destroy_gke_cluster`
first, so nothing survived between the two runs. Every step is a `server.py` tool call; no shell scripts,
no hand-run `kubectl`.

| Phase | Tool | Time |
| --- | --- | ---: |
| Teardown | `destroy_gke_cluster` | 384 s |
| Cluster + TPU node pool | `provision_gke_tpu` | 532 s |
| Apply Deployment + Service | `deploy_vllm` | 6 s |
| Pod → `/v1/models` answering | (poll) | 669 s |
| **Cold start total** | | **1591 s — 26 m 31 s** |

Consistent with the first run's ~28 minutes. The node came up under a **new name**
(`gke-tpu-bcdb7fb0-bw0t`) — you do not name a node pool's nodes — while the LoadBalancer came back on the
**same external IP**, `34.91.103.13`.

Health check 0.77 s. Then `run_vllm_benchmark`, which on this rig runs `vllm bench serve` **inside the
serving pod** via `kubectl exec` rather than in a second container over SSH:

| Concurrency | Output tok/s | TTFT p99 | TPOT mean | $/M output tokens |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 199 | 17 ms | 5.0 ms | $4.14 |
| 8 | 1180 | 48 ms | 6.7 ms | $0.70 |
| 32 | 1508 | 355 ms | — | $0.55 |

1024 in / 128 out, one run per point. Filed as `benchmarks/reports/2026-08-25-gemma4-e2b-v6e1.json`
(schema 1.1) — **the rig's first real measurement**. Chip cost only, at the europe-west4 on-demand list
rate of $2.97/chip-h read live from the Billing Catalog; spot lists at $1.782 and flex-start at $1.35.

**One more correction, found by running the cost tool:** it still told the twin rig's story — "flex-start
self-terminates at `--max-run-duration`, capping the bill." A node pool has no run bound, so that was a
confident wrong statement about money. Fixed, and pinned by
`test_cost_never_claims_a_node_pool_stops_itself`.

## Third run: does the cheap capacity actually work?

Only on-demand and spot pools had been created here, so the $1.35/chip-h flex-start rate was a catalog
number rather than a tested path. Tested the same day:

- `create_tpu_node_pool(provisioning_model="flex-start")` succeeded and the pool came up at **zero nodes** —
  on GKE flex-start is an autoscaling shape (`--num-nodes=0 --total-max-nodes=N`), not a fixed pool. An idle
  flex-start pool costs nothing.
- Scaling the Deployment to 2 replicas left the second Pod `Pending`. **4 m 18 s later DWS granted a chip**,
  node `gke-tpu-49b36129-bxrn` joined, and ~10 min after that the replica was serving.
- Torn down through `destroy_tpu_node_pool` afterwards; the on-demand pool and the endpoint were untouched
  throughout.

**Chip rates, europe-west4, read live from the Billing Catalog:** on-demand $2.9700, spot $1.7820,
flex-start $1.3500 per chip-hour. Fixed overhead that has no equivalent on the two sibling rigs:
`e2-standard-4` system node $0.1475/h (E2 core ×4 + E2 RAM ×16 GB) + `Zonal Kubernetes Clusters` $0.1000/h
= **$0.2475/h**, i.e. $181/month for a cluster with no TPU pool at all.

## Fourth run: the 2-D sweep

12 points through `run_vllm_benchmark`: concurrency 1–64 at 1024 input, then context 512–16,384 at
concurrency 8, output 128 throughout. Filed in `benchmarks/reports/2026-08-25-gemma4-e2b-v6e1.json`.

**The concurrency curve is not monotonic and the dip reproduces.** 1,744 tok/s at c=16, **1,675 at c=32**,
2,134 at c=64. Three repeats at each point: c=16 gave 1777.0/1780.4/1778.9, c=32 gave 1694.4/1696.2/1694.2,
c=64 gave 2351.0/2353.0/2351.4 — **0.1–0.2% run-to-run spread**, so c=32 really is ~4% slower than c=16.

Hypothesis, **unverified**: TPU static-shape padding, the c=32 batch straddling a compiled bucket. Not
checked against the compiled shape list, and recorded as unverified in the report. The measurement stands
on its own; the explanation does not.

Context: throughput falls 1,185 → 487 tok/s from 512 to 16,384 input tokens, p99 TTFT 35 ms → 878 ms. A 32x
longer prompt costs 2.4x per output token. KV is not the constraint — ~19.8 GiB free after weights leaves
16K at c=8 well inside budget.

Cost per million output tokens, all-in: $4.48 at c=1 on-demand, $0.21 at c=64 flex-start — a factor of 21 on
identical hardware.
