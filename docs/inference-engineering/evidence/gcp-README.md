# One Strands agent, every GCP inference path

The Google Cloud counterpart of [aws-inference-strands](https://github.com/xbill9/aws-inference-strands): the same Strands agent, the same questions and the same grading, on seven ways to run a model on Google Cloud, from a hosted API to a TPU you start yourself.

| Backend | What it is | Model |
|---|---|---|
| `gemini` | Gemini on Vertex AI (now Gemini Enterprise Agent Platform), per token | `gemini-3.5-flash-lite` |
| `maas` | Open model as a service on Vertex AI, per token, OpenAI-compatible | `google/gemma-4-26b-a4b-it-maas` |
| `vertex` | Vertex AI endpoint running your vLLM container, `g2-standard-4` + L4 | Gemma 4 E2B repack |
| `cloudrun` | Cloud Run service with an L4, scales to zero | Gemma 4 E2B repack |
| `gke` | GKE Autopilot Deployment on an L4 node | Gemma 4 E2B repack |
| `gce` | Compute Engine `g2-standard-4` VM, vLLM in Docker | Gemma 4 E2B repack |
| `tpu` | Cloud TPU v5e-1 VM, vLLM TPU | `google/gemma-4-E2B-it`, bf16 |

The four L4 backends serve `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4` with `vllm/vllm-openai:v0.30.0`, the checkpoint and image of the AWS run. The TPU runs `vllm/vllm-tpu:v0.30.0`. Every self-hosted backend runs vLLM with `--enable-auto-tool-choice --tool-call-parser gemma4`, so every backend here can drive the agent, the TPU included.

| Script | What it does |
|---|---|
| `demo1_counting.py` | The same tool-using agent and question on each backend, graded on the answer **and** on the filter the model sent |
| `demo2_orchestrator.py` | A Gemini agent that calls every backend as a tool and quotes a scoreboard computed in code |
| `bench_chat.py` | The same 100-token prompt on every backend, timed from the client |
| `server_side.py` | Decode speed read from each vLLM server's own `/metrics`, network left out |
| `parallel.py` | Aggregate throughput with 1, 4 and 16 requests in flight |
| `cost.py` | USD per hour and per million output tokens, from Cloud Billing Catalog list prices and measured throughput |
| `results/summarize.py` | The demo 1 and bench tables, computed from a results directory |
| `mcp_server.py` | An MCP server over the same backends: list, health check, logs, ask, launch, tear down, and demo 1's `count_ids` engine |

[docs/OPTIONS.md](docs/OPTIONS.md) reviews every GCP serving option, hosted to self-run, with Google's documentation for each fact.

---

#### Results, 2026-10-12

Every number below is in a file in [results/2026-10-12](results/2026-10-12) and was computed by code from those files.

**Demo 1.** Eleven ids, "how many are 10 or more?", answer 8, temperature 0, 20 runs per backend and tool set. With `count_ids(op, value)` the engine counts; with `list_ids()` the model counts.

| Backend | Engine: correct | Right filter | Model counts: correct | Median s per run |
|---|---|---|---|---|
| gemini | 20/20 | 20/20 | 20/20 | 2.95 |
| maas | 19/20 | 19/20 | 0/20 | 13.23 |
| vertex | 20/20 | 20/20 | 20/20 | 0.92 |
| cloudrun | 20/20 | 20/20 | 20/20 | 0.52 |
| gke | 20/20 | 20/20 | 20/20 | 0.70 |
| gce | 20/20 | 20/20 | 20/20 | 0.89 |
| tpu | 20/20 | 20/20 | 20/20 | 1.65 |

Gemma 4 26B on MaaS answered **9** in 19 of 20 runs when it counted the rows itself, and 8 in 19 of 20 when the engine counted; its other run in each set was a 503. Every backend that called the tool sent `id >= 10`.

**Demo 2.** A Gemini agent asked all seven backends "What is the capital of France?" in each of three runs and quoted the scoreboard as returned, with no summary of its own. Every backend answered Paris; MaaS refused one call with a 429. Fastest per run: TPU twice and GKE once, at medians of 0.210 s and 0.220 s; Gemini answered in 1.462 s ([demo2-summary.txt](results/2026-10-12/demo2-summary.txt)).

**Decode speed, one request at a time.** The client figure includes the network path to each backend; the server figure is vLLM's own count of the time between tokens.

| Backend | Client median tok/s | Server decode tok/s | Path from the client |
|---|---|---|---|
| cloudrun | 119.3 | 141.9 | HTTPS, us-east4 |
| tpu | 105.3 | 144.1 | SSH forward, us-west4 |
| gke | 104.7 | 139.9 | kubectl port-forward, us-central1 |
| gce | 94.3 | 143.1 | IAP tunnel, northamerica-northeast1 |
| gemini | 93.3 | - | Vertex AI API, global |
| vertex | 90.2 | - | `:rawPredict`, us-central1 |
| maas | 14.2 | - | Vertex AI API, global; 1 of 5 requests refused with 429 |

On the server, the three L4s decode within 2.3% of each other, and the TPU v5e decodes the bf16 checkpoint at the rate the L4s decode the 4-bit one. The spread on the client side is the network.

**Cost.** List prices from the Cloud Billing Catalog API, fetched 2026-10-12; USD per million output tokens for a busy hour ([cost.txt](results/2026-10-12/cost.txt)). The 16-in-flight column is three rounds per level, timed from the client ([parallel.txt](results/2026-10-12/parallel.txt)).

| Backend | USD per hour | One request at a time | 16 in flight |
|---|---|---|---|
| tpu (v5e, flex-start) | 0.60 | 1.58 | 0.138 |
| gke | 0.96 | 2.55 | 0.203 |
| gce | 0.78 | 2.29 | 0.252 |
| cloudrun | 1.05 while up, 0 at zero | 2.44 | not run |
| vertex | 0.92 | 2.83 | not run |
| gemini | per token | 2.50 | 2.50 |
| maas | per token | 0.60 | 0.60 |

GKE Autopilot placed a 4 vCPU, 16 GiB Pod on a `g2-standard-8` node, and is priced as that node plus the Autopilot premium and cluster fee. The TPU on-demand rate is USD 1.20 per hour.

**Getting capacity.** L4 quota was 16 GPUs in every US region. 13 zones in 5 US regions refused `g2-standard-4` with `STOCKOUT` before Montreal launched one; an hour later both Montreal zones were full and a second VM landed in Toronto. TPU v6e-1 was refused in every zone the project can use, on-demand and Spot, and a v6e flex-start request sat queued until it was withdrawn. A v5e-1 flex-start request in us-west4-a got a node in 22 seconds ([capacity.txt](results/2026-10-12/capacity.txt)).

**Launch to ready** ([launch-to-ready.txt](results/2026-10-12/launch-to-ready.txt)): Cloud Run deploy 6.6 min; Compute Engine VM 9.6 min; GKE Autopilot cluster 8.8 min, then 7.7 min for the Deployment; Vertex AI deploy-model 24.4 min, after an 18.9 min first upload through the Docker Hub proxy; TPU v5e 13.3 min from start script to ready, 6.5 min of it XLA compilation.

**Cloud Run from zero** ([coldstart.txt](results/2026-10-12/coldstart.txt), [coldstart-all.txt](results/2026-10-12/coldstart-all.txt)): the request that wakes a scaled-to-zero L4 service is held and answered; requests arriving during the start-up get an immediate 429 when `max-instances` is 1.

| Variant | Cold starts, s | torch.compile, s | Server decode tok/s |
|---|---|---|---|
| default (`stage/up.sh`, startup probe every 60 s) | 361.8, 483.2 | 116.1, 118.6 | 141.9 |
| `--enforce-eager`, probe every 10 s | 222.1, 202.4 | none | 17.2, 17.8 |
| caches on a Cloud Storage volume, warm, probe every 10 s | 240.1 | 6.2 | 138.6 |

A warm compile cache keeps full speed and takes about four minutes from zero; eager mode starts in about the same time and decodes eight times slower. The cache variant's first boot, which writes the caches through Cloud Storage FUSE, ran past Cloud Run's 15-minute startup-probe limit and had to be deployed twice. The model download from Hugging Face took 77.9 s in one default cold start and 185.8 s in the other.

---

#### Bring the backends up

The talk's slides are [on Google Slides](https://docs.google.com/presentation/d/11qC6IwhqEEtOFqmQH3l4f0XGUbokoQi9NkeMN5F4K_Q/edit); `python3 stage/present.py` steps through them with the live command for each slide.

```bash
stage/up.sh 2>&1 | tee stage/up.log      # gce vertex gke run tpu; starts billing
stage/forward.sh && source stage/demo.env # tunnels and port-forwards, writes the variables below
python3 stage/check.py --wait 1800        # one question and one tool call per backend; READY
python3 stage/ask.py cloudrun             # one backend, one question, timed
stage/down.sh                             # delete everything and sweep for leftovers
```

`up.sh` tries each zone in turn when a launch gets no capacity, and falls back to a flex-start TPU request. Nothing listens on a public port: Compute Engine through an IAP tunnel, GKE through `kubectl port-forward`, the TPU through an SSH forward, Cloud Run with `--no-allow-unauthenticated` and an identity token.

Two settings this run depends on:

- `vllm/vllm-openai:v0.30.0`. The `-cu129` build of the same release dies at import ("operator torchvision::nms does not exist") on every platform here.
- `VLLM_ENABLE_CUDA_COMPATIBILITY=0`. Cloud Run and GKE Autopilot ship NVIDIA driver 580, which runs CUDA 13 natively; with the image's forward-compatibility libraries switched on, vLLM fails with CUDA error 803.

#### Install

```bash
pip install 'strands-agents[openai,gemini]' google-genai mcp
```

#### Configure

Each backend is turned on by its variable; unset means skipped. `BACKENDS=gemini,gce` limits a run to some of them. `stage/forward.sh` writes all of them.

| Variable | Backend | Example |
|---|---|---|
| `GEMINI_MODEL`, `GOOGLE_CLOUD_PROJECT` | Gemini (always on) | `gemini-3.5-flash-lite` |
| `MAAS_MODEL` | Gemma MaaS | `google/gemma-4-26b-a4b-it-maas` |
| `VERTEX_URL` | Vertex AI endpoint | `https://us-central1-aiplatform.googleapis.com/v1/projects/P/locations/us-central1/endpoints/ID` |
| `RUN_URL` | Cloud Run | `https://strands-demo-run-....run.app/v1` |
| `GKE_URL` | GKE | `http://localhost:8003/v1` |
| `GCE_URL` | Compute Engine | `http://localhost:8001/v1` |
| `TPU_URL` | Cloud TPU | `http://localhost:8005/v1` |

#### MCP server

`mcp_server.py` is the Google Cloud port of the [aws-inference-strands](https://github.com/xbill9/aws-inference-strands) server: the same tools and annotations, with `gcloud` and `kubectl` in place of boto3 and SSM. `.mcp.json` registers it for Claude Code as `strands-demo`; any MCP client can start it with `python3 mcp_server.py`.

| Tool | Kind | What it does |
|---|---|---|
| `list_backends` | read-only | the Compute Engine VM, TPU VMs and queued resources, Cloud Run services, the GKE cluster and its `vllm` Deployment, and the Vertex AI endpoint, found by the `managed-by=strands-demo` label or name, and how many are billing |
| `backend_health` | read-only | asks one backend for the capital of France and passes only on Paris |
| `backend_logs` | read-only | the vLLM log of one backend |
| `ask_backend` | read-only | one timed chat request, as `stage/ask.py`; uses the variables in `stage/demo.env` |
| `count_ids` | read-only | demo 1's engine: exact count, minimum and maximum for `id <op> value` |
| `launch_log` | read-only | tails `stage/up.log` or `stage/down.log` |
| `launch_backends` | write | runs `stage/up.sh` in the background; billing starts |
| `teardown` | destructive | runs `stage/down.sh` in the background |

Each backend is reached its own way, and health checks and logs need no forward open:

| Backend | `backend_health` | `backend_logs` |
|---|---|---|
| gce | on the VM, `gcloud compute ssh --tunnel-through-iap` | `docker logs vllm`, else the startup-script journal |
| tpu | on the VM, `gcloud compute tpus tpu-vm ssh` | `docker logs vllm`, else the startup-script journal |
| gke | in the Pod, `kubectl exec deploy/vllm` | `kubectl logs deploy/vllm`, else the Pod events |
| cloudrun | HTTPS with an identity token | Cloud Logging, `cloud_run_revision` |
| vertex | `:rawPredict` with an access token | Cloud Logging, `aiplatform.googleapis.com/Endpoint` |
| gemini, maas | the Vertex AI API | none |

Launch and teardown run the stage scripts, so both routes start and stop the same resources. With `--tools mcp`, demo 1 offers the agent only `count_ids` from this server and grades the calls it reads back from the agent's messages, so it is graded exactly as `--tools engine`.

#### How the grading works

Demo 1 asks how many of eleven ids are 10 or more; the answer is 8. With `--tools engine` the agent gets `count_ids(op, value)`, which returns the exact count, minimum and maximum. A run counts as correct when the final answer states 8, and as the **right filter** only when one of its `count_ids` calls selects the same ids as `id >= 10`. Before reading the count, the grader drops restatements of the threshold ("8 ids are 10 or more"), so the number left is the model's answer.

Demo 2 times every request in code and reads each backend's own completion token count. `scoreboard()` sorts and formats the table; the agent is told to quote it as is, and the harness prints which backends the agent never asked.
