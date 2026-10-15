# Second Run: Six AWS Inference Backends From One Script

2026-10-05, 16:32–17:04 UTC. All five self-hosted backends were brought up with `stage/up.sh`, checked with `stage/check.py`, driven through every demo at stage size and at full size, and torn down with `stage/down.sh`. This is the second complete run; the first is in `results/2026-10-05/`. Every number below is in a file in this directory, and every table was computed by code from those files.

---

#### Setup

| Backend | Where it ran | Hardware | Serving | Model |
|---|---|---|---|---|
| bedrock | us-east-1 | managed | Amazon Bedrock Converse | `us.amazon.nova-micro-v1:0` |
| sagemaker | us-east-2 | `ml.g6.xlarge`, NVIDIA L4 | AWS vLLM 0.30.0 container (`vllm@sha256:f50ad6bf…`) | `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4` |
| g6 | us-east-2a | g6.xlarge, NVIDIA L4, driver 595.91.07 | `vllm/vllm-openai:v0.30.0` | the same repack |
| g5g | us-east-1a | g5g.2xlarge, Graviton2 + NVIDIA T4G, driver 595.71.05 | vLLM 0.27.2rc0 built for sm_75, `ami-0b44b90b3d02430ee` | `google/gemma-4-E2B-it`, float16 |
| inf2 | **us-west-2a** | inf2.xlarge, Inferentia2 | `xbill9/gemma4-optb:slim` (`sha256:6cf0ae7e…`) | Gemma 4 E2B compiled for Neuron |
| trn1 | us-east-2c | trn1.2xlarge, Trainium | the same image, same digest | the same build |

Client: strands-agents 1.55.0, openai 2.54.0, boto3 1.43.90. The EC2 backends were reached through SSM port forwards; SageMaker and Bedrock through their AWS APIs. Sources: `instances.txt`, `versions.txt`, `versions-backends.txt`, `sagemaker-status.json`.

---

#### Result 1 — Quota Did Not Buy Capacity

`up.sh` tries each default subnet in a region, then the next region. inf2 had quota in us-east-2 and launched nowhere there:

```
strands-demo-inf2: no capacity in us-east-2 (subnet-0b0f0d29473b0a624), trying the next zone
strands-demo-inf2: no capacity in us-east-2 (subnet-04bce993deca43788), trying the next zone
strands-demo-inf2: no capacity in us-east-2 (subnet-0880f3d5ac599127d), trying the next zone
strands-demo-inf2: no capacity in us-east-1 (subnet-061a363014b302012), trying the next zone
strands-demo-inf2: no capacity in us-east-1 (subnet-09e0f13c0a5b43092), trying the next zone
strands-demo-inf2: launched i-00eae995ffe871f80	us-west-2a
```

trn1 was refused in two us-east-2 zones and launched in the third. g6 and g5g launched in their first zone. Source: `up.log`.

---

#### Result 2 — Launch to First Correct Answer

Launch is the EC2 `LaunchTime` or the SageMaker `CreationTime`. Ready is the first `stage/check.py` pass: a correct answer to one question and, on tool-capable backends, one demo 1 run with a tool call. Checks ran about once a minute, so each EC2 figure can be high by up to one check round.

| Backend | Launch → ready | What took the time |
|---|---|---|
| sagemaker | 8.0 min to InService | endpoint creation |
| trn1 | 12.8 min | image pull, loading the compiled model onto the NeuronCores |
| g6 | 13.9 min | image pull, weight download, CUDA graph capture |
| inf2 | 17.9 min | the same as trn1, in a region the image had never been pulled to |
| g5g | 23.7 min | weights read from a fresh EBS volume: `Loading weights took 530.83 seconds` |

The first run measured g6 at 14.0 min, inf2 at 14.3 min, SageMaker at 8.2 min and g5g's weight load at 520.97 s. Source: `launch-to-ready.txt`, `check.txt`.

---

#### Result 3 — Demo 1: The Engine Counts, the Filter Is Checked

Eleven ids, "how many are 10 or more?", true answer 8, temperature 0, 20 runs per backend. With `count_ids(op, value)` the engine returns the exact count; a run has the right filter when the predicate it sent selects the same ids as `id >= 10`.

| Backend | Correct | Right filter | Filters sent | Median s |
|---|---|---|---|---|
| bedrock | 20/20 | 20/20 | `id >= 10` | 1.05 |
| g6 | 20/20 | 20/20 | `id >= 10` | 0.4 |
| g5g | 20/20 | 20/20 | `id >= 10` | 2.3 |
| sagemaker | 20/20 | 20/20 | `id >= 10` | 0.58 |

With `list_ids()` the model counts the rows itself. All four were correct in 20 of 20, with medians of 1.52 s, 0.89 s, 1.81 s and 1.03 s. That is about half a second slower than the engine runs on Bedrock, g6 and SageMaker, and half a second faster on the g5g. Eleven rows is a small table; a correct count here says nothing about a large one.

The first run had the same scores and the same single filter on every backend it ran. Source: `demo1-engine.txt`, `demo1-rows.txt`, and the matching `.json`.

---

#### Result 4 — Demo 2: Every Backend Asked, Every Time

A Bedrock agent gets one `ask_<backend>` tool per backend and a `scoreboard()` that sorts in code. Five runs, one-word question.

| Backend | Median s | Min | Max |
|---|---|---|---|
| g6 | 0.153 | 0.148 | 0.164 |
| g5g | 0.187 | 0.177 | 0.200 |
| trn1 | 0.192 | 0.155 | 0.220 |
| inf2 | 0.291 | 0.254 | 0.295 |
| sagemaker | 0.340 | 0.334 | 0.388 |
| bedrock | 0.623 | 0.543 | 0.695 |

- All 30 calls answered Paris, with no errors, and the agent asked all six backends in all five runs.
- g6 was fastest in four runs and trn1 in one (run 4, 0.155 s against g6's 0.164 s). g5g and trn1 are 5 ms apart at the median, so their order can swap from run to run.
- Two runs closed with a sentence naming the fastest backend; both named g6, matching the table. In the first run, one closing sentence named Trainium while its table put g6 first. The table comes from code; the sentence comes from the model, and it is the part to check.
- For a two-token answer the order follows the round trip: a port-forwarded EC2 server, then the SageMaker invoke API, then Bedrock. The g5g sits second here and fourth in Result 5, where decode speed dominates.

Source: `demo2-run1.txt` … `demo2-run5.txt`, `demo2-summary.txt`.

---

#### Result 5 — Same Prompt, Every Backend, Repeated

"Write a short paragraph about the ocean.", `max_tokens=100`, temperature 0, 10 repeats after one warm-up request. Tokens per second uses each backend's own completion count and the client's wall clock, so it includes the network and the prefill.

| Backend | Median s | Median tok/s | tok/s range | Tokens |
|---|---|---|---|---|
| g6 | 0.635 | 132.5 | 130.2–132.8 | 84 |
| bedrock | 0.863 | 115.9 | 103.7–133.9 | 100 |
| sagemaker | 0.798 | 105.2 | 104.1–107.4 | 84 |
| trn1 | 2.550 | 36.9 | 36.7–37.1 | 94 |
| inf2 | 2.569 | 36.6 | 35.9–36.6 | 94 |
| g5g | 2.711 | 35.8 | 35.2–37.2 | 97 |

Against the first run, computed in `comparison.txt`:

| Backend | First run tok/s | This run | Change |
|---|---|---|---|
| bedrock | 111.9 | 115.85 | +3.5% |
| g6 | 131.5 | 132.45 | +0.7% |
| g5g | 37.2 | 35.8 | −3.8% |
| sagemaker | 106.3 | 105.2 | −1.0% |
| inf2 | 36.4 | 36.6 | +0.5% |
| trn1 | 37.1 | 36.9 | −0.5% |

Every self-hosted backend landed within 4% of its first-run median, inf2 in a different region. g5g, trn1 and inf2 sit within 1.1 tok/s of each other, so their order is not stable between runs; the three groups are stable. Source: `bench-chat.txt`, `bench-chat.json`, `comparison.txt`.

---

#### Result 6 — Identical Text Where the Stack Is Identical

From the ten repeats in Result 5, compared in code (`comparison.txt`):

- Each self-hosted backend returned one distinct completion across its ten repeats. Bedrock returned 9 distinct completions in 10 at temperature 0.
- **g6 and SageMaker returned the same text**, word for word: the same weights on the same GPU type under vLLM 0.30, one as a VM and one as a managed endpoint.
- **inf2 and trn1 returned the same text**: one image digest, two chips, two regions.
- g6 and g5g differ (different checkpoint and dtype), and the Neuron build differs from both.

The live `ask.py` calls show the same pairs in `ask-g6.txt`, `ask-sagemaker.txt`, `ask-inf2.txt` and `ask-trn1.txt`.

---

#### Stage Timings

Wall clock for each command in `docs/DEMO-SCRIPT.md`, from `stage-timings.tsv`:

| Command | Seconds |
|---|---|
| `stage/ask.py <backend>`, one word | 0.4–1.1 |
| `stage/ask.py <backend>`, 100 tokens | 0.9–3.0 |
| `demo1_counting.py --runs 3` | 15.4 |
| `demo1_counting.py --runs 3 --tools rows` | 18.3 |
| `demo2_orchestrator.py` | 4.4–5.3 |
| `bench_chat.py --repeats 3` | 33.4 |

Every live moment in the talk fits in under 35 seconds.

---

#### Teardown

`stage/down.sh` terminated the four instances in us-east-1, us-east-2 and us-west-2 and deleted the SageMaker endpoint, endpoint config and model; its four-region sweep then listed nothing running. The SSM-only security groups stay, including `sg-05f1f649b55ad462d` that `up.sh` created in us-west-2 for the inf2 failover. Source: `down.log`.

---

#### Scope

One request at a time from one client, through SSM port forwards for the EC2 backends, so the EC2 latencies include the forward and Bedrock and SageMaker include their public API paths. The g5g serves a different checkpoint in float16; g6 and SageMaker serve the QAT w4a16 repack; the Neuron build is its own compile of E2B; Bedrock is Nova Micro, a different model, included as the managed reference. inf2 ran in us-west-2 because us-east-2 and us-east-1 had no capacity.
