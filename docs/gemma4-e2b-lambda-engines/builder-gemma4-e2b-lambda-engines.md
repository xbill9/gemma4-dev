# Five Serving Engines on a Lambda GPU: Gemma 4 E2B Under vLLM, SGLang, llama.cpp, PyTorch and JAX

*Subtitle: The same Gemma 4 E2B serving sweep, run twice on one NVIDIA L4: once on a plain EC2 G6 and once as AWS Lambda GPU functions. Which engine wins, what Lambda adds to every request, and what it gives back.*

This article provides a step by step comparison of five serving engines for Gemma 4 E2B, each deployed as an AWS Lambda function on an NVIDIA L4 GPU. vLLM, SGLang, llama.cpp, a hand-built PyTorch int4 engine and a pure-JAX port each get their own container image, their own function and a local chat page, and each is measured against the same engine on a plain EC2 G6.

https://github.com/xbill9/gemma4-dev/tree/main/docs/gemma4-e2b-lambda-engines

Two questions decide the write-up. Which serving engine is best for this model on this card? And what changes when the same engine runs as a Lambda function instead of on an EC2 instance you start yourself? Every figure below was measured on 2026-11-18 and 2026-11-19 in us-east-1, and every number traces to a file in the article's `evidence/` directory.

---

#### What Was Already Measured

The EC2 G6 sweep ran six engines on one L4 on 2026-11-05: the same Gemma 4 E2B weights on Google's QAT Q4_0 grid, one engine at a time, six prompts, three repeats, timed on the instance. The vLLM rig was the first to move to Lambda, on the day Lambda gained GPU support:

https://dev.to/gde/serverless-gemma-4-e2b-on-a-lambda-gpu-function-with-googles-qat-weights-and-drafter-fh4

This sweep moves the other four. Each Lambda rig keeps its G6 rig's checkpoint, engine version and serving flags, so the platform is the variable.

| Engine | Weights | Lambda rig |
| :--- | :--- | :--- |
| vLLM 0.33.0 | W4A16 QAT, drafter at 4 tokens | `lambda-vllm-g6-2b-q4w4a16emb4` |
| SGLang 0.5.23 | W4A16 QAT, two patches | `lambda-sglang-g6-2b-q4w4a16emb4` |
| llama.cpp `7182ef71e` | exact Q4_0 GGUF, drafter at 2 tokens | `lambda-llamacpp-g6-2b-q4_0` |
| PyTorch int4 engine | W4A16 QAT | `lambda-pytorch-g6-2b-q4w4a16` |
| JAX 0.11.1 port | bf16 reference checkpoint | `lambda-jax-g6-2b` |

---

#### At This Point You Should Have…

- An AWS account with the AWS CLI v2 (2.41 here) and a "Running On-Demand G and VT instances" quota of at least 8 vCPUs in your region
- Python 3 with `boto3` and `huggingface_hub`
- A clone of `gemma4-dev`; each engine is one directory with a `Makefile`, `tpu.env` and `image/Dockerfile`

---

#### Step 1 — One Image per Engine

AWS publishes Lambda base images for vLLM, SGLang and PyTorch, plus a plain CUDA base. Each comes with an engine extension that starts the server during Lambda's Init phase and a handler that forwards each invocation to the server's `/v1/chat/completions`. The other engines get the same pattern from two small files in each rig: `image/engine_ext.py` runs the command in `ENGINE_CMD` and waits for `/health`, and `image/handler.py` forwards the request.

| Rig | Base image | What is added | Image size |
| :--- | :--- | :--- | ---: |
| SGLang | AWS Lambda SGLang (0.5.23) | the G6 rig's two patches, checkpoint | 10,051,465,109 bytes |
| llama.cpp | AWS Lambda base (CUDA 13.0.3) | `llama-server` built for sm_89, GGUF and drafter | 4,224,819,497 bytes |
| PyTorch | AWS Lambda PyTorch (torch 2.14.1) | the int4 engine's five files, both checkpoints | 12,033,715,429 bytes |
| JAX | AWS Lambda base | `jax[cuda13]==0.11.1`, the port, bf16 checkpoint | 11,936,153,444 bytes |

The Dockerfile for llama.cpp compiles the server in a stage on the same base image, so the binary links against the CUDA libraries Lambda runs it with:

```dockerfile
FROM ${BASE_IMAGE} AS build
RUN dnf install -y --setopt=install_weak_deps=False cmake git gcc-c++ make \
      cuda-nvcc-13-0 cuda-cudart-devel-13-0 libcublas-devel-13-0 cuda-driver-devel-13-0
RUN git clone -q https://github.com/ggml-org/llama.cpp /src && git -C /src checkout -q "$LLAMA_CPP_COMMIT"
RUN cmake -S /src -B /src/build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
      -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DBUILD_SHARED_LIBS=OFF \
 && cmake --build /src/build -j "$(nproc)" --target llama-server
```

The PyTorch and JAX engines load their checkpoints by repository name, so their images carry a Hugging Face cache at `/opt/hf` with `refs/main` pinned to the measured revision, read with `HF_HUB_OFFLINE=1`. Everything an engine writes goes to `/tmp`, the one writable path in a Lambda environment.

---

#### Step 2 — Build on EC2, Push to ECR

The images are 4.2 to 12.0 GB, so they are built on one CPU instance in the same region and pushed straight to ECR:

```bash
make builder    # one c7i.4xlarge, found by its Name tag, shared by every rig
make image      # tar image/ to S3, buildx on the builder, push to ECR
make builder-down
```

```
    458 s  BUILD_EXIT=0      # SGLang
    534 s  BUILD_EXIT=0      # llama.cpp
    573 s  BUILD_EXIT=0      # JAX
    612 s  BUILD_EXIT=0      # PyTorch
```

The four builds ran side by side on the one builder. The build uses `--provenance=false --sbom=false` and `oci-mediatypes=false`, because Lambda accepts a single-platform Docker v2 manifest.

---

#### Step 3 — A Capacity Provider and a Function

The functions run on a Lambda Managed Instances capacity provider. Its instance list allows g6.2xlarge and g6.4xlarge, both one L4:

```json
"InstanceRequirements": {"Architectures": ["x86_64"], "AllowedInstanceTypes": ["g6.2xlarge", "g6.4xlarge"]},
"CapacityProviderScalingConfig": {"ScalingMode": "Auto", "MaxVCpuCount": 16}
```

`make deploy` creates the function with `AcceleratorConfig.AcceleratorMemorySize` 16, 16 GiB of memory at 4 GiB per vCPU (24 GiB for JAX), one execution environment, and publishes it. Lambda launches the instance, pulls the image and runs Init before the version turns Active:

| Engine | Lambda Init (engine start) | Publish to Active | Engine start on EC2 G6 |
| :--- | ---: | ---: | ---: |
| llama.cpp | 5.3 s | 696.2 s | 6.1 s |
| JAX | 42.3 s | 532.1 s | – |
| vLLM | 156.4 s, 205.4 s | 484.0 s | 167.3 s |
| PyTorch | 198.3 s | 598.3 s | 236.2 s |
| SGLang | 346.5 s, 347.4 s, 372.5 s | – | 307.2 s |

The engines start in Lambda at the same pace as on EC2. The rest of the eight to twelve minutes from publish to Active is Lambda launching the instance and pulling the image.

---

#### Step 4 — A Chat Page for Each Engine

Every rig ships the same local chat page. Each message becomes one synchronous `Invoke` of the published version, signed with the machine's own AWS credentials, so the browser never sees an AWS key and the function has no public endpoint:

```bash
make up    # one execution environment
make ui    # http://localhost:8768 vLLM, 8769 SGLang, 8770 llama.cpp, 8771 PyTorch, 8772 JAX
make down  # zero environments: the instance is released, the function is kept
```

A six-token reply from llama.cpp, through the page and Lambda:

```
X-Lambda-Seconds: 0.138
{"choices": [{"message": {"role": "assistant", "content": "Hi, hello there."}}], ...
 "timings": {"prompt_n": 14, "prompt_ms": 21.405, "predicted_n": 6, "predicted_ms": 39.963, "draft_n": 8, "draft_n_accepted": 1}}
```

---

#### Step 5 — The Benchmark

`bench.py` sends one prompt with a unique prefix, greedy, thinking off, 256 output tokens with `ignore_eos`, as one request and then four at once. Every rig ran four times, alone on the capacity provider, and was scaled to zero before the next one started. Times are taken on the client, a desktop in the US calling us-east-1, so every figure includes the invoke round trip.

```
llamacpp        one  177.3 tok/s (4 runs) native  207.7 ratio 0.85 | s/request lambda 1.444 native 1.233 added 0.211 | 4 at once  405.7 (4 runs, 382.5-409.4) native 373.2 ratio 1.09
sglang          one  125.3 tok/s (4 runs) native  136.5 ratio 0.92 | s/request lambda 2.042 native 1.875 added 0.166 | 4 at once  458.4 (4 runs, 170.7-472.0) native 475.4 ratio 0.96
pytorch         one  127.0 tok/s (4 runs) native  134.7 ratio 0.94 | s/request lambda 2.014 native 1.901 added 0.113 | 4 at once  138.1 (4 runs, 135.5-138.3) native 134.5 ratio 1.03
jax             one   47.9 tok/s (3 runs) native   48.3 ratio 0.99 | s/request lambda 5.345 native 5.300 added 0.045 | 4 at once   54.6 (4 runs, 46.2-54.7)
```

`analyze.py` prints these from the run files. The first JAX request on a new environment compiled for 38.3 s and is left out of the medians; XLA's compilation cache sits in `/tmp`, which starts empty in every environment.

---

#### Finding 1 — Which Serving Engine Is Best?

Output tokens per second, median of the runs, through Lambda:

| Engine | One request | Four at once | Four vs one |
| :--- | ---: | ---: | ---: |
|  vLLM, drafter at 4 | 218.6 | 554.8 | 2.54x |
|  llama.cpp, drafter at 2 | 177.3 | 405.7 | 2.29x |
| vLLM, no drafter | 130.2 | 486.2 | 3.73x |
| PyTorch int4 | 127.0 | 138.1 | 1.09x |
| SGLang | 125.3 | 458.4 | 3.66x |
| JAX bf16 | 47.9 | 54.6 | 1.14x |

**vLLM is the best engine for Gemma 4 E2B on one L4, on Lambda and on EC2 alike.** With Google's assistant drafter it serves one request at 218.6 tok/s through Lambda and 275.3 on EC2, the fastest of the five on both. Without a drafter it batches best: four requests at once give 3.73x one. Its two four-request runs with the drafter differ widely, 386.3 and 723.2 tok/s, so that median carries less weight than the others.

**llama.cpp is the best small engine.** It is second for one request with its own drafter, its image is 4.2 GB against 10 to 12 GB for the others, and Lambda's Init for it takes 5.3 s. It batches four requests at 2.29x one.

**SGLang matches vLLM without a drafter** at four requests (458.4 against 486.2 tok/s) and takes the longest to start: Lambda's Init for it runs 346.5 to 372.5 s.

**PyTorch and JAX serve one request at a time.** The PyTorch int4 engine matches the batching engines for a single request (127.0 tok/s); JAX serves the 10 GB bf16 reference checkpoint at 47.9.

---

#### Finding 2 — The Gap Between Lambda and a Plain G6

| Engine | One request, Lambda | One request, EC2 G6 | Lambda / EC2 | Seconds Lambda adds per request |
| :--- | ---: | ---: | ---: | ---: |
| vLLM, drafter at 4 | 218.6 | 275.3 | 0.79 | 0.241 |
| llama.cpp, drafter at 2 | 177.3 | 207.7 | 0.85 | 0.211 |
| SGLang | 125.3 | 136.5 | 0.92 | 0.166 |
| vLLM, no drafter | 130.2 | 140.4 | 0.93 | 0.142 |
| PyTorch int4 | 127.0 | 134.7 | 0.94 | 0.113 |
| JAX bf16 | 47.9 | 48.3 | 0.99 | 0.045 |

**Lambda adds 0.045 to 0.241 s to each request, and the fast engines feel it most.** A 256-token reply from JAX takes 5.300 s on EC2 and 5.345 s through Lambda, under 1% more. The same reply from vLLM with its drafter takes 0.930 s on EC2, and Lambda adds 0.241 s, about a quarter. The six-token chat reply above came back through the page in 0.138 s, which bounds the invoke path itself.

The added seconds also hold a difference in prompts: the EC2 sweep averaged six prompts timed on the instance, while this sweep timed one prompt from a desktop. A drafter's speed depends on how many drafted tokens the prompt lets it accept, which is the likely reason the two drafter engines show the largest gap.

**At four requests the gap narrows for SGLang, PyTorch and llama.cpp**: 0.96, 1.03 and 1.09 of EC2, with the added time spread across four replies that run together. vLLM stays at 0.78 with its drafter and 0.91 without.

---

#### Compare and Contrast

| | Lambda GPU function | Plain EC2 G6 |
| :--- | :--- | :--- |
| Who runs the instance | Lambda launches, health-checks and replaces it | you do |
| Speed, one request | 0.79–0.99 of EC2 | 1.00 (the reference) |
| Speed, four at once | 0.78–1.09 of EC2 | 1.00 (the reference) |
| Streaming | no: replies arrive whole | yes, first token in 16–55 ms (vLLM, llama.cpp, SGLang) |
| Start of a new environment | 484–696 s publish to Active | the engine's own start, 6–307 s, once the instance runs |
| Access | IAM-signed `Invoke`, no open port | a port you secure yourself |
| Idle | `make down` releases the instance and keeps the function | stop or terminate the instance |
| Capacity | `InsufficientCapacity` from Lambda; the instance type it picked is not shown | your choice of type and zone |

---

####  Tip: Lambda Can Report a Failed Publish While the Engine Is Still Starting

SGLang needs 346.5 to 372.5 s of Init. Twice, Lambda marked its published version `Failed` ("An internal error occurred") while Init was still running, and then logged a successful init report minutes later. Left alone, the version turned Active by itself. The sweep driver checks readiness with a real request every minute for up to 30 minutes and ignores the publish status.

---

####  Tip: Set the Scaling Right After Publishing, and Retry It

A published version defaults to three execution environments, more than a 16-vCPU G quota holds. `PutFunctionScalingConfig` is refused for a moment after `publish-version`, so `lmi.py` retries it until Lambda accepts 1 / 1.

---

####  Tip: Allow More Than One Instance Size

From 17:01 to 17:38 UTC on 2026-11-19 every publish on a g6.2xlarge-only provider failed with "EC2 capacity isn't currently available in the specified Availability Zones", in all five us-east-1 zones that offer the type. On a provider that also allows g6.4xlarge, the same L4 with more vCPUs, Lambda found an instance on every attempt.

---

#### So, Which One?

For this model on one L4, **vLLM with Google's drafter** is the engine to serve, on either platform. **llama.cpp** is the choice when the image size and the engine's own start time matter most.

**Lambda** suits a model that is called now and then by people or programs with AWS credentials: no instance to manage, no open port, and nothing running between `make down` and `make up`. **A plain G6** suits interactive chat that needs streaming, steady traffic that keeps an instance busy, and anything where a quarter of a second per request matters.

---

#### Summary

The goal of this article was to find the best serving engine for Gemma 4 E2B on one NVIDIA L4 and to measure what running it as an AWS Lambda function changes. The key to the solution was keeping each engine's checkpoint, version and flags identical to the EC2 G6 sweep, so the platform was the only variable. The results were:

- vLLM with the drafter is fastest for one request on both platforms: 218.6 tok/s through Lambda, 275.3 on EC2
- Every engine runs on Lambda from its own image; AWS's images cover vLLM, SGLang and PyTorch, and two small files cover the rest
- Engines start in Lambda as fast as on EC2: Init takes 5.3 s for llama.cpp and 346.5–372.5 s for SGLang
- Lambda adds 0.045–0.241 s to each request: 0.79–0.99 of EC2 speed for one request, 0.78–1.09 at four
- A new environment takes 484–696 s from publish to Active, mostly instance launch and image pull
- No streaming through Lambda, so time to first token cannot be measured or shown

Scope: one NVIDIA L4 per platform in us-east-1. Lambda figures are four runs per engine (vLLM: two to six) of one 256-token prompt, timed from a desktop, with the instance behind each Lambda run either a g6.2xlarge or a g6.4xlarge; EC2 figures are the 2026-11-05 sweep's six prompts and three repeats on a g6.4xlarge, timed on the instance, and the JAX rig's own first run on a g6.2xlarge. vLLM's Lambda runs are from 2026-11-18, the others from 2026-11-19.

The strategy for serving Gemma 4 with five engines on a Lambda GPU was validated with an incremental step by step approach.

---

#### References

- Lambda vLLM rig, first article: https://dev.to/gde/serverless-gemma-4-e2b-on-a-lambda-gpu-function-with-googles-qat-weights-and-drafter-fh4
- Repository: https://github.com/xbill9/gemma4-dev
- AWS Lambda Managed Instances: https://docs.aws.amazon.com/lambda/latest/dg/lambda-managed-instances.html
- vLLM: https://github.com/vllm-project/vllm
- SGLang: https://github.com/sgl-project/sglang
- llama.cpp: https://github.com/ggml-org/llama.cpp
- JAX: https://github.com/jax-ml/jax
