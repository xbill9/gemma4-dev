# AWS Lambda GPU: facts read from AWS pages, 2026-11-18

Notes taken from the pages below, read on 2026-11-18 (the announcement's date). Field names and values
are as the pages give them; prose is paraphrased.

- Announcement: https://aws.amazon.com/blogs/aws/aws-lambda-now-supports-gpu-instances-for-event-functions-on-lambda-managed-instances/
  (18 NOV 2026). GPU support for Event Functions on Lambda Managed Instances (LMI), EC2 G-series instances,
  OCI container images, no drivers, CUDA versions or AMIs to manage. Example Dockerfile is one line,
  `FROM public.ecr.aws/lambda/deep-learning-containers/vllm:cuda-v1`, with no handler code.
- GPU docs: https://docs.aws.amazon.com/lambda/latest/dg/lambda-managed-instances-gpu.html
  - Instance families: G4dn (NVIDIA T4, 16 GB), G5 (A10G, 24 GB), G6 (L4, 24 GB), Gr6 (L4, 24 GB),
    G6e (L40S, 48 GB), G6f (fractional L4: 3, 6 or 12 GB), Gr6f (fractional L4, 12 GB). All sizes except g4dn.metal.
  - One GPU per execution environment. `AcceleratorConfig.AcceleratorMemorySize`: 3, 6, 12, 16, 24 or 48 (GB), a minimum.
  - Limits for GPU functions: container image only; image up to 32 GB (10 GB per file; other functions 10 GB);
    ephemeral storage 512 to 32,768 MB; MemorySize 2,048 to 32,768 MB; PerExecutionEnvironmentMaxConcurrency default 4, max 48.
  - Invocation: synchronous, asynchronous, or event source mappings. Lambda manages drivers; base images for vLLM, SGLang, PyTorch.
  - Scaling on GPU utilization, CPU utilization and concurrency; scale to zero available.
  - Quota: "Running On-Demand G and VT instances" defaults to 0 vCPUs.
- Base images: https://docs.aws.amazon.com/lambda/latest/dg/lambda-managed-instances-gpu-base-images.html
  base:cuda, cupy:cuda, pytorch:cuda, vllm:cuda, sglang:cuda on `public.ecr.aws/lambda/deep-learning-containers/`.
  vLLM image settings by environment variable: MODEL_ID, SERVED_MODEL_NAME, VLLM_GPU_MEM_UTIL (0.8),
  VLLM_MAX_MODEL_LEN, VLLM_SERVER_ARGS (extra `vllm serve` flags), VLLM_SERVER_TIMEOUT (600).
  `/tmp` is the only writable directory.
- Inference: https://docs.aws.amazon.com/lambda/latest/dg/lambda-managed-instances-gpu-inference.html
  A payload with `messages` goes to `/v1/chat/completions`, one with `prompt` to `/v1/completions`; the
  response is OpenAI-compatible. No streaming documented.
- Startup: https://docs.aws.amazon.com/lambda/latest/dg/lambda-managed-instances-gpu-startup.html
  Publishing starts the execution environments and loads the model; the version becomes Active when the
  server is ready. Init timeout: the larger of 130 s and the function timeout, up to 900 s.
- Scaling: https://docs.aws.amazon.com/lambda/latest/dg/lambda-managed-instances-scaling.html
  `PutFunctionScalingConfig` sets `MinExecutionEnvironments` and `MaxExecutionEnvironments` (set together;
  an unset one resets to its default). Default minimum 3 execution environments across Availability Zones,
  no default maximum. Both 0 deactivates the version.
- Operator role: https://docs.aws.amazon.com/lambda/latest/dg/lambda-managed-instances-operator-role.html
  managed policy `AWSLambdaManagedEC2ResourceOperator`, trust principal `lambda.amazonaws.com`.
- Pricing: https://aws.amazon.com/lambda/pricing/ — Lambda Managed Instances: $0.20 per million requests;
  a compute management fee on the EC2 On-Demand price, 15% for CPU instance types and 9.75% for GPU G-series
  instance types; standard EC2 instance pricing. No separate charge for execution duration.
