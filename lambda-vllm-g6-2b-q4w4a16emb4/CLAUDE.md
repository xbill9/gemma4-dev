# CLAUDE.md — `lambda-vllm-g6-2b-q4w4a16emb4`

Gemma 4 E2B as an **AWS Lambda function on a GPU**, through Lambda Managed Instances (LMI) GPU support
(announced 2026-11-18). Lambda creates the g6.2xlarge (one NVIDIA L4) behind a capacity provider and runs the
model from a container image; a client reaches it with `aws lambda invoke`. Created 2026-11-18.

**Slot 1 is the only slot that differs from `gpu-vllm-g6-2b-q4w4a16emb4`**: same checkpoint
(`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4` @ `db715f83`), same vLLM version (0.33.0), same instance
type, same serving flags. The difference is who provisions the instance (`NAMING.md`, slot 1 `lambda`).

No MCP server, skill or plugin: `tpu.env` decides, `lmi.py` does the work through the AWS CLI (boto3 1.43
has no `AcceleratorConfig`; aws-cli 2.41 does), `image/Dockerfile` bakes the checkpoint into AWS's image.

## Read first

- **The image is AWS's Lambda vLLM image** (`public.ecr.aws/lambda/deep-learning-containers/vllm`, pinned
  by digest in `tpu.env`): vLLM 0.33.0, Python 3.13, CUDA 13.0.3, read from its build history. Its engine
  extension starts `vllm serve` during Lambda Init and recycles the environment if `/health` fails; its
  handler (`/var/task/handler.py`) forwards the invocation payload as-is to `/v1/chat/completions` (payload
  has `messages`) or `/v1/completions`. So `chat_template_kwargs`, `tools` and `max_tokens` pass through.
  Serving flags go in `VLLM_SERVER_ARGS`; `lmi.py` builds them from `tpu.env`.
- **The checkpoint is in the image** (`/opt/model`, 2,862,925,039 bytes), so the instance needs no Hugging
  Face access and nothing streams from S3. `image/model/` is gitignored; `make model` fetches it.
- **Lambda's default is 3 execution environments per published version, across AZs.** This account's
  "Running On-Demand G and VT instances" quota in us-east-1 is 16 vCPUs (two g6.2xlarge), so `lmi.py`
  sets 1 / 1 on the version right after publishing. **A published version keeps its minimum running and
  billed** (EC2 on-demand plus a 9.75% management fee for G-series, $0.20 per million requests; Lambda
  pricing page, 2026-11-18). `make down` sets min and max to 0 and releases the instance; `make destroy`
  deletes the function and the capacity provider.
- **Publishing is the cold start.** A version turns Active only after its environments have loaded the
  model; `lmi.py function` records the seconds from publish to Active in `.lmi-state.json`.
- Network: the default VPC's subnets and its default security group (outbound open). Roles:
  `lambda-lmi-gpu-operator` (`AWSLambdaManagedEC2ResourceOperator`) and `lambda-lmi-gpu-exec`
  (`AWSLambdaBasicExecutionRole`).
- **No streaming.** The AWS pages document sync, async and event-source invocation only, and the image's
  handler returns the whole completion. Time to first token cannot be measured through it.

## Commands

`make deploy | status | ask Q='…' [THINK=1] | up | down | destroy | test | lint`.
Tests are `unittest`, offline.
