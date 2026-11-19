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

## Drafter (since 2026-11-19)

It serves with Google's E2B QAT assistant drafter (W4A16, built by `repack/repack_drafter.py`, 4 drafted
tokens; `SPECULATIVE_MODEL` empty turns it off). Measured through Lambda: 1.63x for one request (256 tokens
in 1.14–1.20 s against 1.92–2.02 s), 1.47x at 4, 0.88x at 16 and 0.63x at 48 requests at once
(`benchmarks/runs/2026-11-18-drafter-lambda`). Right for a chat demo with a few users; for heavy batch load
publish without it.

**Republish with `republish.py`, not `lmi.py function`, while environments run.** A republish starts the new
environment before stopping the old one; at the capacity provider's vCPU cap it fails and Lambda rolls
`$LATEST.PUBLISHED` back to the previous code. `republish.py` scales to 0, publishes, scales back to 1/1 and
prints vLLM's engine line, so the configuration that actually started can be checked. Set MIN and MAX together:
with only a minimum, Lambda scaled to a second instance under load.

## Chat page

`make ui` serves `chat_ui.py` on http://localhost:8768 (`UI_PORT`), forked from
`tokenlab-sglang-amd-31b/chat_ui.py` (itself from `~/tpu-inference/tpu-vllm-v5e1-2b/chat_ui.py`). The local proxy
turns each `/v1/chat/completions` POST into a synchronous boto3 `Invoke` of `$LATEST.PUBLISHED` with this machine's
AWS credentials, so nothing about AWS reaches the browser. No stream, so the page shows the whole reply with its round
trip and end-to-end tok/s; the panel shows the version's state and scaling (`/ui/status`, cached 10 s) and this
session's counts. Text only (the function serves `--language-model-only`), so the Token Lab page's image button is gone.

## Commands

`make deploy | status | ui | ask Q='…' [THINK=1] | up | down | destroy | test | lint`.
Tests are `unittest`, offline.
