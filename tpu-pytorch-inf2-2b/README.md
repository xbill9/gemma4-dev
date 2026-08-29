# AWS Inferentia2 + Google TPU PyTorch DevOps MCP

This repository packages two AI DevOps/SRE agents for serving PyTorch LLM
workloads on non-GPU accelerators:

1. **`tpu-pytorch-inf2-2b` MCP server + `tpu-pytorch-inf2-2b-management` skill** — operates AWS EC2
   Inferentia2 (`inf2`) instances: Neuron DLAMIs, the AWS Neuron vLLM Deep
   Learning Container, and the prebuilt Gemma-4 `torch_neuronx` (Option-B)
   serving containers, with health checks and logs over Systems Manager.
2. **`tpu-devops` MCP server + `tpu-management` skill** (bundled) — operates
   Google Cloud TPU capacity (flex-start VMs, queued resources), Gemma 4 vLLM
   serving on v6e/v5p/v5e, and **PyTorch (TorchTPU)** dev VMs
   (`workload="pytorch"`). Its authoritative home is
   [`xbill9/tpu-skill-claude`](https://github.com/xbill9/tpu-skill-claude);
   this repo carries a synced copy of the skill.

## Part 1 — AWS Inferentia2 (`tpu-pytorch-inf2-2b`)

Provisions Neuron DLAMIs, starts serving containers, checks NeuronCore
health, reads logs through Systems Manager, manages instance lifecycle, and
queries the OpenAI-compatible endpoint.

### Quick start

Prerequisites:

- Python 3.11+ and the AWS CLI
- AWS credentials from an IAM role, AWS SSO, or a configured profile
- an existing VPC subnet and security group (allow the serving port only from
  trusted clients)
- an EC2 instance profile with `AmazonSSMManagedInstanceCore`, ECR Public pull
  access, and permission to read the configured Hugging Face secret
- EC2 Inferentia On-Demand or Spot vCPU quota in the target region

```bash
python3 -m pip install -r requirements.txt
./project-setup.sh . --region us-east-1 --instance-type inf2.xlarge
```

Restart Claude Code and confirm that `tpu-pytorch-inf2-2b` appears under `/mcp`.
Store a gated-model token with `save_hf_token`, then call
`create_inf2_instance` with your subnet, security group, and instance-profile
names. Use `get_deployment_config` first if you want a reviewable AWS CLI
command without changing AWS.

Launches default to **spot** capacity (`spot=False` for on-demand). For
Gemma-4 E2B, pass `serving="optb"` to deploy the prebuilt `torch_neuronx`
container (the vLLM DLC cannot serve Gemma-4) — single-device instance types
only; cloud-init adds the swapfile the 16 GB `inf2.xlarge` needs for the
one-time neff load.

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `AWS_REGION` | `us-east-1` | AWS region |
| `AWS_PROFILE` | unset | Optional shared-config profile |
| `MODEL_NAME` | `meta-llama/Llama-3.1-8B-Instruct` | Hugging Face model |
| `INSTANCE_TYPE` | `inf2.xlarge` | Inf2 size |
| `SERVICE_NAME` | `vllm-inf2` | EC2 Name tag |
| `HF_SECRET_ID` | `vllm/hf-token` | Secrets Manager secret |
| `VLLM_PORT` | `8000` | Host API port |
| `VLLM_IMAGE` | Neuron SDK 2.31 vLLM DLC | Versioned Neuron container |
| `OPTB_IMAGE` | `docker.io/xbill9/gemma4-optb:slim` | Gemma-4 E2B Option-B container |
| `NEURON_AMI_NAME` | `Deep Learning AMI Neuron*Ubuntu*` | Regional AMI filter |

Supported topology defaults are derived from EC2 hardware: `inf2.xlarge` and
`inf2.8xlarge` expose 1 device/2 NeuronCores, `inf2.24xlarge` exposes
6/12, and `inf2.48xlarge` exposes 12/24.

### Serving: prebuilt container, or the native engine

Two paths, and they are not interchangeable.

**Container — what the MCP tools deploy.** `serving="optb"` launches
`docker.io/xbill9/gemma4-optb:slim`, a `torch_neuronx`-traced graph with the
compiled neffs and the weights baked in, serving on port 8080. Every measurement
in `benchmarks/` came off this path.

**Native engine — `torch_generate.py` + `torch_openai_server.py`.** The same
design without the bake: it traces from a checkpoint at start-up (or reloads
saved graphs from `--neff-dir`) and serves OpenAI routes on port 8000 under
uvicorn. It defaults to the dense reference `google/gemma-4-E2B-it` rather than
the QAT checkpoint in the container.

```bash
# on the instance, with the Neuron DLAMI's torch/torch-neuronx already present
python3 -m pip install -r requirements-serving.txt

# smoke-test the engine in-process, outside HTTP
python3 torch_generate.py --prompt "What is AWS Inferentia?" --stats

# device against CPU, token-for-token, plus per-slot isolation
python3 torch_generate.py --parity --batch 4

# serve
python3 torch_openai_server.py --port 8000 --neff-dir /opt/gemma4/neff
```

> **No MCP tool deploys the native engine yet.** `create_inf2_instance` still
> renders cloud-init for the container, so the engine is started by hand for now.
> It has **not been run on a device** — it is the container's proven graph design,
> reorganised and parameterised, and nothing more than that until a run says
> otherwise.

Three properties are worth knowing before touching it:

- **The embedding lookup runs on the host, on purpose.** A gather over the
  4.70 GB per-layer table returns zeros on a NeuronCore rather than raising,
  which decodes to an EOS — a clean `200 OK` with an empty completion and no
  error anywhere. `docs/neuron-jax-quirks.md` has the evidence.
- **`--batch`, `--max-total` and `--prompt-bucket` are traced into the graph.**
  Changing one is a recompile, not a flag, which is why `--batch` defaults to 1
  and `--neff-dir` is worth setting.
- **Empty output is the signature failure here, not an edge case.** Check the
  text, never the token count.

### Tools

- `get_deployment_config`, `create_inf2_instance`, `list_inf2_instances`
- `start_inf2_instance`, `stop_inf2_instance`, `terminate_inf2_instance`
- `verify_neuron_health`, `get_vllm_logs`, `get_endpoint`, `query_model`
- `save_hf_token`, `check_inf2_quotas`, `get_help`

Remote operations use SSM Run Command, so the generated deployment does not
require an SSH key or an internet-wide port 22 rule. Instances are tagged
`ManagedBy=tpu-pytorch-inf2-2b`; discovery is deliberately scoped to that tag.

## Part 2 — Google Cloud TPU (`tpu-devops`, bundled skill)

The bundled `tpu-management` skill (`.claude/skills/tpu-management/`) wraps
the `tpu-devops` FastMCP agent covering the full TPU serving lifecycle:

- **Capacity discovery & provisioning:** sweep zones for available capacity
  (`find_tpu_vm` for flex-start VMs, `find_tpu` for queued resources), check
  quotas, estimate cost, create flex-start TPU VMs (v6e/v5p) or legacy queued
  resources (v5e) with an auto-serving startup script, then
  `wait_for_vllm_ready`.
- **PyTorch (TorchTPU) workloads:** create flex-start VMs with
  `workload="pytorch"` — the startup script installs PyTorch + the TorchTPU
  backend (no docker, no HF token) and smoke-tests
  `torch.compile(backend="tpu")`; `wait_for_pytorch_ready` polls until ready
  and `verify_pytorch_tpu` reruns the smoke test over SSH.
- **Serving stack control:** manage the vLLM Docker container
  (`manage_vllm_docker`), fetch endpoints and the gcloud deployment
  one-liner, store the HF token in Secret Manager.
- **Health, logs & diagnostics:** system status dashboard, model health
  verification, vLLM/docker/system/serial logs, Cloud Logging retrieval, and
  Gemma-4-powered log triage (`analyze_cloud_logging`).
- **Inference & benchmarking:** query the deployed Gemma 4 endpoint
  (optional TTFT/throughput stats), run `vllm bench serve`.

Install options, the plugin marketplace flow, and the full lifecycle guide
live in the authoritative repo
([`xbill9/tpu-skill-claude`](https://github.com/xbill9/tpu-skill-claude)) and
in `.claude/skills/tpu-management/SKILL.md` here. The skill's
`references/tpu-guide.md` covers flex-start zones per TPU family, quota
metrics, and troubleshooting.

## Development

```bash
make skill
make test
make lint
make skill-package
```

The root `server.py`, `project-setup.sh`, `requirements.txt`,
`requirements-serving.txt`, `torch_generate.py` and `torch_openai_server.py` are
authoritative for the **inf2** agent; `make skill` refreshes the bundled
copies under `.claude/skills/tpu-pytorch-inf2-2b-management/` and `skills/`. The
`tpu-management` skill copy is maintained in its own repo — update it there
and re-sync rather than editing the snapshot here.

## Benchmarks

`benchmarks/` holds serving reports (`serving-report.schema.json` format) and
raw run logs, including Gemma-4 E2B on a single-core `inf2.xlarge`
(`torch_neuronx` Option-B, ~44-46 tok/s) and TorchTPU E2B runs on a
flex-start v6e-1.
