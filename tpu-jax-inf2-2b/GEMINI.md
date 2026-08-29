# Gemini workspace context: AWS Inferentia2 + Google TPU PyTorch DevOps

This project provides two MCP servers for serving PyTorch LLM workloads on
non-GPU accelerators: `tpu-pytorch-inf2-2b` (AWS EC2 Inferentia2 + Neuron serving)
and, via the bundled `tpu-management` skill, `tpu-devops` (Google Cloud TPU
capacity, Gemma 4 vLLM serving, and TorchTPU dev VMs).

## Inferentia2 (`tpu-pytorch-inf2-2b`)

The root `server.py`, `project-setup.sh`, and `requirements.txt` files are
authoritative. Run `make skill` after edits to refresh
`.claude/skills/tpu-pytorch-inf2-2b-management/mcp/` and `skills/tpu-pytorch-inf2-2b-management/`.

The server uses:

- boto3 with the standard AWS credential chain
- EC2 Inf2 instances and a regional Neuron DLAMI
- the versioned AWS Neuron vLLM Deep Learning Container, or the prebuilt
  Gemma-4 `torch_neuronx` Option-B container (`serving="optb"` — required
  for Gemma-4, which the vLLM DLC cannot serve)
- AWS Systems Manager rather than SSH for remote commands
- AWS Secrets Manager for gated-model tokens
- a `ManagedBy=tpu-pytorch-inf2-2b` tag to avoid touching unrelated instances
- spot capacity by default (`spot=False` for on-demand)

Configuration is controlled with `AWS_REGION`, optional `AWS_PROFILE`,
`MODEL_NAME`, `INSTANCE_TYPE`, `SERVICE_NAME`, `HF_SECRET_ID`, `VLLM_PORT`,
`VLLM_IMAGE`, `OPTB_IMAGE`, and `NEURON_AMI_NAME`.

Never expose credentials in user data or output. Treat instance termination as
permanent and require explicit user approval.

## TPU (`tpu-devops`, bundled skill)

`.claude/skills/tpu-management/` is a synced snapshot; its authoritative home
is the `tpu-skill-claude` repo. It provisions flex-start TPU VMs / queued
resources, runs Gemma 4 vLLM serving (v6e/v5p/v5e), and creates PyTorch
(TorchTPU) dev VMs with `workload="pytorch"` that smoke-test
`torch.compile(backend="tpu")`.

### TorchTPU Coding & Compilation Guardrails

- **No Legacy Imports**: Do **NOT** use PyTorch/XLA (`import torch_xla`) or `import torch_tpu`. Use pure PyTorch APIs targeting `torch.device("tpu")` or `torch.accelerator.current_accelerator()`.
- **Backend Compilation**: Always compile PyTorch models using `torch.compile(model, backend="tpu")` to maximize FLOPS utilization.
- **Graph Break Prevention**: Avoid mixing Python control flow (`if`/`else` on tensor values), scalar conversions (`.item()`), or `print()` statements inside compiled forward passes.
- **MXU Tile Alignment**: Align tensor batch sizes and sequence dimensions to multiples of **128** (or 8/16) using `pad_to_multiple_of=128` to fit the TPU Matrix Multiply Unit (MXU) systolic array tiles.
- **BFloat16 Precision**: Prefer native `torch.bfloat16` precision over `float32` (which is emulated).
- **Installation Pipeline**: On TPU VMs (Python 3.12/3.11), install `torch_tpu` from the GCP Artifact Registry:
  ```bash
  pip install --pre --index-url \
    "https://oauth2accesstoken:$(gcloud auth print-access-token)@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/" \
    torch_tpu
  ```
- **Marimo Documentation**: Inspect or run interactive documentation notebooks via:
  ```bash
  marimo edit /torch_tpu/docs/notebooks/
  ```

### Gemini CLI via LiteLLM proxy against a self-hosted TPU endpoint

Route Gemini CLI traffic to the Gemma 4 model served from a TPU VM:

```yaml
model_list:
  - model_name: "gemma4-tpu"
    litellm_params:
      model: "openai/google/gemma-4-31B-it"
      api_base: "http://YOUR_TPU_IP_ADDRESS:8000/v1"
      api_key: "none"
    router_settings:
      model_group_alias:
        "gemini-2.0-flash": "gemma4-tpu"
        "gemini-2.0-flash-lite": "gemma4-tpu"
        "gemini-1.5-flash": "gemma4-tpu"
        "gemini-1.5-pro": "gemma4-tpu"
```

```bash
pip install 'litellm[proxy]'
litellm --config litellm_config.yaml --port 4000
export GOOGLE_GEMINI_BASE_URL="http://localhost:4000"
export GEMINI_API_KEY="local-proxy-token"
export GEMINI_MODEL="google/gemma-4-31B-it"   # match the served model
```

Adjust `model` to the served checkpoint (`MODEL_NAME` env var of the agent).
If the CLI ignores `GOOGLE_GEMINI_BASE_URL`, check `gemini --help` for the
current base-URL override.

## Analysis standards

- Dependency portability: avoid assuming third-party analysis libraries like
  `pandas` are installed. Prefer standard libraries (`csv`, `json`) for data
  parsing and aggregation scripts.
