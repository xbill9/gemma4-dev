# CLAUDE.md — AWS Inferentia2 + TPU PyTorch DevOps MCP

This repository packages the `tpu-pytorch-inf2-2b-management` Claude Code skill and the
`tpu-pytorch-inf2-2b` FastMCP server for AWS EC2 Inf2 instances, plus a bundled copy
of the `tpu-management` skill / `tpu-devops` MCP agent for Google Cloud TPUs
(authoritative home: the `tpu-skill-claude` repo).

## Authoritative files

- `server.py`: inf2 MCP server and AWS lifecycle implementation
- `project-setup.sh`: skill installer and MCP registration
- `requirements.txt`: runtime dependencies
- `.claude/skills/tpu-pytorch-inf2-2b-management/SKILL.md`: inf2 workflow and guardrails
- `.claude/skills/tpu-management/`: bundled snapshot of the TPU skill — edit
  it in its own repo and re-sync; only consume it here (SKILL.md lifecycle,
  `references/tpu-guide.md` zones/quotas/troubleshooting)

Run `make skill` after editing sources. This refreshes the inf2 MCP snapshots
and the plugin-layout copy. Run `make skill-package` when publishing a release.

## Engineering rules (inf2)

- Use boto3 and the standard AWS credential provider chain.
- Require explicit existing subnet, security-group, and instance-profile
  identifiers for creation. Do not create broad network or IAM policy.
- Use SSM Run Command for remote administration; do not require SSH keys.
  SSM `AWS-RunShellScript` executes under dash — no bashisms, and PATH lacks
  `/opt/aws/neuron/bin`.
- Scope instance discovery to the `ManagedBy=tpu-pytorch-inf2-2b` tag.
- Store Hugging Face tokens in Secrets Manager, never user data.
- Accept supported Inf2 sizes only and derive device/Core topology from them.
- Launches default to spot; surface capacity errors and offer on-demand
  rather than silently retrying.
- Treat stopping and termination as destructive. Termination is permanent
  (one-time spot instances cannot be stopped, only terminated).
- Keep Neuron SDK, PyTorch, vLLM, and DLC versions compatible as a unit.
- The vLLM DLC cannot serve Gemma-4 (`optimum-neuron` has no model class —
  the endpoint comes up healthy and serves gibberish). Use `serving="optb"`
  (prebuilt `torch_neuronx` Option-B container) for Gemma-4 E2B.

## Technical standards: PyTorch (TorchTPU) on TPU

When writing or reviewing PyTorch code destined for TPU VMs (see the TPU
SKILL.md "PyTorch (TorchTPU) on TPU" section for the full list):

- **Plain `torch` only:** never `import torch_tpu` (installing the package
  registers the TPU backend via autoload) and never PyTorch/XLA
  (`torch_xla`) — flag either on sight. Target the TPU with
  `torch.device("tpu")` / `device="tpu"`; `.cpu()` forces materialization.
  (Exception: kernel authoring and internal tooling use documented
  `torch_tpu._internal` modules — `pallas`, `profiler`, `execution_mode`.)
- **Never pip-install `torch` manually** on TorchTPU VMs: the authenticated
  torch_tpu index pulls the matching CPU torch automatically. If the backend
  fails to autoload with an `undefined symbol` error, the torch base version
  drifted — reinstall the pinned `+cpu` base from the same index.
  TorchTPU requires Python 3.11+ (3.12 preferred).
- **Compile for TPU:** `torch.compile(model, backend="tpu")`; avoid graph
  breaks in compiled forward passes (no Python control flow on tensor
  values, `.item()`, or prints).
- **MXU-friendly shapes and dtype:** batch sizes / tensor dims in multiples
  of 128 (or at least 8/16); prefer `torch.bfloat16` (native on the MXU —
  float32 is emulated).
- **Pallas custom kernels:** `pallas.jax_op` requires type annotations on
  the JAX wrapper signature. Warm up fused/compiled paths at least twice
  before benchmarking (verify zero cache misses via
  `torch.tpu._get_cache_stats()`).

## Technical standards: vLLM & Gemma 4 tool calling (TPU serving)

- **Optimization flags:** `--tensor-parallel-size 8` (TPU v6e-8),
  `--disable_chunked_mm_input`, `--max-model-len 16384`
- **Tool parsing:** `--enable-auto-tool-choice`, `--tool-call-parser gemma4`,
  `--reasoning-parser gemma4`
- **Multimodal:** `--limit-mm-per-prompt '{"image":4,"audio":1}'` and
  `--max_num_batched_tokens 4096`

## Coding standards

- Dependency portability: do not assume third-party libraries like `pandas`
  are installed. Prefer the standard library (`csv`, `json`) for data
  parsing and aggregation scripts.

Validation without AWS/GCP credentials can cover Python compilation, shell
syntax, tool schemas, cloud-init rendering, and package synchronization. Live
lifecycle checks require an authorized AWS account with Inf2 quota or a GCP
project with TPU quota.
