# CLAUDE.md — AWS Inferentia2 + TPU PyTorch DevOps MCP

This repository packages the `tpu-pytorch-inf2-2b-management` Claude Code skill and the
`tpu-pytorch-inf2-2b` FastMCP server for AWS EC2 Inf2 instances, plus a bundled copy
of the `tpu-management` skill / `tpu-devops` MCP agent for Google Cloud TPUs
(authoritative home: the `tpu-skill-claude` repo).

## Authoritative files

- `server.py`: inf2 MCP server and AWS lifecycle implementation
- `torch_generate.py`: the native `torch_neuronx` Gemma-4 engine, plus a one-shot
  CLI. Do not confuse it with `torchtpu_generate.py` beside it, which targets a
  **Google TPU** and is governed by the TorchTPU rules below; none of those apply
  to Neuron.
- `torch_openai_server.py`: OpenAI-compatible FastAPI/SSE server over that engine.
  Imports the engine rather than copying it — see below.
- `project-setup.sh`: skill installer and MCP registration
- `requirements.txt`: control-plane dependencies
- `requirements-serving.txt`: what gets installed **on the instance**. Disjoint
  from `requirements.txt`, and it deliberately lists neither `torch` nor
  `torch-neuronx` (both ship in the DLAMI, matched to the runtime) nor
  `optimum-neuron` (no Gemma-4 model class).
- `.claude/skills/tpu-pytorch-inf2-2b-management/SKILL.md`: inf2 workflow and guardrails
- `.claude/skills/tpu-management/`: bundled snapshot of the TPU skill — edit
  it in its own repo and re-sync; only consume it here (SKILL.md lifecycle,
  `references/tpu-guide.md` zones/quotas/troubleshooting)

Run `make skill` after editing sources. This refreshes the inf2 MCP snapshots
and the plugin-layout copy. Run `make skill-package` when publishing a release.

## Two serving paths, and only one of them is wired

- **Container (what `server.py` deploys today).** `serving="optb"` launches the
  prebuilt `docker.io/xbill9/gemma4-optb:slim` image — a `torch_neuronx`-traced
  graph with the neffs and weights baked in, serving on port 8080. This is the
  path with measurements behind it.
- **Native engine (`torch_generate.py` + `torch_openai_server.py`).** The same
  design, un-baked: it traces from a checkpoint at start-up (or reloads neffs
  from `--neff-dir`) and serves OpenAI routes on 8000 under uvicorn. It defaults
  to the **dense reference** `google/gemma-4-E2B-it`, not the QAT checkpoint the
  container carries.

**There is no MCP tool that deploys the native engine yet.** `create_inf2_instance`
still renders cloud-init for the container. Until that lands, the engine is run by
hand on the instance, and any claim that this rig "serves through the native
engine" is false. The G5g sibling's `deploy_torch_server` is the shape the missing
tool should take.

## Engine rules (Neuron)

These are load-bearing and every one of them has a silent failure behind it —
`docs/neuron-jax-quirks.md` has the measurements.

- **The embedding gather stays on the host.** `embed_tokens` and
  `get_per_layer_inputs` run on the CPU and their outputs are fed into the traced
  graph as `inputs_embeds` / `per_layer_inputs`. The per-layer table is 4.70 GB,
  and a gather that large on a NeuronCore returns **zeros instead of raising** —
  which decodes to the pad id, which is an EOS, so the endpoint answers `200 OK`
  with an empty completion and nothing reports a fault. Moving that gather onto
  the device is the single most expensive change anyone could make here.
- **Shapes are traced, not requested.** `batch`, `max_total` and `prompt_bucket`
  are baked into the graph; a caller cannot vary them and a new value is a
  multi-minute recompile. Sampling parameters stay on the host for the same
  reason — as trace-time constants they would be a compile per distinct
  `temperature`.
- **The engine exists once.** `torch_openai_server.py` imports
  `NeuronGemmaEngine`; it does not carry a second copy. A drifted copy of the KV
  blend or the per-stream masks produces plausible text, not an error.
- **Check content, never token count.** An isolation run once recorded 20 tokens
  as a 2700x win; that same configuration returned an empty string end-to-end.
  `torch_generate.py --parity` is the check that means something: device against
  CPU token-for-token, *and* duplicate prompts in different slots emitting
  identical sequences, which is what catches a per-stream mask bug that B=1 hides.

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
