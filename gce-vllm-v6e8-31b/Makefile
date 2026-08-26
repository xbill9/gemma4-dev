# Configuration - tpu.env is the single source of truth; server.py and mcp-run.sh read
# the same file. Override per-invocation with e.g. `make status ZONE=europe-west4-a`.
-include tpu.env

PROJECT_ID ?= $(GOOGLE_CLOUD_PROJECT)
REGION ?= $(GOOGLE_CLOUD_REGION)
ZONE ?= $(GOOGLE_CLOUD_ZONE)
MODEL_PATH ?= $(MODEL_NAME)
BUCKET_NAME ?= $(PROJECT_ID)-bucket
# The hand-provisioned tpu-vm is named after this rig directory, matching the RESOURCE_ID
# default server.py derives the same way. Set SERVICE_NAME in tpu.env (or on the command
# line) to point these targets at a VM whose name predates that, e.g.
#   make status SERVICE_NAME=tpu-2B-v6e8-devops-agent
SERVICE_NAME ?= $(notdir $(CURDIR))
# Key this rig's MCP server is registered under in the client, and the name server.py
# advertises. It prefixes every tool (mcp__$(MCP_SERVER_NAME)__find_tpu), so it has to be
# unique across the sibling rigs — they used to share "tpu-devops". Same derivation as
# RIG_NAME in server.py; tpu.env or the command line override it.
MCP_SERVER_NAME ?= $(notdir $(CURDIR))

install:
	pip install -r requirements.txt

run:
	python server.py

test:
	python test_agent.py

lint:
	ruff check .
	ruff format --check .
	mypy .

format:
	ruff format .
	ruff check --fix .

# Regenerates GemmaTools.md from the @mcp.tool() decorators in server.py
tools:
	python generate_tools_doc.py

# Writes .mcp.json registering this rig's server under $(MCP_SERVER_NAME). Idempotent, and
# it merges into an existing .mcp.json rather than replacing it, so other servers in the
# file survive. .mcp.json is gitignored at the monorepo root; mcp_config.json is the
# committed example of the same entry.
mcp-config:
	@python3 -c 'import json,sys; \
path, name, cmd = sys.argv[1:4]; \
config = json.load(open(path)) if __import__("os").path.exists(path) else {}; \
config.setdefault("mcpServers", {})[name] = {"command": cmd, "args": [], "env": {}}; \
json.dump(config, open(path, "w"), indent=2); \
open(path, "a").write("\n"); \
print(f"✅ registered MCP server {name!r} in {path}")' \
		"$(CURDIR)/.mcp.json" "$(MCP_SERVER_NAME)" "$(CURDIR)/mcp-run.sh"

clean:
	rm -rf __pycache__
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.log" -exec rm -rf {} +

# Provisioning model for the make path. These are gcloud's own SCREAMING_CASE values for
# `compute instances create`: STANDARD, SPOT, FLEX_START, RESERVATION_BOUND.
#
# UNLIKE THE SIBLING RIG, this is not a second vocabulary for a second path. There, the
# Makefile spoke `tpu-vm` while the MCP tools spoke `queued-resources`, so `make status`
# described a different TPU node from the one the agent had provisioned and the two could
# never see each other. Here both call `gcloud compute instances`, so a VM made by
# `make deploy-tpu` is the same object `list_tpu_instances` returns.
TPU_PROVISIONING_MODEL ?= STANDARD
MACHINE_TYPE ?= ct6e-standard-8t
IMAGE_FAMILY ?= ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e
IMAGE_PROJECT ?= ubuntu-os-accelerator-images
BOOT_DISK_SIZE_GB ?= 200
MAX_RUN_DURATION ?= 4h
REQUEST_VALID_FOR ?= 2h

# An eight-chip v6e slice as a single Compute Engine instance (ct6e-standard-8t is one
# host, 360 vCPU / 1440 GB). --maintenance-policy=TERMINATE is required (a TPU instance
# cannot live-migrate) and --scopes=cloud-platform is what lets the startup script read the
# HF token out of Secret Manager at boot.
#
# THIS BILLS EIGHT CHIPS. Catalog rates are per chip-hour, so a 4h run here costs what a
# 32h run cost on the single-chip fork. `make cost` (estimate_deployment_cost) prices the
# whole slice; read it before you create.
deploy-tpu:
	@echo "🚀 Creating $(TPU_PROVISIONING_MODEL) $(MACHINE_TYPE) instance ($(SERVICE_NAME)) in $(ZONE)..."
	gcloud compute instances create $(SERVICE_NAME) \
		--project=$(PROJECT_ID) \
		--zone=$(ZONE) \
		--machine-type=$(MACHINE_TYPE) \
		--image-family=$(IMAGE_FAMILY) \
		--image-project=$(IMAGE_PROJECT) \
		--maintenance-policy=TERMINATE \
		--boot-disk-size=$(BOOT_DISK_SIZE_GB)GB \
		--scopes=cloud-platform \
		--provisioning-model=$(TPU_PROVISIONING_MODEL) \
		--max-run-duration=$(MAX_RUN_DURATION) \
		--instance-termination-action=DELETE \
		--labels=rig=$(SERVICE_NAME),purpose=make

# Spot: discounted, preempted whenever the capacity is wanted back. Note that in us-east5
# v6e spot lists DEARER than flex-start — check before assuming this is the cheap one.
deploy-tpu-spot:
	@$(MAKE) deploy-tpu TPU_PROVISIONING_MODEL=SPOT

# On-demand: full price, no preemption. Same as plain `make deploy-tpu`.
deploy-tpu-ondemand:
	@$(MAKE) deploy-tpu TPU_PROVISIONING_MODEL=STANDARD

# Flex-start via Dynamic Workload Scheduler. The sibling rig's Makefile has no equivalent —
# gcloud's tpu-vm path offers no flex-start at all, so it was reachable only through the MCP
# tools' Queued Resource path. Blocks until capacity is granted or the request expires.
deploy-tpu-flex:
	@$(MAKE) deploy-tpu TPU_PROVISIONING_MODEL=FLEX_START \
		EXTRA_FLAGS="--request-valid-for-duration=$(REQUEST_VALID_FOR)"

# Note: Running the container requires SSHing into the TPU VM.
# This target provides the command for reference.
run-container:
	@echo "🛠 Run this command on the TPU VM to start vLLM:"
	@echo "sudo docker run -t --rm --name vllm-gemma4 --privileged --net=host \\"
	@echo "    -v /dev/shm:/dev/shm --shm-size 10gb \\"
	@echo "    -e HF_HOME=/dev/shm \\"
	@echo "    -e HF_TOKEN=\$$HF_TOKEN \\"
	@echo "    vllm/vllm-tpu:nightly \\"
	@echo "    vllm serve $(MODEL_PATH) \\"
	@echo "    --max-model-len $(MAX_MODEL_LEN) \\"
	@echo "    --tensor-parallel-size $(TENSOR_PARALLEL_SIZE) \\"
	@echo "    --disable_chunked_mm_input \\"
	@echo "    --max_num_batched_tokens 4096 \\"
	@echo "    --enable-auto-tool-choice \\"
	@echo "    --tool-call-parser gemma4 \\"
	@echo "    --reasoning-parser gemma4"

# Deploy the vLLM inference stack
deploy: deploy-tpu

# Destroy the vLLM inference stack
destroy-tpu:
	@echo "🗑️  Destroying TPU VM $(SERVICE_NAME) in $(ZONE)..."
	gcloud compute tpus tpu-vm delete $(SERVICE_NAME) \
		--project=$(PROJECT_ID) \
		--zone=$(ZONE) \
		--quiet

destroy: destroy-tpu

# Check status of the TPU VM
status:
	@echo "🔍 Checking status of TPU VM $(SERVICE_NAME) in $(ZONE)..."
	gcloud compute tpus tpu-vm describe $(SERVICE_NAME) \
		--project=$(PROJECT_ID) \
		--zone=$(ZONE)

# Get the IP address of the deployed TPU VM
endpoint:
	@gcloud compute tpus tpu-vm describe $(SERVICE_NAME) \
		--project=$(PROJECT_ID) \
		--zone=$(ZONE) \
		--format='value(networkEndpoints[0].ipAddress)'

# Run the performance benchmarking suite
benchmark:
	@echo "📊 Discovering endpoint for $(SERVICE_NAME)..."
	@IP=$$(gcloud compute tpus tpu-vm describe $(SERVICE_NAME) --project=$(PROJECT_ID) --zone=$(ZONE) --format='value(networkEndpoints[0].accessConfig.externalIp)'); \
	if [ -z "$$IP" ]; then IP=$$(gcloud compute tpus tpu-vm describe $(SERVICE_NAME) --project=$(PROJECT_ID) --zone=$(ZONE) --format='value(networkEndpoints[0].ipAddress)'); fi; \
	if [ -z "$$IP" ]; then echo "❌ Error: Could not find IP for $(SERVICE_NAME)"; exit 1; fi; \
	echo "🚀 Starting benchmark against http://$$IP:8000"; \
	python benchmarking_suite.py --url http://$$IP:8000 --model $(MODEL_PATH)

# --- Profiling: xprof / TensorBoard -------------------------------------------------------
# vLLM's own /start_profile is ABSENT from vllm-tpu:nightly (VLLM_TORCH_PROFILER_DIR is not a
# known variable and the route 404s, measured 2026-08-25). capture_profile.sh injects an
# in-process trigger instead; see profiler_sidecar.py.
#
# `profile` RESTARTS the engine, which costs ~7 min of recompilation on this checkpoint.
TRACE_DIR ?= $(lastword $(sort $(wildcard benchmarks/runs/*xprof*)))

profile:
	@echo "📈 Capturing an xprof trace (restarts the engine — ~7 min recompile)..."
	./capture_profile.sh

trace-analyze:
	@if [ -z "$(TRACE_DIR)" ]; then echo "❌ No benchmarks/runs/*xprof* dir. Run 'make profile' first."; exit 1; fi
	@echo "🔬 Analysing $(TRACE_DIR)"
	python3 analyze_trace.py $(TRACE_DIR)

tensorboard:
	@if [ -z "$(TRACE_DIR)" ]; then echo "❌ No benchmarks/runs/*xprof* dir. Run 'make profile' first."; exit 1; fi
	@echo "🌐 TensorBoard on http://localhost:6006  (Ctrl-C to stop)"
	tensorboard --logdir $(TRACE_DIR)

xprof:
	@if [ -z "$(TRACE_DIR)" ]; then echo "❌ No benchmarks/runs/*xprof* dir. Run 'make profile' first."; exit 1; fi
	@echo "🌐 xprof on http://localhost:8791  (Ctrl-C to stop)"
	xprof --logdir $(TRACE_DIR)

PROMPT ?= "What is Site Reliability Engineering?"

# Query the vLLM model via the TPU VM endpoint (assuming it's running)
query:
	@echo "🔍 Querying $(SERVICE_NAME) with prompt: \"$(PROMPT)\"..."
	@IP=$$(gcloud compute tpus tpu-vm describe $(SERVICE_NAME) --project=$(PROJECT_ID) --zone=$(ZONE) --format='value(networkEndpoints[0].accessConfig.externalIp)'); \
	if [ -z "$$IP" ]; then IP=$$(gcloud compute tpus tpu-vm describe $(SERVICE_NAME) --project=$(PROJECT_ID) --zone=$(ZONE) --format='value(networkEndpoints[0].ipAddress)'); fi; \
	curl -s -X POST "http://$$IP:8000/v1/completions" \
		-H "Content-Type: application/json" \
		-d '{"model": "$(MODEL_PATH)", "prompt": "$(PROMPT)", "max_tokens": 128, "temperature": 0.2}' | python3 -m json.tool

.PHONY: install run test lint format tools mcp-config clean deploy-tpu deploy-tpu-spot deploy-tpu-ondemand deploy destroy-tpu destroy status endpoint benchmark run-container query profile trace-analyze tensorboard xprof help

help:
	@echo "Make targets:"
	@echo ""
	@echo "  install        - Install Python dependencies"
	@echo "  run            - Start the MCP server on stdio"
	@echo "  test           - Run the test suite (unittest)"
	@echo "  lint           - ruff check + ruff format --check + mypy"
	@echo "  format         - Apply ruff formatting and autofixes"
	@echo "  tools          - Regenerate GemmaTools.md from server.py"
	@echo "  mcp-config     - Write .mcp.json registering this server as $(MCP_SERVER_NAME)"
	@echo "  clean          - Remove caches and logs"
	@echo ""
	@echo "  deploy-tpu     - Create the TPU v6e-8 instance ($(SERVICE_NAME) in $(ZONE), $(TPU_PROVISIONING_MODEL))"
	@echo "  deploy-tpu-spot     - Same, as a preemptible Spot VM (discounted, no run limit)"
	@echo "  deploy-tpu-ondemand - Same, at standard on-demand pricing"
	@echo "  destroy-tpu    - Delete the TPU v6e-8 instance"
	@echo "  status         - Describe the TPU instance"
	@echo "  endpoint       - Print the TPU instance IP"
	@echo "  run-container  - Print the docker run command to paste on the TPU VM"
	@echo "  benchmark      - Run benchmarking_suite.py against the live endpoint"
	@echo "  query          - Send one completion request (override with PROMPT=\"...\")"
	@echo ""
	@echo "For the MCP tool catalog see GemmaTools.md, or call the get_help tool."
